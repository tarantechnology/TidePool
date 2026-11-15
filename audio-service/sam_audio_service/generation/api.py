"""HTTP routes for music analysis and part generation.

POST /v1/analyze-music   multipart: ``audio`` (the mix, optional when stems are given), any number of
                         ``stem_<name>`` files (``stem_vocals``, ``stem_drums``, ``stem_bass``,
                         ``stem_other``, ``stem_guitar`` ...), ``metadata`` JSON ``{"melody": true}``.
                         Or JSON ``{"audio_ref": ..., "stems": {"vocals": "<ref>", ...}}``.
                         -> {tempo, beats, downbeats, key, chords[{start,end,chord}], sections,
                             melody_midi, vocal_activity, groove, register, space, context, warnings}
POST /v1/generate-part   same inputs; metadata ``{instrument?, prompt?, role?, style?, sections?|interval?,
                         density?, engine?, seed?, level_db?, duck_db?, analysis?, return_mode?}``
                         -> {part, audio (wav, same length/rate as the song), midi_base64, notes,
                             engine (+licence), renderer, analysis, windows, mix, checks, warnings}
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from ..audio_io import decode, encode_wav_float32, encode_wav_pcm16
from ..errors import InvalidMedia, InvalidRequest, StorageFailure
from ..media import resolve_ref, to_file_ref
from .engine import ENGINES, Generator, PartRequest

MAX_STEMS = 12


def plain(obj: Any) -> Any:
    """numpy scalars/arrays -> JSON-native types (analysis and reports are built with numpy)."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class AnalyzeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    melody: bool = True
    request_id: str | None = None


class GenerateMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument: str | None = Field(default=None, max_length=80)
    prompt: str | None = Field(default=None, max_length=400)
    role: str | None = Field(default=None, max_length=40)
    style: str | None = Field(default=None, max_length=40)
    sections: list[str | int] | None = Field(default=None, max_length=32)
    interval: Interval | None = None
    density: float | None = Field(default=None, ge=0, le=1)
    engine: str = "auto"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    level_db: float = Field(default=0.0, ge=-24, le=12)
    duck_db: float = Field(default=3.0, ge=0, le=12)
    humanize: bool = True
    analysis: dict[str, Any] | None = None
    return_mode: str = Field(default="base64", pattern="^(base64|file)$")
    output_format: str = Field(default="pcm_f32le", pattern="^(pcm_f32le|pcm_s16le|flac)$")
    # Render at this rate when the upload is only a low-rate reference (cached analysis, no stems).
    output_sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    include_analysis: bool = True
    audio_checks: bool = False
    request_id: str | None = None

    @field_validator("engine")
    @classmethod
    def _engine(cls, v: str) -> str:
        if v not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        return v


class RefBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    audio_ref: str | None = None
    stems: dict[str, str] | None = None


def _flac(samples: np.ndarray, sr: int, ffmpeg: str = "ffmpeg") -> bytes:
    """16-bit FLAC of the part: a generated layer is mostly silence/sparse, so this is a fraction of a WAV."""
    import subprocess
    proc = subprocess.run([ffmpeg, "-v", "error", "-f", "wav", "-i", "pipe:0", "-c:a", "flac", "-compression_level", "2", "-f", "flac", "pipe:1"],
                          input=encode_wav_pcm16(samples, sr), capture_output=True, check=True, timeout=300)
    return proc.stdout


def register(app: FastAPI, settings, generator: Generator) -> None:
    def decode_file(path: Path):
        return decode(path=path, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)

    def _spool(upload: UploadFile, dest: Path) -> Path:
        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_input_bytes:
                        raise InvalidMedia("audio exceeds AUDIO_MAX_INPUT_BYTES", details={"max_input_bytes": settings.max_input_bytes})
                    out.write(chunk)
        except OSError as exc:
            raise StorageFailure(f"could not spool upload: {exc.strerror or exc}") from exc
        if size == 0:
            raise InvalidMedia(f"upload {upload.filename!r} is empty")
        return dest

    def _align(mix, stems: dict[str, Any]):
        """Decode results -> (mix (C,n) | None, sr, {name: (C,n)}) all at the mix's length."""
        sr = (mix or next(iter(stems.values()))).sample_rate
        n = mix.frames if mix is not None else max(s.frames for s in stems.values())
        out = {}
        for k, s in stems.items():
            x = s.samples
            out[k] = x[:, :n] if x.shape[1] >= n else np.pad(x, ((0, 0), (0, n - x.shape[1])))
        return (mix.samples if mix is not None else None), sr, out

    async def _inputs(request: Request, model):
        ctype = request.headers.get("content-type", "")
        tmp = tempfile.TemporaryDirectory(prefix="gen-req-")
        work = Path(tmp.name)
        if ctype.startswith("application/json"):
            try:
                raw = await request.json()
                body = RefBody.model_validate(raw)
                meta = model.model_validate({k: v for k, v in raw.items() if k not in ("audio_ref", "stems")})
            except (ValidationError, ValueError) as exc:
                raise InvalidRequest("invalid request", details={"errors": json.loads(exc.json()) if isinstance(exc, ValidationError) else str(exc)}) from exc
            if not body.audio_ref and not body.stems:
                raise InvalidRequest("audio_ref or stems is required")
            if len(body.stems or {}) > MAX_STEMS:
                raise InvalidRequest(f"at most {MAX_STEMS} stems")
            mix = await run_in_threadpool(decode_file, resolve_ref(body.audio_ref, settings, workdir=work)) if body.audio_ref else None
            stems = {}
            for name, ref in (body.stems or {}).items():
                stems[name] = await run_in_threadpool(decode_file, resolve_ref(ref, settings, workdir=work))
            return tmp, meta, mix, stems
        limit = int(os.getenv("AUDIO_MAX_METADATA_BYTES", str(64 * 1024 * 1024)))
        async with request.form(max_files=MAX_STEMS + 1, max_fields=MAX_STEMS + 4, max_part_size=limit) as form:
            raw_meta = form.get("metadata", "{}")
            if not isinstance(raw_meta, str):
                raise InvalidRequest("metadata must be a JSON string field")
            try:
                payload = json.loads(raw_meta or "{}")
                meta = model.model_validate(payload)
            except json.JSONDecodeError as exc:
                raise InvalidRequest(f"metadata is not valid JSON: {exc}") from exc
            except ValidationError as exc:
                raise InvalidRequest("invalid metadata", details={"errors": json.loads(exc.json())}) from exc
            mix, stems = None, {}
            for key, value in form.multi_items():
                if not isinstance(value, UploadFile):
                    continue
                suffix = Path(value.filename or "").suffix.lower() or ".bin"
                if key == "audio":
                    p = await run_in_threadpool(_spool, value, work / ("mix" + suffix))
                    mix = await run_in_threadpool(decode_file, p)
                elif key.startswith("stem_") and len(key) > 5:
                    name = key[5:].lower()
                    p = await run_in_threadpool(_spool, value, work / (f"stem_{len(stems)}" + suffix))
                    stems[name] = await run_in_threadpool(decode_file, p)
                else:
                    raise InvalidRequest(f"unexpected file field {key!r} (use 'audio' and 'stem_<name>')")
            if mix is None and not stems:
                raise InvalidRequest("multipart request needs an 'audio' file and/or 'stem_<name>' files")
            return tmp, meta, mix, stems

    @app.post("/v1/analyze-music")
    async def analyze_music(request: Request) -> dict[str, Any]:
        tmp, meta, mix, stems = await _inputs(request, AnalyzeMetadata)
        try:
            mix_arr, sr, stem_arr = _align(mix, stems)
            analysis = await run_in_threadpool(generator.analyze, mix_arr, sr, stem_arr, meta.melody)
            return plain({"request_id": meta.request_id or uuid.uuid4().hex, "analysis": analysis})
        finally:
            tmp.cleanup()

    @app.post("/v1/generate-part")
    async def generate_part(request: Request) -> dict[str, Any]:
        tmp, meta, mix, stems = await _inputs(request, GenerateMetadata)
        try:
            mix_arr, sr, stem_arr = _align(mix, stems)
            if meta.output_sample_rate and meta.output_sample_rate != sr and mix_arr is not None:
                from . import dsp
                mix_arr = await run_in_threadpool(dsp.resample, mix_arr, sr, meta.output_sample_rate)
                stem_arr = {k: dsp.resample(v, sr, meta.output_sample_rate) for k, v in stem_arr.items()}
                sr = meta.output_sample_rate
            if mix_arr is not None and mix_arr.shape[0] == 1:
                mix_arr = np.repeat(mix_arr, 2, axis=0)
            req = PartRequest(instrument=meta.instrument, prompt=meta.prompt, role=meta.role, style=meta.style, sections=meta.sections,
                              interval=(meta.interval.start, meta.interval.end) if meta.interval else None, density=meta.density, engine=meta.engine,
                              seed=meta.seed, level_db=meta.level_db, duck_db=meta.duck_db, humanize=meta.humanize, analysis=meta.analysis,
                              audio_checks=meta.audio_checks)
            res = await run_in_threadpool(generator.generate, mix_arr, sr, stem_arr, req)
            request_id = meta.request_id or uuid.uuid4().hex
            if meta.output_format == "flac":
                wav = await run_in_threadpool(_flac, res.audio, res.sample_rate, settings.ffmpeg)
                fmt = "flac/s16"
            else:
                wav = (encode_wav_float32 if meta.output_format == "pcm_f32le" else encode_wav_pcm16)(res.audio, res.sample_rate)
                fmt = "wav/" + meta.output_format
            audio: dict[str, Any] = {"format": fmt, "sample_rate": res.sample_rate, "channels": int(res.audio.shape[0]),
                                     "sample_count": int(res.audio.shape[1]), "duration": round(res.audio.shape[1] / res.sample_rate, 6), "start_time": 0.0}
            if meta.return_mode == "file":
                out_dir = Path(settings.output_root) / request_id
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / ("part.flac" if fmt.startswith("flac") else "part.wav")).write_bytes(wav)
                    if res.midi:
                        (out_dir / "part.mid").write_bytes(res.midi)
                except OSError as exc:
                    raise StorageFailure(f"could not write part: {exc}") from exc
                audio["audio_ref"] = to_file_ref(out_dir / ("part.flac" if fmt.startswith("flac") else "part.wav"))
            else:
                audio["audio_base64"] = base64.b64encode(wav).decode("ascii")
            spec = res.spec
            return plain({
                "request_id": request_id,
                "part": {"name": spec.display_name, "kind": "music", "tags": ["generated", spec.family, spec.role], **spec.to_dict()},
                "audio": audio,
                "midi_base64": base64.b64encode(res.midi).decode("ascii") if res.midi else None,
                "notes": [n.to_dict() for n in res.notes],
                "engine": res.engine,
                "renderer": res.renderer,
                "windows": [[round(a, 4), round(b, 4)] for a, b in res.windows],
                "mix": res.mix,
                "checks": res.checks,
                "analysis": res.analysis if meta.include_analysis else None,
                "elapsed_seconds": res.elapsed,
                "warnings": res.warnings,
            })
        finally:
            tmp.cleanup()
