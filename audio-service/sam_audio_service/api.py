"""HTTP API for the audio inference service.

Endpoints
---------
POST /separate       multipart ``audio`` + ``metadata`` JSON string. This is the contract
                     Person 4's backend already implements (INTEGRATION.md). Response is a
                     superset of ``{"stems": [{"name", "audio_base64"}]}``.
POST /v1/separate    JSON body with ``audio_ref`` instead of uploaded bytes (proposed).
POST /v1/route       multipart, same as /separate: the routing decision only (dry run).
POST /v1/words       multipart: word timestamps, fillers, profanity, pauses (faster-whisper).
POST /v1/enhance     multipart: speech enhancement of a dialogue/vocal layer (DeepFilterNet3).
POST /v1/beats       multipart: beats, downbeats and tempo (beat_this; librosa fallback).
GET  /health         liveness + model status.
GET  /capabilities   prompting modes, limits, error codes, contract version.

This service is not an application backend: no uploads, projects, job queue or export.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import ValidationError

from . import CONTRACT_VERSION, __version__
from .audio_io import DecodedAudio, decode, encode_wav_pcm16, encode_wav_float32, resample
from .config import Settings, load_settings
from .contracts import ModelInfo, RouteInfo, RouteResponse, SeparationMetadata, SeparationRequest, SeparationResponse, SourceInfo, StemResult
from .engine import Engine, write_stem_file
from .errors import ERROR_CODES, AudioServiceError, InvalidMedia, InvalidRequest, StorageFailure
from .media import resolve_ref, to_file_ref
from .speech import SpeechTools
from .speech.contracts import BeatsMetadata, BeatsResponse, EnhanceMetadata, EnhanceResponse, WordsMetadata, WordsResponse
from .speech.service import BEATS_SAMPLE_RATE, ENHANCE_SAMPLE_RATE, WORDS_SAMPLE_RATE
from .stems import STEM_ALIASES
from .stems.vocabulary import requestable

log = logging.getLogger("sam_audio_service")


def create_app(settings: Settings | None = None, engine: Engine | None = None, speech: SpeechTools | None = None) -> FastAPI:
    settings = settings or load_settings()
    engine = engine or Engine(settings)
    speech = speech or SpeechTools(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.load_on_startup:
            try:
                engine.load()
            except Exception:  # noqa: BLE001 - keep serving /health; requests return MISSING_MODEL
                log.warning("model did not load on startup; will retry on first request")
            speech.load()
        yield
        engine.unload()
        speech.unload()

    app = FastAPI(title="SoundCanvas Audio Inference Service", version=__version__, lifespan=lifespan, description=__doc__)
    app.state.settings, app.state.engine, app.state.speech = settings, engine, speech

    @app.exception_handler(AudioServiceError)
    async def _service_error(_: Request, exc: AudioServiceError):
        return JSONResponse(exc.to_payload(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        err = InvalidRequest("invalid request", details={"errors": json.loads(json.dumps(exc.errors(), default=str))})
        return JSONResponse(err.to_payload(), status_code=err.status)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(AudioServiceError(f"internal error: {type(exc).__name__}").to_payload(), status_code=500)

    @app.get("/health")
    def health() -> dict[str, Any]:
        status = engine.status()
        return {"status": "ok" if status["loaded"] else "degraded", "service": "sam-audio-service", "version": __version__, "contract_version": CONTRACT_VERSION, "model": status,
                "splitter": engine.splitter_status(), "classifier": engine.classifier_status(), "speech": speech.status()}

    @app.get("/ready")
    def ready():
        status = engine.status()
        return JSONResponse({"status": "ready" if status["loaded"] else "not_ready", "model": status}, status_code=200 if status["loaded"] else 503)

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        backend = engine.backend
        return {
            "service": "sam-audio-service",
            "version": __version__,
            "contract_version": CONTRACT_VERSION,
            "backend": backend.name,
            "is_mock": backend.is_mock,
            "independent_candidates": backend.supports_independent_candidates,
            "model": engine.status(),
            "prompting_modes": {
                "text": True,
                "span_anchors": backend.supports_spans,
                "span_prediction": backend.supports_span_prediction,
                "visual": backend.supports_visual,
            },
            "input": {"sample_rate": backend.sample_rate, "accepted": "any FFmpeg-decodable audio or video; WAV PCM preferred", "max_duration_seconds": settings.max_duration_seconds, "max_input_bytes": settings.max_input_bytes, "max_prompts": settings.max_prompts},
            "output": {"format": "wav/pcm_f32le (default), wav/pcm_s16le (lossy)", "sample_rate": backend.sample_rate, "channels": "matches source (1 or 2)", "residual_modes": ["mixture_minus_targets", "model", "none"], "return_modes": ["base64", "file"]},
            "defaults": {"predict_spans": settings.predict_spans, "reranking_candidates": settings.reranking_candidates, "stereo_mode": settings.stereo_mode, "chunk_seconds": settings.chunk_seconds, "chunk_batch": settings.chunk_batch, "chunk_workers": settings.chunk_batch, "passes": 1,
                         "refine": {"mode": settings.refine, "p": settings.refine_p, "n_fft": settings.refine_n_fft, "floor": settings.refine_floor, "blend": settings.refine_blend, "threshold": settings.refine_threshold, "residual_threshold": settings.refine_residual_threshold, "gate_db": settings.refine_gate_db},
                         "candidates": settings.candidates, "candidate_select": settings.candidate_select},
            "limits": {"max_passes": settings.max_passes, "max_prompts": settings.max_prompts, "max_duration_seconds": settings.max_duration_seconds},
            "music_stems": _music_stems_capabilities(),
            "routing": {"default_mode": settings.routing_default, "modes": ["auto", "prompted", "music_stems"], "classifier": engine.classifier.name, "music_threshold": settings.router_music_threshold},
            **speech.capabilities(),
            "error_codes": ERROR_CODES,
        }

    def _music_stems_capabilities() -> dict[str, Any]:
        splitter = engine.splitter
        if splitter is None:
            return {"available": False, "backend": None, "model_id": None, "license": None, "stems": [], "max_duration_seconds": settings.max_duration_seconds}
        stems = requestable(splitter.stems)
        return {
            "available": True, "backend": splitter.name, "is_mock": splitter.is_mock, "model_id": splitter.info().get("model_id"), "license": splitter.info().get("license"),
            "stems": stems, "aliases": {alias: c for alias, c in STEM_ALIASES.items() if c in stems and alias != c},
            "max_duration_seconds": settings.max_duration_seconds,
        }

    def _parse_metadata(raw: str) -> SeparationMetadata:
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise InvalidRequest(f"metadata is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidRequest("metadata must be a JSON object")
        try:
            return SeparationMetadata.model_validate(payload)
        except ValidationError as exc:
            raise InvalidRequest("invalid metadata", details={"errors": json.loads(exc.json())}) from exc

    def _respond(audio: DecodedAudio, metadata: SeparationMetadata, workdir: Path) -> SeparationResponse:
        request_id = metadata.request_id or uuid.uuid4().hex
        result = engine.separate(audio, metadata, workdir=workdir)
        stems: list[StemResult] = []
        for stem in result.stems:
            wav = (encode_wav_float32 if metadata.options.output_format == "pcm_f32le" else encode_wav_pcm16)(stem.samples, audio.sample_rate)
            item = StemResult(
                name=stem.name, stem_id=stem.stem_id, role=stem.role, prompt=stem.prompt, label=stem.label, start_time=0.0, format="wav/" + metadata.options.output_format,
                duration=round(stem.samples.shape[1] / audio.sample_rate, 6), sample_rate=audio.sample_rate, channels=int(stem.samples.shape[0]),
                sample_count=int(stem.samples.shape[1]), spans=stem.spans, source_asset_id=metadata.asset_id, refinement=stem.refinement, tags=stem.tags,
            )
            if metadata.options.return_mode == "file":
                try:
                    item.audio_ref = to_file_ref(write_stem_file(settings.output_root, request_id, stem, wav))
                except OSError as exc:
                    raise StorageFailure(f"could not write stem: {exc}") from exc
            else:
                item.audio_base64 = base64.b64encode(wav).decode("ascii")
            stems.append(item)
        source = SourceInfo(asset_id=metadata.asset_id, duration=round(audio.duration, 6), sample_rate=audio.sample_rate, channels=audio.channels, sample_count=audio.frames)
        routing = RouteInfo(**result.routing) if result.routing else None
        return SeparationResponse(stems=stems, model=ModelInfo(**result.model), source=source, request_id=request_id, routing=routing, warnings=result.warnings)

    def _route(audio: DecodedAudio, metadata: SeparationMetadata, workdir: Path) -> RouteResponse:
        decision = engine.route(audio, metadata)
        source = SourceInfo(asset_id=metadata.asset_id, duration=round(audio.duration, 6), sample_rate=audio.sample_rate, channels=audio.channels, sample_count=audio.frames)
        return RouteResponse(routing=RouteInfo(**decision.to_dict()), source=source, warnings=decision.warnings)

    def _separate_upload(audio: StarletteUploadFile, metadata: str, handle=_respond):
        meta = _parse_metadata(metadata)
        return _spool(audio, lambda upload, workdir: handle(decode(path=upload, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout), meta, workdir))

    def _spool(audio: StarletteUploadFile, handle):
        """Write the upload to a temp file (bounded), then ``handle(path, workdir)``."""
        # Spool to a real file: MP4/MOV with a trailing moov atom cannot be demuxed from a pipe.
        with tempfile.TemporaryDirectory(prefix="sam-audio-req-") as tmp:
            workdir = Path(tmp)
            suffix = Path(audio.filename or "").suffix.lower() or ".bin"
            upload = workdir / ("upload" + suffix)
            size = 0
            try:
                with upload.open("wb") as out:
                    while chunk := audio.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > settings.max_input_bytes:
                            raise InvalidMedia("audio exceeds AUDIO_MAX_INPUT_BYTES", details={"max_input_bytes": settings.max_input_bytes})
                        out.write(chunk)
            except OSError as exc:  # e.g. ENOSPC on the spool directory: retryable, not a client error
                raise StorageFailure(f"could not spool upload: {exc.strerror or exc}") from exc
            if size == 0:
                raise InvalidMedia("audio upload is empty")
            return handle(upload, workdir)

    multipart_body = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {"type": "object", "required": ["audio"], "properties": {"audio": {"type": "string", "format": "binary"}, "metadata": {"type": "string"}}}}}}}

    async def _read_multipart(request: Request, handle, upload_handler=None):
        # Detection RLE masks routinely exceed Starlette's 1 MiB default form-field limit.
        # Bound metadata explicitly; uploaded files have a separate streaming size limit.
        limit = int(os.getenv("AUDIO_MAX_METADATA_BYTES", str(64 * 1024 * 1024)))
        async with request.form(max_files=1, max_fields=2, max_part_size=limit) as form:
            audio, metadata = form.get("audio"), form.get("metadata", "{}")
            if not isinstance(audio, StarletteUploadFile) or not isinstance(metadata, str):
                raise InvalidRequest("multipart request requires an audio file and metadata string")
            return await run_in_threadpool(upload_handler or _separate_upload, audio, metadata, handle)

    @app.post("/separate", response_model=SeparationResponse, response_model_exclude_none=True,
              openapi_extra=multipart_body)
    async def separate(request: Request) -> SeparationResponse:
        return await _read_multipart(request, _respond)

    @app.post("/v1/route", response_model=RouteResponse, openapi_extra=multipart_body)
    async def route_upload(request: Request) -> RouteResponse:
        """Dry run: which separator a /separate call with this audio and metadata would use."""
        return await _read_multipart(request, _route)

    @app.post("/v1/separate", response_model=SeparationResponse, response_model_exclude_none=True)
    def separate_ref(body: SeparationRequest) -> SeparationResponse:
        with tempfile.TemporaryDirectory(prefix="sam-audio-req-") as tmp:
            workdir = Path(tmp)
            path = resolve_ref(body.audio_ref, settings, workdir=workdir)
            if path.stat().st_size > settings.max_input_bytes:
                raise InvalidMedia("referenced media exceeds AUDIO_MAX_INPUT_BYTES", details={"max_input_bytes": settings.max_input_bytes})
            decoded = decode(path=Path(path), ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)
            return _respond(decoded, SeparationMetadata.model_validate(body.model_dump(exclude={"audio_ref"})), workdir)

    # ------------------------------------------------------------------ speech / rhythm tools
    def _speech_meta(raw: str, model):
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise InvalidRequest(f"metadata is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidRequest("metadata must be a JSON object")
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise InvalidRequest("invalid metadata", details={"errors": json.loads(exc.json())}) from exc

    def _decode(path: Path, rate: int, max_channels: int = 2) -> DecodedAudio:
        return decode(path=path, target_rate=rate, max_channels=max_channels, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)

    def _speech_upload(model, rate: int, max_channels: int, fn):
        def handler(audio: StarletteUploadFile, metadata: str, _handle):
            meta = _speech_meta(metadata, model)
            return _spool(audio, lambda path, workdir: fn(path, _decode(path, rate, max_channels), meta, workdir))
        return handler

    def _words(path: Path, audio: DecodedAudio, meta: WordsMetadata, workdir: Path) -> WordsResponse:
        return WordsResponse(**speech.transcribe(audio, meta, meta.request_id or uuid.uuid4().hex))

    def _enhance(path: Path, audio: DecodedAudio, meta: EnhanceMetadata, workdir: Path) -> EnhanceResponse:
        request_id = meta.request_id or uuid.uuid4().hex
        out, report = speech.enhance(audio, meta)
        rate = ENHANCE_SAMPLE_RATE
        if meta.preserve_sample_rate and audio.source_sample_rate != ENHANCE_SAMPLE_RATE:
            # Same rate and length as the upload: resample back and fit to the original frame count.
            original = _decode(path, audio.source_sample_rate, 2)
            out = resample(out, ENHANCE_SAMPLE_RATE, audio.source_sample_rate, ffmpeg=settings.ffmpeg, timeout=settings.ffmpeg_timeout)
            n = original.frames
            out = out[:, :n] if out.shape[1] >= n else np.pad(out, ((0, 0), (0, n - out.shape[1])))
            rate = audio.source_sample_rate
        wav = (encode_wav_float32 if meta.output_format == "pcm_f32le" else encode_wav_pcm16)(out, rate)
        item = EnhanceResponse(request_id=request_id, format="wav/" + meta.output_format, sample_rate=rate, channels=int(out.shape[0]),
                               sample_count=int(out.shape[1]), duration=round(out.shape[1] / rate, 6), atten_lim_db=report["atten_lim_db"],
                               levels=report["levels"], model=report["model"], source=report["source"], timing=report["timing"], warnings=report["warnings"])
        if meta.return_mode == "file":
            root = settings.output_root.resolve()
            directory = (root / request_id).resolve()
            if directory.parent != root:
                raise InvalidRequest("request_id must be a single path segment")
            try:
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / "enhanced.wav"
                target.write_bytes(wav)
            except OSError as exc:
                raise StorageFailure(f"could not write enhanced audio: {exc}") from exc
            item.audio_ref = to_file_ref(target)
        else:
            item.audio_base64 = base64.b64encode(wav).decode("ascii")
        return item

    def _beats(path: Path, audio: DecodedAudio, meta: BeatsMetadata, workdir: Path) -> BeatsResponse:
        return BeatsResponse(**speech.track_beats(audio, meta, meta.request_id or uuid.uuid4().hex))

    @app.post("/v1/words", response_model=WordsResponse, openapi_extra=multipart_body)
    async def words(request: Request) -> WordsResponse:
        """Word-level transcript with timestamps, plus filler / profanity indices and pauses."""
        return await _read_multipart(request, None, _speech_upload(WordsMetadata, WORDS_SAMPLE_RATE, 1, _words))

    @app.post("/v1/enhance", response_model=EnhanceResponse, response_model_exclude_none=True, openapi_extra=multipart_body)
    async def enhance(request: Request) -> EnhanceResponse:
        """Denoised speech, same length, channels and (by default) sample rate as the upload."""
        return await _read_multipart(request, None, _speech_upload(EnhanceMetadata, ENHANCE_SAMPLE_RATE, 2, _enhance))

    @app.post("/v1/beats", response_model=BeatsResponse, openapi_extra=multipart_body)
    async def beats(request: Request) -> BeatsResponse:
        """Beat and downbeat times (seconds) with tempo and beats per bar."""
        return await _read_multipart(request, None, _speech_upload(BeatsMetadata, BEATS_SAMPLE_RATE, 1, _beats))

    return app


app = create_app()
