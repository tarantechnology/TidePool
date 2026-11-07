"""HTTP API for the audio inference service.

Endpoints
---------
POST /separate       multipart ``audio`` + ``metadata`` JSON string. This is the contract
                     Person 4's backend already implements (INTEGRATION.md). Response is a
                     superset of ``{"stems": [{"name", "audio_base64"}]}``.
POST /v1/separate    JSON body with ``audio_ref`` instead of uploaded bytes (proposed).
GET  /health         liveness + model status.
GET  /capabilities   prompting modes, limits, error codes, contract version.

This service is not an application backend: no uploads, projects, job queue or export.
"""
from __future__ import annotations

import base64
import json
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import CONTRACT_VERSION, __version__
from .audio_io import DecodedAudio, decode, encode_wav_pcm16
from .config import Settings, load_settings
from .contracts import ModelInfo, SeparationMetadata, SeparationRequest, SeparationResponse, SourceInfo, StemResult
from .engine import Engine, write_stem_file
from .errors import ERROR_CODES, AudioServiceError, InvalidMedia, InvalidRequest, StorageFailure
from .media import resolve_ref, to_file_ref

log = logging.getLogger("sam_audio_service")


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or load_settings()
    engine = engine or Engine(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.load_on_startup:
            try:
                engine.load()
            except Exception:  # noqa: BLE001 - keep serving /health; requests return MISSING_MODEL
                log.warning("model did not load on startup; will retry on first request")
        yield
        engine.backend.unload()

    app = FastAPI(title="SoundCanvas Audio Inference Service", version=__version__, lifespan=lifespan, description=__doc__)
    app.state.settings, app.state.engine = settings, engine

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
        return {"status": "ok" if status["loaded"] else "degraded", "service": "sam-audio-service", "version": __version__, "contract_version": CONTRACT_VERSION, "model": status}

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        backend = engine.backend
        return {
            "service": "sam-audio-service",
            "version": __version__,
            "contract_version": CONTRACT_VERSION,
            "backend": backend.name,
            "is_mock": backend.is_mock,
            "model": engine.status(),
            "prompting_modes": {
                "text": True,
                "span_anchors": backend.supports_spans,
                "span_prediction": backend.supports_span_prediction,
                "visual": backend.supports_visual,
            },
            "input": {"sample_rate": backend.sample_rate, "accepted": "any FFmpeg-decodable audio or video; WAV PCM preferred", "max_duration_seconds": settings.max_duration_seconds, "max_input_bytes": settings.max_input_bytes, "max_prompts": settings.max_prompts},
            "output": {"format": "wav/pcm_s16le", "sample_rate": backend.sample_rate, "channels": "matches source (1 or 2)", "residual_modes": ["mixture_minus_targets", "model", "none"], "return_modes": ["base64", "file"]},
            "defaults": {"predict_spans": settings.predict_spans, "reranking_candidates": settings.reranking_candidates, "stereo_mode": settings.stereo_mode, "chunk_seconds": settings.chunk_seconds},
            "error_codes": ERROR_CODES,
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
            wav = encode_wav_pcm16(stem.samples, audio.sample_rate)
            item = StemResult(
                name=stem.name, stem_id=stem.stem_id, role=stem.role, prompt=stem.prompt, label=stem.label, start_time=0.0,
                duration=round(stem.samples.shape[1] / audio.sample_rate, 6), sample_rate=audio.sample_rate, channels=int(stem.samples.shape[0]),
                sample_count=int(stem.samples.shape[1]), spans=stem.spans, source_asset_id=metadata.asset_id,
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
        return SeparationResponse(stems=stems, model=ModelInfo(**result.model), source=source, request_id=request_id, warnings=result.warnings)

    @app.post("/separate", response_model=SeparationResponse, response_model_exclude_none=True)
    def separate(audio: UploadFile = File(...), metadata: str = Form("{}")) -> SeparationResponse:
        meta = _parse_metadata(metadata)
        # Spool to a real file: MP4/MOV with a trailing moov atom cannot be demuxed from a pipe.
        with tempfile.TemporaryDirectory(prefix="sam-audio-req-") as tmp:
            workdir = Path(tmp)
            suffix = Path(audio.filename or "").suffix.lower() or ".bin"
            upload = workdir / ("upload" + suffix)
            size = 0
            with upload.open("wb") as out:
                while chunk := audio.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_input_bytes:
                        raise InvalidMedia("audio exceeds AUDIO_MAX_INPUT_BYTES", details={"max_input_bytes": settings.max_input_bytes})
                    out.write(chunk)
            if size == 0:
                raise InvalidMedia("audio upload is empty")
            decoded = decode(path=upload, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)
            return _respond(decoded, meta, workdir)

    @app.post("/v1/separate", response_model=SeparationResponse, response_model_exclude_none=True)
    def separate_ref(body: SeparationRequest) -> SeparationResponse:
        with tempfile.TemporaryDirectory(prefix="sam-audio-req-") as tmp:
            workdir = Path(tmp)
            path = resolve_ref(body.audio_ref, settings, workdir=workdir)
            if path.stat().st_size > settings.max_input_bytes:
                raise InvalidMedia("referenced media exceeds AUDIO_MAX_INPUT_BYTES", details={"max_input_bytes": settings.max_input_bytes})
            decoded = decode(path=Path(path), ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)
            return _respond(decoded, SeparationMetadata.model_validate(body.model_dump(exclude={"audio_ref"})), workdir)

    return app


app = create_app()
