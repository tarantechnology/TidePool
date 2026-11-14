"""HTTP API for the audio inference service.

Endpoints
---------
POST /separate       multipart ``audio`` + ``metadata`` JSON string. This is the contract
                     Person 4's backend already implements (INTEGRATION.md). Response is a
                     superset of ``{"stems": [{"name", "audio_base64"}]}``.
POST /v1/separate    JSON body with ``audio_ref`` instead of uploaded bytes (proposed).
POST /v1/route       multipart, same as /separate: the routing decision only (dry run).
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
from .audio_io import DecodedAudio, decode, encode_wav_pcm16, encode_wav_float32
from .config import Settings, load_settings
from .contracts import ModelInfo, RouteInfo, RouteResponse, SeparationMetadata, SeparationRequest, SeparationResponse, SourceInfo, StemResult
from .engine import Engine, write_stem_file
from .errors import ERROR_CODES, AudioServiceError, InvalidMedia, InvalidRequest, StorageFailure
from .media import resolve_ref, to_file_ref
from .explode import DemucsVocals, explode as run_explode, recipe_info
from .stems import STEM_ALIASES
from .stems.vocabulary import requestable

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
        engine.unload()

    app = FastAPI(title="SoundCanvas Audio Inference Service", version=__version__, lifespan=lifespan, description=__doc__)
    app.state.settings, app.state.engine = settings, engine
    demucs = DemucsVocals()

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
                "splitter": engine.splitter_status(), "classifier": engine.classifier_status()}

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
                "tagging": backend.supports_tagging,
                "explode": True,
            },
            "explode": {**recipe_info(), "dialogue_model_installed": demucs.available},
            "input": {"sample_rate": backend.sample_rate, "accepted": "any FFmpeg-decodable audio or video; WAV PCM preferred", "max_duration_seconds": settings.max_duration_seconds, "max_input_bytes": settings.max_input_bytes, "max_prompts": settings.max_prompts},
            "output": {"format": "wav/pcm_f32le (default), wav/pcm_s16le (lossy)", "sample_rate": backend.sample_rate, "channels": "matches source (1 or 2)", "residual_modes": ["mixture_minus_targets", "model", "none"], "return_modes": ["base64", "file"]},
            "defaults": {"predict_spans": settings.predict_spans, "reranking_candidates": settings.reranking_candidates, "stereo_mode": settings.stereo_mode, "chunk_seconds": settings.chunk_seconds, "chunk_batch": settings.chunk_batch, "chunk_workers": settings.chunk_batch, "passes": 1,
                         "refine": {"mode": settings.refine, "p": settings.refine_p, "n_fft": settings.refine_n_fft, "floor": settings.refine_floor, "blend": settings.refine_blend, "threshold": settings.refine_threshold, "residual_threshold": settings.refine_residual_threshold, "gate_db": settings.refine_gate_db},
                         "candidates": settings.candidates, "candidate_select": settings.candidate_select},
            "limits": {"max_passes": settings.max_passes, "max_prompts": settings.max_prompts, "max_duration_seconds": settings.max_duration_seconds},
            "music_stems": _music_stems_capabilities(),
            "routing": {"default_mode": settings.routing_default, "modes": ["auto", "prompted", "music_stems"], "classifier": engine.classifier.name, "music_threshold": settings.router_music_threshold},
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
        return _encode(audio, metadata, engine.separate(audio, metadata, workdir=workdir))

    def _encode(audio: DecodedAudio, metadata: SeparationMetadata, result) -> SeparationResponse:
        request_id = metadata.request_id or uuid.uuid4().hex
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
            decoded = decode(path=upload, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)
            return handle(decoded, meta, workdir)

    multipart_body = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {"type": "object", "required": ["audio"], "properties": {"audio": {"type": "string", "format": "binary"}, "metadata": {"type": "string"}}}}}}}

    async def _read_multipart(request: Request, handle):
        # Detection RLE masks routinely exceed Starlette's 1 MiB default form-field limit.
        # Bound metadata explicitly; uploaded files have a separate streaming size limit.
        limit = int(os.getenv("AUDIO_MAX_METADATA_BYTES", str(64 * 1024 * 1024)))
        async with request.form(max_files=1, max_fields=2, max_part_size=limit) as form:
            audio, metadata = form.get("audio"), form.get("metadata", "{}")
            if not isinstance(audio, StarletteUploadFile) or not isinstance(metadata, str):
                raise InvalidRequest("multipart request requires an audio file and metadata string")
            return await run_in_threadpool(_separate_upload, audio, metadata, handle)

    @app.post("/separate", response_model=SeparationResponse, response_model_exclude_none=True,
              openapi_extra=multipart_body)
    async def separate(request: Request) -> SeparationResponse:
        return await _read_multipart(request, _respond)

    @app.post("/v1/route", response_model=RouteResponse, openapi_extra=multipart_body)
    async def route_upload(request: Request) -> RouteResponse:
        """Dry run: which separator a /separate call with this audio and metadata would use."""
        return await _read_multipart(request, _route)

    def _tag_upload(audio: StarletteUploadFile, max_sounds: int) -> dict[str, Any]:
        from dataclasses import asdict
        from .errors import UnsupportedOperation
        from .tagging import VOCABULARY, WINDOW_SECONDS, pick, windows_of
        import numpy as np
        backend = engine.backend
        if not backend.supports_tagging:
            raise UnsupportedOperation(f"backend {backend.name} cannot tag sounds")
        engine.ensure_loaded()
        with tempfile.TemporaryDirectory(prefix="sam-audio-tag-") as tmp:
            upload = Path(tmp) / ("upload" + (Path(audio.filename or "").suffix.lower() or ".bin"))
            with upload.open("wb") as out:
                while chunk := audio.file.read(1024 * 1024):
                    out.write(chunk)
            decoded = decode(path=upload, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)
        mono = decoded.samples.mean(axis=0).astype(np.float32)
        windows = windows_of(mono, decoded.sample_rate)
        # Silent windows (padding, a quiet tail) would vote for whatever sounds most like silence.
        active = np.array([float(np.sqrt(np.mean(w ** 2))) > 1e-3 for w in windows])
        sims, scale = backend.tag(windows, [prompt for prompt, _, _ in VOCABULARY])
        heard = pick(sims, scale, max_sounds=max_sounds + 4, active=active)
        checked = False
        if engine.classifier.can_tag and not engine._classifier_error:
            from .tagging import verify
            try:
                with engine._classifier_lock:
                    probs = engine.classifier.tag([w for w, a in zip(windows, active) if a], decoded.sample_rate)
                    names = list(engine.classifier.class_names())
                heard = verify(heard, probs, names)
                checked = True
            except Exception as exc:  # noqa: BLE001 - suggestions still work without the second opinion
                log.warning("tag verification failed: %s: %s", type(exc).__name__, exc)
        heard = heard[:max_sounds]
        return {"sounds": [asdict(h) for h in heard], "window_seconds": WINDOW_SECONDS, "windows": len(windows), "verified": checked,
                "is_mock": backend.is_mock, "model": engine.status()}

    def _explode_upload(audio: StarletteUploadFile, metadata: str, split_music: bool = False) -> SeparationResponse:
        """One request, every layer: see explode.py for the recipe."""
        import time
        from .contracts import InferenceOptions
        meta = _parse_metadata(metadata)
        prompts = [p if isinstance(p, str) else p.text for p in meta.prompts]
        if not prompts:
            raise InvalidRequest("at least one sound is required")
        if len(prompts) > settings.max_prompts:
            raise InvalidRequest(f"at most {settings.max_prompts} sounds per request")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="sam-audio-explode-") as tmp:
            workdir = Path(tmp)
            upload = workdir / ("upload" + (Path(audio.filename or "").suffix.lower() or ".bin"))
            with upload.open("wb") as out:
                while chunk := audio.file.read(1024 * 1024):
                    out.write(chunk)
            decoded = decode(path=upload, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, timeout=settings.ffmpeg_timeout)
            engine.ensure_loaded()
            options = meta.options.model_copy(update={"residual": "mixture_minus_targets"})
            notes: list[str] = []
            last_model: dict[str, Any] = {}

            def separate(samples, prompt):
                part = DecodedAudio(samples=samples.astype("float32"), sample_rate=decoded.sample_rate, source_sample_rate=decoded.source_sample_rate, source_channels=decoded.source_channels)
                result = engine.separate(part, SeparationMetadata(prompts=[prompt], options=options), workdir=workdir)
                notes.extend(w for w in result.warnings if w not in notes)
                last_model.update(result.model)
                return next(s.samples for s in result.stems if s.role == "target")

            vocals = None
            if demucs.available and os.getenv("AUDIO_EXPLODE_DEMUCS", "1") != "0":
                vocals = lambda samples: demucs.vocals(samples, decoded.sample_rate)  # noqa: E731
            splitter = None
            if split_music:
                if engine.splitter is None:
                    notes.append("Music was kept whole: no music stem splitter is configured (AUDIO_STEM_SPLITTER).")
                else:
                    from .stems import mixture_consistency
                    import torch, torchaudio

                    def splitter(samples):
                        engine.ensure_splitter_loaded()
                        rate = engine.splitter.sample_rate
                        x = np.asarray(samples, dtype=np.float32)
                        if rate != decoded.sample_rate:
                            x = torchaudio.functional.resample(torch.from_numpy(x), decoded.sample_rate, rate).numpy()
                        with engine._lock:
                            parts = engine.splitter.split(x)
                        if rate != decoded.sample_rate:
                            parts = {k: torchaudio.functional.resample(torch.from_numpy(np.asarray(v, dtype=np.float32)), rate, decoded.sample_rate).numpy() for k, v in parts.items()}
                        n = samples.shape[-1]
                        parts = {k: np.pad(v, ((0, 0), (0, max(0, n - v.shape[-1]))))[:, :n].astype(np.float32) for k, v in parts.items()}
                        parts = {k: v for k, v in parts.items() if k in engine.splitter.stems}
                        return mixture_consistency(np.asarray(samples, dtype=np.float32), parts)
            done = run_explode(decoded.samples, decoded.sample_rate, prompts, separate, vocals, split_music=splitter)
        from .engine import EngineResult, StemArrays
        stems = [StemArrays(layer.name, "target", layer.samples, layer.prompt, layer.name, None) for layer in done.layers]
        stems.append(StemArrays(options.residual_name, "residual", done.residual, None, "background", None))
        status = engine.status()
        model = last_model or {"backend": status["backend"], "model_id": status.get("model_id"), "device": status.get("device"), "predict_spans": False,
                               "reranking_candidates": 1, "stereo_mode": "n/a", "chunked": False, "is_mock": status["is_mock"]}
        model["elapsed_seconds"] = time.monotonic() - started
        result = EngineResult(stems=stems, model=model,
                              warnings=done.warnings + notes + ["Explode recipe: " + ", ".join(f"{l.name} ({l.how})" for l in done.layers)], elapsed_seconds=time.monotonic() - started)
        return _encode(decoded, meta, result)

    @app.post("/explode", response_model=SeparationResponse, response_model_exclude_none=True)
    async def explode_endpoint(audio: UploadFile = File(...), metadata: str = Form("{}"), split_music: bool = Form(False)) -> SeparationResponse:
        """Same request and response shape as /separate; the sounds become layers by the Explode recipe.
        ``split_music`` also splits music layers into their instruments with the stem splitter."""
        return await run_in_threadpool(_explode_upload, audio, metadata, split_music)

    @app.post("/tag")
    async def tag(audio: UploadFile = File(...), max_sounds: int = Form(6)) -> dict[str, Any]:
        """Which sounds are in this audio, strongest first: separation-ready prompts for Explode."""
        return await run_in_threadpool(_tag_upload, audio, max(1, min(int(max_sounds), 10)))

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
