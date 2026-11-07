"""Separation engine: turns a decoded mixture + request metadata into stems.

Responsibilities kept here (backend-agnostic): prompt translation (text, span anchors,
Person 2 visual guidance), stereo strategy, chunked inference with crossfades for long
inputs, residual computation, sample-accurate length preservation and timing metadata.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import DecodedAudio, sanitise, to_mono
from .backends import Backend, create_backend
from .backends.base import Anchor, BackendOptions, BackendPrompt, BackendResult, VisualInput
from .config import Settings
from .contracts import InferenceOptions, PromptSpec, SeparationMetadata, VisualPrompt
from .errors import (
    AudioServiceError,
    InferenceFailure,
    InsufficientGpuMemory,
    InvalidRequest,
    MissingModel,
    UnsupportedOperation,
)
from .media import resolve_ref

log = logging.getLogger("sam_audio_service")


@dataclass
class StemArrays:
    name: str
    role: str  # target | residual
    samples: np.ndarray  # float32 (channels, frames)
    prompt: str | None
    label: str | None
    spans: list[list[float]] | None
    stem_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class EngineResult:
    stems: list[StemArrays]
    model: dict[str, Any]
    warnings: list[str]
    elapsed_seconds: float


def _oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or type(exc).__name__ == "OutOfMemoryError"


class Engine:
    def __init__(self, settings: Settings, backend: Backend | None = None):
        self.settings = settings
        self.backend = backend or create_backend(settings)
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        try:
            self.backend.load()
            self._load_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced through /health and MISSING_MODEL
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")
            raise

    def ensure_loaded(self) -> None:
        if self.backend.loaded:
            return
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001
            if _oom(exc):
                raise InsufficientGpuMemory("not enough GPU memory to load the model", details={"model_id": self.settings.model_id}) from exc
            raise MissingModel(f"model is not available: {self._load_error}", details={"model_id": self.settings.model_id}) from exc

    def status(self) -> dict[str, Any]:
        return {"backend": self.backend.name, "is_mock": self.backend.is_mock, "loaded": self.backend.loaded, "load_error": self._load_error, **self.backend.info()}

    # ------------------------------------------------------------------ translation
    def translate_visual(self, prompt: PromptSpec, shared: dict[str, Any] | None, source_duration: float, warnings: list[str]) -> tuple[VisualInput | None, list[Anchor]]:
        """Translate Person 2's visual guidance into what the model actually consumes.

        Time ranges become ``+`` span anchors. A video + mask video pair becomes a
        VisualInput. Anything else (bboxes, track ids without masks) is accepted but
        cannot steer the model, so it is reported as a warning rather than an error.
        """
        raw = prompt.visual_prompt.model_dump() if prompt.visual_prompt else (shared or None)
        if not raw:
            return None, []
        try:
            visual = VisualPrompt.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"visual_prompt ignored: {exc}")
            return None, []
        anchors: list[Anchor] = [("+", tr.start, min(tr.end, source_duration)) for tr in visual.all_time_ranges() if tr.start < source_duration]
        if not visual.has_model_input():
            ignored = sorted(k for k in raw if k not in {"time_range", "time_ranges"})
            if ignored:
                warnings.append("visual_prompt has no video_ref+mask_video_ref pair; fields " + ", ".join(ignored) + " cannot steer SAM Audio and were ignored" + (" (time ranges were used as span anchors)" if anchors else ""))
            return None, anchors
        if not self.backend.supports_visual:
            warnings.append("backend does not support visual prompting; using text and span anchors only")
            return None, anchors
        video = resolve_ref(visual.video_ref, self.settings)
        mask = resolve_ref(visual.mask_video_ref, self.settings)
        return VisualInput(str(video), str(mask), 0.0, source_duration, visual.object_id, visual.label), anchors

    def _resolve_options(self, options: InferenceOptions) -> tuple[BackendOptions, str, float]:
        if options.model_id and options.model_id != self.settings.model_id:
            raise UnsupportedOperation(f"this service serves {self.settings.model_id!r}; model_id {options.model_id!r} is not loaded", details={"available": [self.settings.model_id]})
        predict = self.settings.predict_spans if options.predict_spans is None else options.predict_spans
        rerank = options.reranking_candidates or self.settings.reranking_candidates
        if predict and not self.backend.supports_span_prediction:
            raise UnsupportedOperation("backend cannot predict spans")
        stereo = options.stereo_mode or self.settings.stereo_mode
        chunk = self.settings.chunk_seconds if options.chunk_seconds is None else options.chunk_seconds
        return BackendOptions(predict_spans=predict, reranking_candidates=rerank, seed=options.seed), stereo, chunk

    # ------------------------------------------------------------------ main entry
    def separate(self, audio: DecodedAudio, metadata: SeparationMetadata) -> EngineResult:
        warnings: list[str] = []
        try:
            prompts = metadata.resolved_prompts(self.settings.max_prompts)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc
        if not prompts:
            raise InvalidRequest("at least one prompt is required")
        if audio.duration > self.settings.max_duration_seconds:
            raise UnsupportedOperation(f"audio is {audio.duration:.1f}s; the limit is {self.settings.max_duration_seconds:.0f}s", details={"max_duration_seconds": self.settings.max_duration_seconds})
        if audio.frames == 0:
            raise InvalidRequest("audio is empty")
        backend_options, stereo_mode, chunk_seconds = self._resolve_options(metadata.options)
        self.ensure_loaded()
        if self.backend.is_mock:
            warnings.append("MOCK backend: stems are band-pass filtered copies of the mixture, not real separation")

        started = time.perf_counter()
        mixture = audio.samples
        stems: list[StemArrays] = []
        model_residuals: list[np.ndarray] = []
        chunked = False
        with self._lock:
            for spec in prompts:
                visual, visual_anchors = self.translate_visual(spec, metadata.visual_prompt, audio.duration, warnings)
                anchors: list[Anchor] = [(a.token, a.start, min(a.end, audio.duration)) for a in spec.anchors if a.start < audio.duration] + visual_anchors
                if anchors and not self.backend.supports_spans:
                    warnings.append("backend ignores span anchors")
                    anchors = []
                prompt = BackendPrompt(text=spec.text, anchors=anchors or None, visual=visual)
                if visual is not None and chunk_seconds and audio.duration > chunk_seconds:
                    warnings.append("chunking disabled for visually prompted request so frames stay aligned")
                channel_inputs = [mixture[c] for c in range(mixture.shape[0])] if stereo_mode == "per_channel" else [to_mono(mixture)]
                outs: list[BackendResult] = []
                for signal in channel_inputs:
                    result, was_chunked = self._separate_signal(signal, prompt, backend_options, 0.0 if visual is not None else chunk_seconds, audio.duration)
                    chunked = chunked or was_chunked
                    outs.append(result)
                target = np.stack([sanitise(o.target) for o in outs]) if stereo_mode == "per_channel" else np.repeat(sanitise(outs[0].target)[None, :], mixture.shape[0], axis=0)
                target = target[:, : audio.frames].astype(np.float32)
                if target.shape[1] < audio.frames:
                    target = np.pad(target, ((0, 0), (0, audio.frames - target.shape[1])))
                spans = [[float(s), float(e)] for s, e in (outs[0].spans or [])] if outs[0].spans is not None else ([[a[1], a[2]] for a in anchors if a[0] == "+"] or None)
                if outs[0].residual is not None:
                    res = np.stack([o.residual for o in outs]) if stereo_mode == "per_channel" else np.repeat(outs[0].residual[None, :], mixture.shape[0], axis=0)
                    model_residuals.append(res[:, : audio.frames].astype(np.float32))
                label = spec.visual_prompt.label if spec.visual_prompt and spec.visual_prompt.label else None
                stems.append(StemArrays(spec.display_name, "target", target, spec.text or None, label or spec.display_name, spans))

        residual_mode = metadata.options.residual
        if residual_mode == "mixture_minus_targets":
            residual = (mixture - sum(s.samples for s in stems)).astype(np.float32)
            stems.append(StemArrays(metadata.options.residual_name, "residual", residual, None, "everything else", None))
        elif residual_mode == "model":
            if len(prompts) != 1 or not model_residuals:
                warnings.append("residual=model needs exactly one prompt and a backend residual; fell back to mixture_minus_targets")
                residual = (mixture - sum(s.samples for s in stems)).astype(np.float32)
            else:
                residual = model_residuals[0]
            stems.append(StemArrays(metadata.options.residual_name, "residual", residual, None, "everything else", None))

        elapsed = time.perf_counter() - started
        info = self.backend.info()
        model = {
            "backend": self.backend.name,
            "model_id": info.get("model_id"),
            "device": info.get("device"),
            "dtype": info.get("dtype"),
            "predict_spans": backend_options.predict_spans,
            "reranking_candidates": backend_options.reranking_candidates,
            "stereo_mode": stereo_mode,
            "chunked": chunked,
            "elapsed_seconds": round(elapsed, 3),
            "is_mock": self.backend.is_mock,
        }
        return EngineResult(stems, model, warnings, elapsed)

    # ------------------------------------------------------------------ chunking
    def _call_backend(self, signal: np.ndarray, prompt: BackendPrompt, options: BackendOptions) -> BackendResult:
        try:
            result = self.backend.separate(signal.astype(np.float32, copy=False), prompt, options)
        except AudioServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _oom(exc):
                raise InsufficientGpuMemory("GPU ran out of memory during inference; try a shorter input, AUDIO_CHUNK_SECONDS or fewer reranking candidates", details={"frames": int(len(signal))}) from exc
            raise InferenceFailure(f"{type(exc).__name__}: {exc}") from exc
        if result.target.shape[-1] != len(signal):
            n = len(signal)
            fixed = np.zeros(n, dtype=np.float32)
            m = min(n, result.target.shape[-1])
            fixed[:m] = result.target[:m]
            result.target = fixed
            if result.residual is not None:
                res = np.zeros(n, dtype=np.float32)
                res[:m] = result.residual[:m]
                result.residual = res
        return result

    def _separate_signal(self, signal: np.ndarray, prompt: BackendPrompt, options: BackendOptions, chunk_seconds: float, duration: float) -> tuple[BackendResult, bool]:
        sr = self.backend.sample_rate
        n = len(signal)
        if not chunk_seconds or duration <= chunk_seconds:
            return self._call_backend(signal, prompt, options), False
        overlap = int(self.settings.chunk_overlap_seconds * sr)
        size = int(chunk_seconds * sr)
        step = size - overlap
        target = np.zeros(n, dtype=np.float64)
        residual = np.zeros(n, dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        spans: list[tuple[float, float]] = []
        has_residual = True
        start = 0
        while start < n:
            end = min(start + size, n)
            if end - start < overlap and start > 0:  # tiny tail: fold into previous window
                break
            piece = signal[start:end]
            local_anchors = None
            if prompt.anchors:
                local_anchors = []
                for token, a_start, a_end in prompt.anchors:
                    s = max(a_start - start / sr, 0.0)
                    e = min(a_end - start / sr, (end - start) / sr)
                    if e > s:
                        local_anchors.append((token, s, e))
                local_anchors = local_anchors or None
            local_prompt = BackendPrompt(prompt.text, local_anchors, prompt.visual)
            out = self._call_backend(piece, local_prompt, options)
            ramp = np.ones(end - start)
            if start > 0:
                ramp[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False)
            if end < n:
                ramp[-overlap:] = np.minimum(ramp[-overlap:], np.linspace(1.0, 0.0, overlap, endpoint=False))
            target[start:end] += out.target * ramp
            if out.residual is None:
                has_residual = False
            else:
                residual[start:end] += out.residual * ramp
            weight[start:end] += ramp
            if out.spans:
                spans += [(s + start / sr, e + start / sr) for s, e in out.spans]
            if end >= n:
                break
            start += step
        weight = np.maximum(weight, 1e-6)
        merged = BackendResult(
            target=(target / weight).astype(np.float32),
            residual=(residual / weight).astype(np.float32) if has_residual else None,
            spans=_merge_spans(spans) if spans else None,
        )
        return merged, True


def _merge_spans(spans: list[tuple[float, float]], gap: float = 0.05) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged]


def write_stem_file(root: Path, request_id: str, stem: StemArrays, data: bytes) -> Path:
    directory = root / request_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem.stem_id}.wav"
    path.write_bytes(data)
    return path
