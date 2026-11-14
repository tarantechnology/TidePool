"""Separation engine: turns a decoded mixture + request metadata into stems.

Responsibilities kept here (backend-agnostic): prompt translation (text, span anchors,
Person 2 visual guidance), stereo strategy, chunked inference with crossfades for long
inputs, residual computation, sample-accurate length preservation and timing metadata.

Multi-pass ("nested") separation: SAM Audio sometimes only attenuates the described sound,
leaving a quieter copy in the residual. With ``options.passes = n`` each prompt is rerun on
its own residual chain (mixture minus that prompt's cumulative target) up to ``n`` times and
the extra targets are summed, so ``sum(stems) == mixture`` still holds exactly.

Long inputs are cut into overlapping windows and handed to the backend ``AUDIO_CHUNK_BATCH``
windows at a time (``Backend.separate_many``). Windows are never run on threads: the model
is shared and not re-entrant, so parallelism belongs in the batch dimension. See
``docs/chunk_concurrency.md``.
"""
from __future__ import annotations

import logging
import sys
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
from .candidates import choose, choose_residual
from .config import Settings
from .contracts import InferenceOptions, PromptSpec, SeparationMetadata
from .refine import mask_refine
from .suppression import suppress_target
from .errors import (
    AudioServiceError,
    InferenceFailure,
    InsufficientGpuMemory,
    InvalidRequest,
    MissingModel,
    UnsupportedOperation,
    is_out_of_memory,
)
from .visual_prompts import translate_visual_prompt

log = logging.getLogger("sam_audio_service")

# A pass is "negligible" when it extracts less than this fraction of pass 1's energy (-30 dB).
NEGLIGIBLE_PASS_RATIO = 10 ** (-30 / 10)
ENERGY_FLOOR_DB = -120.0


def _energy(samples: np.ndarray) -> float:
    return float(np.sum(np.square(samples, dtype=np.float64)))


def _spectral_similarity(a: np.ndarray, b: np.ndarray, n_fft: int = 2048) -> float:
    """Cosine similarity of the long-term magnitude spectra of two (C, T) signals (mono)."""
    from .refine import stft
    def profile(x):
        mono = np.asarray(x, dtype=np.float64).mean(axis=0) if x.ndim == 2 else np.asarray(x, dtype=np.float64)
        if mono.size < n_fft:
            mono = np.pad(mono, (0, n_fft - mono.size))
        return np.abs(stft(mono, n_fft)).mean(axis=0)
    pa, pb = profile(a), profile(b)
    denom = float(np.linalg.norm(pa) * np.linalg.norm(pb))
    return float(np.dot(pa, pb) / denom) if denom > 0 else 0.0


def _energy_db(energy: float, reference: float) -> float:
    if energy <= 0 or reference <= 0:
        return ENERGY_FLOOR_DB
    return round(max(ENERGY_FLOOR_DB, 10 * float(np.log10(energy / reference))), 3)


@dataclass
class StemArrays:
    name: str
    role: str  # target | residual
    samples: np.ndarray  # float32 (channels, frames)
    prompt: str | None
    label: str | None
    spans: list[list[float]] | None
    stem_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    refinement: dict[str, Any] | None = None  # multi-pass bookkeeping for target stems


@dataclass
class EngineResult:
    stems: list[StemArrays]
    model: dict[str, Any]
    warnings: list[str]
    elapsed_seconds: float


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
            if is_out_of_memory(exc):
                raise InsufficientGpuMemory("not enough GPU memory to load the model", details={"model_id": self.settings.model_id}) from exc
            raise MissingModel(f"model is not available: {self._load_error}", details={"model_id": self.settings.model_id}) from exc

    def status(self) -> dict[str, Any]:
        return {"backend": self.backend.name, "is_mock": self.backend.is_mock, "loaded": self.backend.loaded, "load_error": self._load_error, **self.backend.info()}

    # ------------------------------------------------------------------ translation
    def translate_visual(self, prompt: PromptSpec, shared: dict[str, Any] | None, metadata: SeparationMetadata, source_duration: float, warnings: list[str], workdir: Path | None = None) -> tuple[VisualInput | None, list[Anchor]]:
        """Translate Person 2's visual guidance into what the model actually consumes.

        Explicit sound time ranges become ``+`` anchors; visible selection intervals do not. Masks plus a
        reference to the original video become a VisualInput. Anything else is accepted
        but cannot steer the model, so it is reported as a warning rather than an error.
        """
        raw = prompt.visual_prompt.model_dump(exclude_none=True) if prompt.visual_prompt else shared
        try:
            translation = translate_visual_prompt(raw, source=metadata.source, source_duration=source_duration, settings=self.settings, backend_supports_visual=self.backend.supports_visual, workdir=workdir)
        except AudioServiceError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise InvalidRequest(f"visual_prompt is malformed: {type(exc).__name__}: {exc}") from exc
        warnings.extend(translation.warnings)
        return translation.visual, translation.anchors

    def _resolve_options(self, options: InferenceOptions) -> tuple[BackendOptions, str, float]:
        if options.passes is not None and not 1 <= options.passes <= self.settings.max_passes:
            raise InvalidRequest(f"options.passes must be between 1 and {self.settings.max_passes} (AUDIO_MAX_PASSES)", details={"max_passes": self.settings.max_passes})
        if options.model_id and options.model_id != self.settings.model_id:
            raise UnsupportedOperation(f"this service serves {self.settings.model_id!r}; model_id {options.model_id!r} is not loaded", details={"available": [self.settings.model_id]})
        predict = self.settings.predict_spans if options.predict_spans is None else options.predict_spans
        rerank = options.reranking_candidates or self.settings.reranking_candidates
        if predict and not self.backend.supports_span_prediction:
            raise UnsupportedOperation("backend cannot predict spans")
        stereo = options.stereo_mode or self.settings.stereo_mode
        chunk = self.settings.chunk_seconds if options.chunk_seconds is None else options.chunk_seconds
        if chunk and chunk <= 2 * self.settings.chunk_overlap_seconds:
            raise InvalidRequest(f"options.chunk_seconds must exceed {2 * self.settings.chunk_overlap_seconds:g}s (twice the crossfade overlap) or be 0", details={"chunk_overlap_seconds": self.settings.chunk_overlap_seconds})
        k = options.candidates or self.settings.candidates
        if not 1 <= k <= 8:
            raise InvalidRequest("options.candidates must be between 1 and 8")
        select = options.candidate_select or self.settings.candidate_select
        return BackendOptions(predict_spans=predict, reranking_candidates=rerank, seed=options.seed, candidates=k, candidate_select=select), stereo, chunk

    def _resolve_refine(self, options: InferenceOptions) -> dict[str, Any] | None:
        """Effective mask-refinement parameters, or None when refinement is off."""
        s = self.settings
        mode = options.refine or s.refine
        if mode != "mask":
            return None
        pick = lambda value, default: default if value is None else value  # noqa: E731
        return {
            "mode": "mask",
            "p": float(pick(options.refine_p, s.refine_p)),
            "n_fft": int(pick(options.refine_n_fft, s.refine_n_fft)),
            "floor": float(pick(options.refine_floor, s.refine_floor)),
            "blend": float(pick(options.refine_blend, s.refine_blend)),
            "threshold": float(pick(options.refine_threshold, s.refine_threshold)),
            "residual_threshold": float(pick(options.refine_residual_threshold, s.refine_residual_threshold)),
            "gate_db": float(pick(options.refine_gate_db, s.refine_gate_db)),
        }

    def _resolve_suppression(self, options: InferenceOptions) -> dict[str, Any] | None:
        mode = options.suppression or self.settings.suppression
        if mode == 'off':
            return None
        if self.backend.name != 'audiosep':
            raise UnsupportedOperation('Conservative suppression requires AudioSep')
        strength = self.settings.suppression_strength if options.suppression_strength is None else options.suppression_strength
        return {'mode':'conservative', 'strength':strength, 'mask_threshold':.85,
                'activity_low_db':-36., 'activity_high_db':-24., 'n_fft':2048}

    @staticmethod
    def _refine_target(mixture: np.ndarray, target: np.ndarray, cfg: dict[str, Any], stereo_mode: str) -> np.ndarray:
        kwargs = {k: cfg[k] for k in ("p", "n_fft", "floor", "blend", "threshold", "residual_threshold", "gate_db")}
        if stereo_mode == "per_channel":
            return np.concatenate([mask_refine(mixture[c : c + 1], target[c : c + 1], **kwargs) for c in range(mixture.shape[0])])
        return mask_refine(mixture, target, **kwargs)

    # ------------------------------------------------------------------ main entry
    def separate(self, audio: DecodedAudio, metadata: SeparationMetadata, workdir: Path | None = None) -> EngineResult:
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
        batch = metadata.options.chunk_batch or self.settings.chunk_batch
        passes = metadata.options.passes or 1
        suppression_cfg = self._resolve_suppression(metadata.options)
        # The conservative AudioSep path already rebuilds a source-phase mask.
        # Applying the older mask again would alter the tested processing.
        refine_cfg = None if suppression_cfg else self._resolve_refine(metadata.options)
        k_cands = backend_options.candidates
        if (batch > 1 or k_cands > 1) and backend_options.seed is not None:
            warnings.append(
                f"seeded inference is reproducible for AUDIO_CHUNK_BATCH={batch} and candidates={k_cands}; "
                "a different chunk batch size or candidate count draws different noise"
            )
        if self.backend.is_mock:
            warnings.append("MOCK backend: stems are band-pass filtered copies of the mixture, not real separation")

        started = time.perf_counter()
        mixture = audio.samples
        stems: list[StemArrays] = []
        model_residuals: list[np.ndarray] = []
        chunked = False
        oom_retry = False
        effective_batch = batch
        max_passes_run = 1
        candidates_ignored_warned = False
        with self._lock:
            self.ensure_loaded()  # under the lock so concurrent first requests load the model once
            torch = sys.modules.get('torch')
            cuda = torch.cuda if torch is not None and torch.cuda.is_available() and str(self.backend.info().get('device','')).startswith('cuda') else None
            if cuda: cuda.reset_peak_memory_stats()
            for spec in prompts:
                visual, visual_anchors = self.translate_visual(spec, metadata.visual_prompt, metadata, audio.duration, warnings, workdir)
                raw_visual=spec.visual_prompt.model_dump(exclude_none=True) if spec.visual_prompt else metadata.visual_prompt
                if raw_visual and (raw_visual.get('selection') or raw_visual.get('kind')=='visual_selection' or raw_visual.get('mask_video_ref')) and visual is None:
                    raise InvalidRequest('Visual guidance could not be used. Try text plus explicit sound time anchors. ' + '; '.join(warnings[-3:]))
                if not spec.text and visual is None:
                    raise InvalidRequest("prompt has no text and its visual guidance could not be used for separation: " + ("; ".join(warnings[-3:]) or "no visual guidance given"))
                if any(a.end > audio.duration + 1e-6 for a in spec.anchors):
                    raise InvalidRequest('anchor extends beyond the source audio timeline')
                anchors: list[Anchor] = [(a.token, a.start, min(a.end, audio.duration)) for a in spec.anchors if a.start < audio.duration] + visual_anchors
                if anchors and not self.backend.supports_spans:
                    warnings.append("backend ignores span anchors")
                    anchors = []
                prompt = BackendPrompt(text=spec.text, anchors=anchors or None, visual=visual)
                if backend_options.reranking_candidates > 1:
                    info=self.backend.info()
                    if self.backend.is_mock or not (info.get('text_ranker_loaded') or (visual is not None and info.get('visual_ranker_loaded'))):
                        raise UnsupportedOperation('Candidate ranking is not enabled on the audio service. Use one candidate or enable the pretrained ranker.')
                    if visual is not None and not info.get('visual_ranker_loaded'):
                        warnings.append('Visual ranker unavailable; candidates are ranked using the sound description.')
                if visual is not None and chunk_seconds and audio.duration > chunk_seconds:
                    warnings.append("chunking disabled for visually prompted request so frames stay aligned")
                # Pass 1 separates the mixture; pass k >= 2 reruns the SAME prompt (text, anchors,
                # visual) on this prompt's residual chain, mixture - sum(previous passes).
                cumulative = np.zeros_like(mixture, dtype=np.float32)
                source = mixture
                mixture_energy = _energy(mixture)
                first_energy = 0.0
                per_pass_db: list[float] = []
                pass_spans: list[tuple[float, float]] = []
                saw_spans = False
                stopped_early = False
                stop_reason: str | None = None
                first_target: np.ndarray | None = None
                model_residual: np.ndarray | None = None
                for k in range(1, passes + 1):
                    pass_started = time.perf_counter()
                    target, outs, was_chunked, retried = self._separate_pass(source, prompt, backend_options, stereo_mode, visual, chunk_seconds, audio, batch)
                    chunked = chunked or was_chunked
                    if k_cands > 1 and any(o.extra.get("candidates_ignored") for o in outs) and not candidates_ignored_warned:
                        warnings.append(f"backend ignored options.candidates={k_cands}; one draw per window was used")
                        candidates_ignored_warned = True
                    if retried:
                        effective_batch = 1
                        oom_retry = True
                    energy = _energy(target)
                    log.info("prompt %r pass %d/%d: %.1f dB re mixture in %.2fs", spec.display_name, k, passes, _energy_db(energy, mixture_energy), time.perf_counter() - pass_started)
                    if k == 1:
                        first_energy = energy
                        first_target = target
                    elif energy < first_energy * NEGLIGIBLE_PASS_RATIO:
                        # Discarded: it stays in the residual, so reconstruction is unaffected.
                        warnings.append(f"pass {k} extracted negligible energy; stopped after {k - 1} pass(es)")
                        stopped_early, stop_reason = True, "negligible"
                        break
                    elif energy > first_energy * 10 ** (self.settings.pass_max_ratio_db / 10):
                        # A leftover is quieter than the first pass; something this loud is a
                        # different sound the model pulled from a residual that had none left.
                        warnings.append(f"pass {k} extracted {_energy_db(energy, first_energy):+.1f} dB relative to pass 1 (limit {self.settings.pass_max_ratio_db:g} dB): probably a different sound; discarded, stopped after {k - 1} pass(es)")
                        stopped_early, stop_reason = True, "too_much"
                        break
                    elif first_target is not None and (similarity := _spectral_similarity(first_target, target)) < self.settings.pass_min_similarity:
                        warnings.append(f"pass {k} does not sound like pass 1 (spectral similarity {similarity:.2f} < {self.settings.pass_min_similarity:g}); discarded, stopped after {k - 1} pass(es)")
                        stopped_early, stop_reason = True, "dissimilar"
                        break
                    cumulative += target
                    per_pass_db.append(_energy_db(energy, mixture_energy))
                    if outs[0].spans is not None:
                        saw_spans = True
                        pass_spans.extend((float(a), float(b)) for a, b in outs[0].spans)
                    if outs[0].residual is not None:
                        res = np.stack([o.residual for o in outs]) if stereo_mode == "per_channel" else np.repeat(outs[0].residual[None, :], mixture.shape[0], axis=0)
                        model_residual = res[:, : audio.frames].astype(np.float32)
                    else:
                        model_residual = None
                    if k == passes:
                        break
                    if first_energy <= 0:
                        warnings.append("pass 1 extracted negligible energy; stopped after 1 pass(es)")
                        stopped_early, stop_reason = True, "silent_first_pass"
                        break
                    source = (mixture - cumulative).astype(np.float32)
                    if not np.any(source):
                        warnings.append(f"residual is silent after pass {k}; stopped after {k} pass(es)")
                        stopped_early, stop_reason = True, "silent_residual"
                        break
                target = cumulative
                if suppression_cfg is not None:
                    target = mixture - suppress_target(mixture, target, **{k:v for k,v in suppression_cfg.items() if k != 'mode'})
                elif refine_cfg is not None:
                    # Post-processing on the summed target only: passes above ran on the
                    # raw model residual chain, and the residual stem is still derived as
                    # mixture - sum(targets) below, so stems keep summing to the mixture.
                    target = self._refine_target(mixture, target, refine_cfg, stereo_mode)
                    if visual is not None:
                        sr = self.backend.sample_rate
                        s0 = min(audio.frames, max(0, int(round(visual.interval[0] * sr))))
                        s1 = min(audio.frames, max(s0, int(round(visual.interval[1] * sr))))
                        target[:, :s0] = 0.0
                        target[:, s1:] = 0.0
                passes_run = len(per_pass_db)
                max_passes_run = max(max_passes_run, passes_run)
                if visual is not None:
                    # A near-silent result can indicate the visible object was not
                    # audible or the model failed to associate it with the sound.
                    lo = max(0, round(visual.interval[0] * audio.sample_rate))
                    hi = min(audio.frames, round(visual.interval[1] * audio.sample_rate))
                    if hi > lo:
                        input_rms = float(np.sqrt(np.mean(mixture[:, lo:hi].astype(np.float64) ** 2)))
                        target_rms = float(np.sqrt(np.mean(target[:, lo:hi].astype(np.float64) ** 2)))
                        if input_rms > 1e-5 and target_rms < input_rms * 0.01:
                            warnings.append('Very little sound was isolated from the selected object. Listen to the result and try text plus an explicit sound time anchor. No edit has been applied.')
                if saw_spans:
                    # One pass keeps the backend's spans verbatim; several are unioned.
                    spans = [[s, e] for s, e in (_merge_spans(pass_spans) if passes_run > 1 else pass_spans)]
                else:
                    spans = [[a[1], a[2]] for a in anchors if a[0] == "+"] or None
                if model_residual is not None:
                    model_residuals.append(model_residual)
                label = spec.visual_prompt.label if spec.visual_prompt and spec.visual_prompt.label else None
                refinement = {"passes_requested": passes, "passes_run": passes_run, "per_pass_energy_db": per_pass_db, "stopped_early": stopped_early}
                if stop_reason:  # only when the chain stopped early (the API drops None fields)
                    refinement["stop_reason"] = stop_reason
                stems.append(StemArrays(spec.display_name, "target", target, spec.text or None, label or spec.display_name, spans, refinement=refinement))

            peak_memory = int(cuda.max_memory_allocated()) if cuda else None
        if oom_retry:
            warnings.append("GPU memory was insufficient for parallel chunks; processing retried sequentially")

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
            "chunk_seconds": chunk_seconds,
            "chunk_overlap_seconds": self.settings.chunk_overlap_seconds,
            "peak_memory_bytes": peak_memory,
            "requested_chunk_batch": batch,
            "effective_chunk_batch": effective_batch if chunked else 1,
            "passes": max_passes_run,
            "refine": refine_cfg,
            "suppression": suppression_cfg,
            "candidates": k_cands,
            "candidate_select": backend_options.candidate_select if k_cands > 1 else "first",
            "elapsed_seconds": round(elapsed, 3),
            "is_mock": self.backend.is_mock,
        }
        return EngineResult(stems, model, warnings, elapsed)

    # ------------------------------------------------------------------ one pass
    def _separate_pass(self, source: np.ndarray, prompt: BackendPrompt, options: BackendOptions, stereo_mode: str, visual: VisualInput | None, chunk_seconds: float, audio: DecodedAudio, batch: int) -> tuple[np.ndarray, list[BackendResult], bool, bool]:
        """Separate ``source`` (channels, frames) for one prompt: the mixture on pass 1, a
        residual on later passes. Returns the (channels, frames) float32 target, the
        per-channel backend results, whether chunking ran, and whether a batched OOM forced
        a sequential retry."""
        channel_inputs = [source[c] for c in range(source.shape[0])] if stereo_mode == "per_channel" else [to_mono(source)]
        outs: list[BackendResult] = []
        chunked = retried = False
        for signal in channel_inputs:
            if visual is not None:
                result, was_chunked = self._separate_interval(signal, prompt, options, visual.interval), False
            else:
                result, was_chunked = self._separate_signal(signal, prompt, options, chunk_seconds, audio.duration, batch)
            chunked = chunked or was_chunked
            retried = retried or bool(result.extra.get("parallel_oom_retry"))
            outs.append(result)
        target = np.stack([sanitise(o.target) for o in outs]) if stereo_mode == "per_channel" else np.repeat(sanitise(outs[0].target)[None, :], source.shape[0], axis=0)
        target = target[:, : audio.frames].astype(np.float32)
        if target.shape[1] < audio.frames:
            target = np.pad(target, ((0, 0), (0, audio.frames - target.shape[1])))
        return target, outs, chunked, retried

    # ------------------------------------------------------------------ chunking
    def _call_backend(self, signal: np.ndarray, prompt: BackendPrompt, options: BackendOptions) -> BackendResult:
        """One unchunked window: with candidates there is no neighbour, so pick without one."""
        result = self._call_backend_many([signal], [prompt], options)[0]
        if result.candidates is not None:
            idx, target = choose(result.candidates, None, 0, options.candidate_select)
            result.residual = choose_residual(result.candidate_residuals, idx)
            result.target = target
            result.extra = {**result.extra, "candidate_choices": [idx]}
        return result

    @staticmethod
    def _flag_ignored_candidates(results: list[BackendResult], options: BackendOptions) -> None:
        if options.candidates > 1:
            for r in results:
                if r.candidates is None:
                    r.extra = {**r.extra, "candidates_ignored": True}

    def _call_backend_many(self, signals: list[np.ndarray], prompts: list[BackendPrompt], options: BackendOptions) -> list[BackendResult]:
        """One backend call for ``k`` windows, with error classification and length repair."""
        try:
            results = self.backend.separate_many([s.astype(np.float32, copy=False) for s in signals], prompts, options)
        except AudioServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            if is_out_of_memory(exc):
                raise InsufficientGpuMemory("GPU ran out of memory during inference; try a shorter input, AUDIO_CHUNK_SECONDS, a smaller AUDIO_CHUNK_BATCH, fewer reranking candidates or fewer options.candidates", details={"frames": int(max(len(s) for s in signals)), "batch": len(signals), "candidates": options.candidates}) from exc
            raise InferenceFailure(f"{type(exc).__name__}: {exc}") from exc
        if len(results) != len(signals):
            raise InferenceFailure(f"backend returned {len(results)} results for {len(signals)} windows")
        for signal, result in zip(signals, results, strict=True):
            self._fit_length(result, len(signal))
        self._flag_ignored_candidates(results, options)
        return results

    @staticmethod
    def _fit_length(result: BackendResult, n: int) -> None:
        """Trim/pad a backend result in place so it matches its own window exactly.

        Target and residual are checked independently: a batched backend pads the batch to
        its longest window, and the crossfade merge adds both into ``n``-sample slices, so a
        residual left at the padded length would raise a broadcast error rather than be
        silently wrong.
        """
        def fit(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None or arr.shape[-1] == n:
                return arr
            m = min(n, arr.shape[-1])
            fixed = np.zeros(n, dtype=np.float32)
            fixed[:m] = arr[:m]
            return fixed

        for attr in ("target", "residual"):
            setattr(result, attr, fit(getattr(result, attr)))
        for attr in ("candidates", "candidate_residuals"):
            arrs = getattr(result, attr)
            if arrs is not None:
                setattr(result, attr, [fit(a) for a in arrs])

    def _separate_interval(self, signal: np.ndarray, prompt: BackendPrompt, options: BackendOptions, interval: tuple[float, float]) -> BackendResult:
        """Visual prompting: separate only the interval the masks cover, then place the
        result back on the full timeline so stems keep ``start_time = 0``."""
        sr = self.backend.sample_rate
        n = len(signal)
        s0 = min(n, max(0, int(round(interval[0] * sr))))
        s1 = min(n, max(s0, int(round(interval[1] * sr))))
        if s1 - s0 < sr // 10:
            raise InvalidRequest("visual prompt interval is shorter than 100 ms or outside the audio", details={"interval": list(interval)})
        if (s1 - s0) / sr > self.settings.max_visual_interval_seconds:
            raise UnsupportedOperation(
                f"visual prompt interval is {(s1 - s0) / sr:.1f}s; visually prompted separation runs as one model window and is limited to "
                f"{self.settings.max_visual_interval_seconds:.0f}s (AUDIO_MAX_VISUAL_INTERVAL_SECONDS). Select a shorter interval or use text + span anchors.",
                details={"interval": list(interval), "max_visual_interval_seconds": self.settings.max_visual_interval_seconds},
            )
        local_anchors = None
        if prompt.anchors:
            local_anchors = [(t, max(a - s0 / sr, 0.0), min(b - s0 / sr, (s1 - s0) / sr)) for t, a, b in prompt.anchors]
            local_anchors = [a for a in local_anchors if a[2] > a[1]] or None
        out = self._call_backend(signal[s0:s1], BackendPrompt(prompt.text, local_anchors, prompt.visual), options)
        target = np.zeros(n, dtype=np.float32)
        target[s0:s1] = out.target
        residual = None
        if out.residual is not None:
            residual = signal.astype(np.float32).copy()
            residual[s0:s1] = out.residual
        spans = [(a + s0 / sr, b + s0 / sr) for a, b in out.spans] if out.spans else [(s0 / sr, s1 / sr)]
        return BackendResult(target=target, residual=residual, spans=[(round(a, 3), round(b, 3)) for a, b in spans], extra=out.extra)

    def _separate_signal(self, signal: np.ndarray, prompt: BackendPrompt, options: BackendOptions, chunk_seconds: float, duration: float, chunk_batch: int | None = None) -> tuple[BackendResult, bool]:
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
        choices: list[int | None] = []
        ignored = False
        has_residual = True
        def windows():
            start = 0
            while start < n:
                end = min(start + size, n)
                if end - start < overlap and start > 0:
                    break
                local_anchors = None
                if prompt.anchors:
                    local_anchors = []
                    for token, a_start, a_end in prompt.anchors:
                        s = max(a_start - start / sr, 0.0)
                        e = min(a_end - start / sr, (end - start) / sr)
                        if e > s:
                            local_anchors.append((token, s, e))
                    local_anchors = local_anchors or None
                yield start, end, BackendPrompt(prompt.text, local_anchors, prompt.visual)
                if end >= n:
                    break
                start += step

        def merge(start, end, out):
            nonlocal has_residual, ignored
            ramp = np.ones(end - start)
            if start > 0:
                ramp[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False)
            if end < n:
                ramp[-overlap:] = np.minimum(ramp[-overlap:], np.linspace(1.0, 0.0, overlap, endpoint=False))
            out_target, out_residual = out.target, out.residual
            if out.candidates is not None:
                # Windows merge in timeline order, so [start, start + overlap) already holds
                # the previous window's (ramped) contribution; dividing by the weight so far
                # recovers that window's own output for the consistency check.
                prev = target[start : start + overlap] / np.maximum(weight[start : start + overlap], 1e-6) if start > 0 else None
                idx, out_target = choose(out.candidates, prev, overlap, options.candidate_select)
                out_residual = choose_residual(out.candidate_residuals, idx)
                choices.append(idx)
            elif out.extra.get("candidates_ignored"):
                ignored = True
            target[start:end] += out_target * ramp
            if out_residual is None:
                has_residual = False
            else:
                residual[start:end] += out_residual * ramp
            weight[start:end] += ramp
            if out.spans:
                spans.extend((s + start / sr, e + start / sr) for s, e in out.spans)

        def run_windows(batch_size):
            """Collect up to ``batch_size`` windows, hand them to the backend in ONE call,
            then merge the results in timeline order so the crossfade is unchanged."""
            pending: list[tuple[int, int, BackendPrompt]] = []

            def flush():
                if not pending:
                    return
                results = self._call_backend_many([signal[a:b] for a, b, _ in pending], [p for _, _, p in pending], options)
                for (a, b, _), out in zip(pending, results, strict=True):
                    merge(a, b, out)
                pending.clear()

            for start, end, local_prompt in windows():
                pending.append((start, end, local_prompt))
                if len(pending) >= batch_size:
                    flush()
            flush()

        # Each window becomes ``options.candidates`` batch items in the backend, so a model
        # call never holds more than ``chunk_batch`` items.
        batch_size = max(1, (chunk_batch or self.settings.chunk_batch) // max(1, options.candidates))
        retried_sequentially = False
        try:
            run_windows(batch_size)
        except InsufficientGpuMemory:
            if batch_size == 1:
                raise
            retried_sequentially = True
            log.warning("batched chunks exceeded GPU memory; retrying this request sequentially")
            target.fill(0); residual.fill(0); weight.fill(0); spans.clear(); choices.clear(); has_residual = True
            run_windows(1)
        weight = np.maximum(weight, 1e-6)
        extra: dict[str, Any] = {"parallel_oom_retry": retried_sequentially}
        if choices:
            extra["candidate_choices"] = list(choices)
        if ignored:
            extra["candidates_ignored"] = True
        merged = BackendResult(
            target=(target / weight).astype(np.float32),
            residual=(residual / weight).astype(np.float32) if has_residual else None,
            spans=_merge_spans(spans) if spans else None,
            extra=extra,
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
    root = root.resolve()
    directory = (root / request_id).resolve()
    if directory.parent != root:
        raise InvalidRequest("request_id must be a single path segment")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem.stem_id}.wav"
    path.write_bytes(data)
    return path
