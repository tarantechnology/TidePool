"""Real SAM Audio backend: mono 48 kHz windows in, separated stems out.

Everything model-specific lives here. The engine owns chunking, stereo, residuals and
length bookkeeping, so this class only ever sees windows that already match the visual
interval they are given.

Concurrency: the model and processor are loaded once and shared, and a ``torch.nn.Module``
is not safe to enter from two threads (see ``docs/chunk_concurrency.md``). Every call is
therefore serialised by an instance lock, and "several windows at once" is expressed as a
processor BATCH through :meth:`SamAudioBackend.separate_many`, which is the parallelism the
GPU actually wants.

Upstream shape (verified against ``sam_audio`` sources, not guessed)
--------------------------------------------------------------------
* ``SAMAudio.from_pretrained(model_id, **kwargs)``: any kwarg whose key exists in the
  repo's ``config.json`` *overrides* that config entry. ``span_predictor=None`` drops the
  multi-billion-parameter PE-A-Frame span predictor; ``text_ranker=None`` /
  ``visual_ranker=None`` drop the CLAP/Judge/ImageBind re-rankers. ``token`` is forwarded
  to the Hub for the gated checkpoints.
* ``SAMAudioProcessor.from_pretrained(model_id)``; ``audio_sampling_rate == 48000`` and
  ``audio_hop_length == 1920`` (one feature frame per 40 ms).
* ``processor(descriptions=[...], audios=[(C, T) float32 @ 48 kHz, ...], anchors=..., masked_videos=...)``
  takes k items at once, averages each audio to mono internally and pads the batch to the
  longest item. ``anchors`` is one list per item. Anchor tokens are ``"+"`` (target
  present) and ``"-"`` (absent), in seconds relative to the audio passed in.
* ``model.separate(batch, noise=None, ode_opt=..., reranking_candidates=1, predict_spans=False)``
  returns ``result.target`` / ``result.residual`` as **lists** of 1-D mono tensors, one per
  batch item, whose length is ``ceil(T_max / 1920) * 1920`` — longer than a short item's
  own input, so every output is trimmed back to ITS OWN input length here.
* Span prediction only runs when the span predictor was loaded **and**
  ``batch.anchors is None``; afterwards ``batch.anchors`` holds predicted
  ``["+", start, end]`` lists, which is where ``BackendResult.spans`` comes from.

Memory (user-reported in upstream issue #24, not measured by us)
---------------------------------------------------------------
* ``sam-audio-small``: ~21.6 GB to load, ~28 GB peak on a 15 s clip.
* ``sam-audio-base``: ~24 GB to load.
* ``sam-audio-large``: ~38 GB peak on a 15 s clip; ~30 s is about the maximum on an
  80 GB A100.
* Dropping the span predictor and the re-rankers saves a large fraction of that.
* FP16 breaks separation quality, so float32 is the supported dtype and ``bfloat16`` is an
  explicit opt-in only. Maintainers recommend chunking long inputs into 10-30 s windows.

Recommended configuration per GPU class (``AUDIO_*`` env vars)
--------------------------------------------------------------
======================  =========================================================================
24 GB (3090/4090/A10)   ``AUDIO_MODEL_ID=facebook/sam-audio-small`` (or ``-base``),
                        ``AUDIO_LOAD_SPAN_PREDICTOR=false``, ``AUDIO_CHUNK_SECONDS=10``-``15``,
                        ``AUDIO_RERANKING_CANDIDATES=1``
40-48 GB (A100 40/L40S) ``facebook/sam-audio-large``, ``AUDIO_CHUNK_SECONDS=15``-``20``,
                        span predictor on, ``AUDIO_RERANKING_CANDIDATES=1``-``2``
80 GB (A100/H100 80)    ``facebook/sam-audio-large``, ``AUDIO_CHUNK_SECONDS=30``,
                        ``AUDIO_RERANKING_CANDIDATES`` up to ``8``
======================  =========================================================================

Visual polarity A/B (``AUDIO_VISUAL_MASK_MODE``)
------------------------------------------------
``mask_videos`` keeps pixels where the mask is 0. The paper describes the model seeing
``V ⊙ M`` (object kept, background zeroed), which is also the only reading under which the
ImageBind re-ranker — it scores audio against the *masked* video — makes sense. The
official visual notebook nevertheless passes object masks straight in, which zeroes the
object; upstream issue #75 is unresolved. Hence two modes:

* ``keep_object`` (default) passes ``1 - object_mask`` so the object survives;
* ``notebook`` passes the object mask unchanged, reproducing the notebook.

Both are one env var apart so the same clip can be run twice and compared. Run the A/B as
soon as checkpoint access lands (``scripts/smoke_sam.py --video ... --mask-mode both``):
same mixture, same seed, same prompt, flip only the mode, and compare target energy inside
the object's active spans. Until then the default follows the paper and matches Person 2's
``build_processor_batch``, which also inverts foreground masks.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import nullcontext
from typing import Any

import numpy as np

from ..errors import InferenceFailure, InsufficientGpuMemory, MissingModel, is_out_of_memory
from ..visual import MASK_MODES, MaskMode, prepare_masked_video
from .base import Backend, BackendOptions, BackendPrompt, BackendResult

log = logging.getLogger("sam_audio_service")

# Exception text that means "the weights are not available to us", as opposed to a bug.
_ACCESS_HINTS = (
    "gated", "awaiting a review", "401", "403", "404", "unauthorized", "authorization",
    "access to model", "repository not found", "entry not found", "is not a local folder",
    "could not connect", "connectionerror", "offline mode", "token",
)
# "did the GPU run out of memory?" lives in errors.py so the engine's batch-1 retry and
# this module's InsufficientGpuMemory mapping can never disagree. CUBLAS/cuDNN/driver
# allocation failures count, not just the literal "out of memory".
_is_oom = is_out_of_memory


def _looks_like_missing_weights(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return isinstance(exc, (OSError, FileNotFoundError)) or any(hint in blob for hint in _ACCESS_HINTS)


def _batchable_groups(mixtures: list[np.ndarray], prompts: list[BackendPrompt]) -> list[list[int]]:
    """Split window indices into consecutive groups, one ``model.separate`` call each.

    Windows only share a call when the processor can describe them in one batch:

    * a visually prompted window runs alone (``masked_videos`` is per-item upstream and
      frames must stay aligned with exactly the audio they came from), and so does an
      empty window (nothing to encode);
    * anchored and unanchored windows do not mix. An empty per-item anchor list is in fact
      accepted upstream and conditions identically to ``anchors=None``
      (``Batch.process_anchors`` builds the same ``[<null>, <pad>]`` row), but
      ``anchors is None`` is also the batch-wide "predict spans" gate
      (``SAMAudio.separate``), so mixing would silently disable span prediction for
      windows that would have got it when run alone.
    """
    groups: list[list[int]] = []
    previous: object = object()
    for i, (mixture, prompt) in enumerate(zip(mixtures, prompts, strict=True)):
        alone = prompt.visual is not None or int(np.asarray(mixture).shape[-1]) == 0
        key = None if alone else bool(prompt.anchors)
        if groups and key is not None and key == previous:
            groups[-1].append(i)
        else:
            groups.append([i])
        previous = key
    return groups


class SamAudioBackend(Backend):
    """SAM Audio inference for one window. See the module docstring for the upstream contract."""

    name = "sam"
    is_mock = False
    supports_visual = True
    supports_spans = True
    sample_rate = 48_000

    def __init__(
        self,
        *,
        model_id: str = "facebook/sam-audio-large",
        device: str = "auto",
        dtype: str = "float32",
        mask_mode: MaskMode = "keep_object",
        load_span_predictor: bool = True,
        load_rankers: bool = True,
        load_visual_ranker: bool = True,
        judge_only: bool = False,
        ode_method: str = "midpoint",
        ode_step_size: float = 2 / 32,
        frame_size: int = 336,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        ffmpeg_timeout: int = 600,
    ):
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
        self.model_id = model_id
        self.requested_device = device
        self.requested_dtype = dtype.lower()
        self.mask_mode: MaskMode = mask_mode
        self.load_span_predictor = load_span_predictor
        self.load_rankers = load_rankers
        self.load_visual_ranker = load_visual_ranker
        self.judge_only = judge_only
        self.frame_size = frame_size
        self.ode_opt: dict[str, Any] = {"method": ode_method, "options": {"step_size": ode_step_size}}
        self.ffmpeg, self.ffprobe, self.ffmpeg_timeout = ffmpeg, ffprobe, ffmpeg_timeout
        # Span prediction needs the span predictor; shadow the class flag per instance.
        self.supports_span_prediction = load_span_predictor

        self._torch: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._parameters: int | None = None
        self._load_seconds: float | None = None
        self._facts: dict[str, Any] = {}
        # Serialises load and inference. The model is a single shared torch.nn.Module and
        # must never be entered twice at once, whatever the engine does above us.
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------- lifecycle
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Import ``sam_audio``, resolve the device and pull the weights. Idempotent."""
        if self.loaded:
            return
        with self._lock:
            if not self.loaded:
                self._load_locked()

    def _load_locked(self) -> None:
        torch = self._import_torch()
        SAMAudio, SAMAudioProcessor = self._import_sam_audio()
        device = self._resolve_device(torch, self.requested_device)
        dtype = self._resolve_dtype(torch, self.requested_dtype, device)

        started = time.perf_counter()
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = SAMAudio.from_pretrained(self.model_id, **self._model_kwargs())
            processor = SAMAudioProcessor.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001 - classified below
            self._release(torch)
            if _is_oom(exc):
                raise
            if _looks_like_missing_weights(exc):
                raise MissingModel(
                    f"could not load {self.model_id}: {type(exc).__name__}: {exc}. The SAM Audio "
                    "checkpoints are gated; request access on huggingface.co and set HF_TOKEN.",
                    details={"model_id": self.model_id},
                ) from exc
            raise InferenceFailure(f"loading {self.model_id} failed: {type(exc).__name__}: {exc}") from exc

        model = model.eval().to(device=device)
        if dtype is not torch.float32:
            model = model.to(dtype=dtype)
        if self.load_rankers and self.judge_only:
            # Loading both checkpoints before moving SAM to the GPU can exhaust a
            # 50 GB host even when the models fit in VRAM. Load the judge afterwards.
            try:
                from sam_audio.model.config import JudgeRankerConfig
                from sam_audio.ranking.judge import JudgeRanker
                model.text_ranker = JudgeRanker(JudgeRankerConfig()).eval().to(device=device)
            except Exception as exc:
                self._release(torch)
                raise InferenceFailure(f"loading the audio judge failed: {type(exc).__name__}: {exc}") from exc
        self._torch, self._model, self._processor = torch, model, processor
        self._device, self._dtype = device, dtype
        self._parameters = sum(p.numel() for p in model.parameters())
        self._load_seconds = round(time.perf_counter() - started, 3)
        self._facts = self._collect_facts(SAMAudio, model, processor)
        log.info(
            "loaded %s on %s (%s, %.2fB params, span_predictor=%s, rankers=%s) in %.1fs",
            self.model_id, device, self.requested_dtype, self._parameters / 1e9,
            self._facts["span_predictor_loaded"],
            self._facts["visual_ranker_loaded"] or self._facts["text_ranker_loaded"],
            self._load_seconds,
        )

    def unload(self) -> None:
        with self._lock:  # never drop the model out from under an in-flight separation
            torch = self._torch
            self._model = self._processor = None
            self._parameters = None
            self._release(torch)
            self._torch = None

    def _release(self, torch: Any | None) -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ------------------------------------------------------------------------- imports
    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise MissingModel(
                "torch is not installed; install the service with the 'sam' extra "
                "(uv pip install -e '.[sam]')",
            ) from exc
        return torch

    @staticmethod
    def _import_sam_audio() -> tuple[Any, Any]:
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor
        except Exception as exc:  # noqa: BLE001 - ImportError, but also torchcodec/dacvae failures
            raise MissingModel(
                f"sam_audio is not importable ({type(exc).__name__}: {exc}); install the service "
                "with the 'sam' extra. The processor also needs torchcodec.",
            ) from exc
        return SAMAudio, SAMAudioProcessor

    def _model_kwargs(self) -> dict[str, Any]:
        """Config overrides + Hub token. Keys absent from config.json are ignored upstream."""
        kwargs: dict[str, Any] = {}
        if not self.load_span_predictor:
            kwargs["span_predictor"] = None
        if not self.load_rankers:
            kwargs["text_ranker"] = None
            kwargs["visual_ranker"] = None
        else:
            if not self.load_visual_ranker:
                kwargs["visual_ranker"] = None
            if self.judge_only:
                kwargs["text_ranker"] = None
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            kwargs["token"] = token
        return kwargs

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        """``auto`` -> cuda > mps > cpu. Anything else is passed to torch verbatim."""
        wanted = (requested or "auto").strip().lower()
        if wanted == "auto":
            if torch.cuda.is_available():
                wanted = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                wanted = "mps"
            else:
                wanted = "cpu"
        try:
            device = torch.device(wanted)
        except (RuntimeError, ValueError) as exc:
            raise InferenceFailure(f"AUDIO_DEVICE={requested!r} is not a valid torch device") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise MissingModel("AUDIO_DEVICE asks for cuda but no CUDA device is visible")
        return device

    @staticmethod
    def _resolve_dtype(torch: Any, requested: str, device: Any) -> Any:
        if requested in ("float32", "fp32", "", None):
            return torch.float32
        if requested in ("bfloat16", "bf16"):
            if device.type == "cpu":
                raise InferenceFailure("AUDIO_DTYPE=bfloat16 needs a GPU; use float32 on cpu")
            log.warning("AUDIO_DTYPE=bfloat16 is an opt-in experiment; float32 is the supported dtype")
            return torch.bfloat16
        if requested in ("float16", "fp16", "half"):
            raise InferenceFailure(
                "float16 is refused: it is reported to break SAM Audio separation quality. "
                "Use AUDIO_DTYPE=float32, or bfloat16 if you accept the risk.",
            )
        raise InferenceFailure(f"AUDIO_DTYPE={requested!r} is not supported (float32 | bfloat16)")

    def _collect_facts(self, cls: Any, model: Any, processor: Any) -> dict[str, Any]:
        codec = getattr(model, "audio_codec", None)
        vision = getattr(model, "vision_encoder", None)
        return {
            "revision": getattr(cls, "revision", None),
            "model_sample_rate": getattr(model, "sample_rate", None),
            "hop_length": getattr(codec, "hop_length", None),
            "processor_sample_rate": getattr(processor, "audio_sampling_rate", None),
            "processor_hop_length": getattr(processor, "audio_hop_length", None),
            "vision_image_size": getattr(vision, "image_size", None),
            "vision_dim": getattr(vision, "dim", None),
            "span_predictor_loaded": hasattr(model, "span_predictor"),
            "visual_ranker_loaded": getattr(model, "visual_ranker", None) is not None,
            "text_ranker_loaded": getattr(model, "text_ranker", None) is not None,
        }

    # ----------------------------------------------------------------------------- info
    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_id": self.model_id,
            "device": str(self._device) if self._device is not None else self.requested_device,
            "dtype": self.requested_dtype,
            "sample_rate": self.sample_rate,
            "loaded": self.loaded,
            "mask_mode": self.mask_mode,
            "frame_size": self.frame_size,
            "ode": {"method": self.ode_opt["method"], "step_size": self.ode_opt["options"]["step_size"]},
            "load_span_predictor": self.load_span_predictor,
            "load_rankers": self.load_rankers,
            "supports_span_prediction": self.supports_span_prediction,
            "parameters": self._parameters,
            "load_seconds": self._load_seconds,
            "torch_version": getattr(self._torch, "__version__", None),
            **self._facts,
        }
        torch = self._torch
        if torch is not None and self._device is not None and self._device.type == "cuda":
            out["cuda_device_name"] = torch.cuda.get_device_name(self._device)
            out["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self._device))
            out["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(self._device))
        return out

    # ------------------------------------------------------------------------ inference
    def separate(self, mixture: np.ndarray, prompt: BackendPrompt, options: BackendOptions) -> BackendResult:
        return self.separate_many([mixture], [prompt], options)[0]

    def separate_many(
        self, mixtures: list[np.ndarray], prompts: list[BackendPrompt], options: BackendOptions
    ) -> list[BackendResult]:
        """Separate ``k`` windows, batching them into as few model calls as possible.

        The model and processor are shared, so every call is serialised by ``self._lock``:
        a ``torch.nn.Module`` is not re-entrant (weight-norm forward pre-hooks,
        parametrisation caches and per-module buffers are mutated during ``forward``), and
        per-thread CUDA streams do not make it so. GPU-side parallelism therefore comes
        from the batch dimension, never from threads. See ``docs/chunk_concurrency.md``.
        """
        if len(mixtures) != len(prompts):
            raise ValueError(f"separate_many needs one prompt per mixture, got {len(mixtures)} and {len(prompts)}")
        if not mixtures:
            return []
        self.load()
        with self._lock:
            if self._model is None:  # unloaded between load() and here
                raise MissingModel(f"{self.model_id} is not loaded")
            out: list[BackendResult] = []
            for group in _batchable_groups(mixtures, prompts):
                out.extend(self._separate_batch([mixtures[i] for i in group], [prompts[i] for i in group], options))
            return out

    def _separate_batch(
        self, mixtures: list[np.ndarray], prompts: list[BackendPrompt], options: BackendOptions
    ) -> list[BackendResult]:
        """ONE processor batch and ONE ``model.separate`` call for every window given."""
        torch, model, processor = self._torch, self._model, self._processor
        lengths = [int(np.asarray(m).shape[-1]) for m in mixtures]
        if all(n == 0 for n in lengths):
            empty = np.zeros(0, dtype=np.float32)
            return [BackendResult(target=empty.copy(), residual=empty.copy(), spans=[]) for _ in mixtures]

        # Shape (1, T) per item: the processor averages channels to mono, so hand it mono
        # already. ``batch_audio`` pads the batch to the longest item; each output is
        # trimmed back to its OWN input length below.
        audios = [
            torch.from_numpy(np.ascontiguousarray(np.asarray(m, dtype=np.float32).reshape(-1))).unsqueeze(0)
            for m in mixtures
        ]
        descriptions = [self._description(p) for p in prompts]
        anchors = (
            [[(token, float(start), float(end)) for token, start, end in (p.anchors or ())] for p in prompts]
            if any(p.anchors for p in prompts)
            else None
        )

        masked_videos = frames = batch = None
        visual = prompts[0].visual if len(prompts) == 1 else None
        # Everything that can touch device memory is inside ONE try: building the batch and
        # moving it to the GPU allocates k windows' worth of audio (and a visual window's
        # frames) before the model is ever entered, so an allocation failure there has to
        # free the batch and be classified exactly like one inside ``model.separate``.
        # Without this, an OOM in ``.to(device)`` leaked the batch and reached the engine as
        # an INFERENCE_FAILURE instead of a retryable INSUFFICIENT_GPU_MEMORY.
        try:
            if visual is not None:
                frames, masks = prepare_masked_video(
                    visual,
                    size=self.frame_size,
                    mask_mode=self.mask_mode,
                    ffmpeg=self.ffmpeg,
                    ffprobe=self.ffprobe,
                    timeout=self.ffmpeg_timeout,
                )
                masked_videos = processor.mask_videos([frames], [masks])

            batch = processor(
                descriptions=descriptions,
                audios=audios,
                anchors=anchors,
                masked_videos=masked_videos,
            ).to(self._device)
            if self._dtype is not torch.float32:
                batch.audios = batch.audios.to(self._dtype)

            # Span prediction is skipped upstream when anchors are supplied, so be explicit.
            predict_spans = bool(options.predict_spans and self.supports_span_prediction and anchors is None)
            candidates = max(1, int(options.reranking_candidates or 1))
            notes: list[str] = []
            if visual is not None and self.mask_mode == "notebook" and candidates > 1 and getattr(model, "visual_ranker", None) is not None:
                # The ImageBind re-ranker scores audio against the masked video; with the
                # object zeroed out that comparison is meaningless, so re-ranking is off.
                notes.append("reranking disabled: notebook mask mode removes the object the visual re-ranker needs")
                candidates = 1
            if options.seed is not None:
                # One seeded draw for the whole batch: reproducible for a fixed batch size.
                torch.manual_seed(int(options.seed))

            with torch.inference_mode(), self._autocast(torch):
                result = model.separate(
                    batch,
                    predict_spans=predict_spans,
                    reranking_candidates=candidates,
                    ode_opt=self.ode_opt,
                )
        except Exception as exc:  # noqa: BLE001 - free first, then let the engine classify
            # Drop the batch/frame references before reclaiming, so the traceback holding
            # these locals alive does not pin a window's worth of device memory.
            batch = masked_videos = audios = frames = None
            self._release(torch)
            if _is_oom(exc):
                raise InsufficientGpuMemory(
                    "GPU ran out of memory during separation; lower AUDIO_CHUNK_BATCH, shorten "
                    "AUDIO_CHUNK_SECONDS, lower AUDIO_RERANKING_CANDIDATES, or set "
                    "AUDIO_LOAD_SPAN_PREDICTOR=false",
                    details={
                        "batch": len(mixtures),
                        "frames": max(lengths),
                        "seconds": round(max(lengths) / self.sample_rate, 3),
                        "model_id": self.model_id,
                    },
                ) from exc
            raise

        results: list[BackendResult] = []
        for i, length in enumerate(lengths):
            extra = {
                "mask_mode": self.mask_mode if visual is not None else None,
                "visual_frames": int(frames.shape[0]) if frames is not None else 0,
                "predict_spans": predict_spans,
                "reranking_candidates": candidates,
                "description": descriptions[i],
                "batch": len(mixtures),
                "notes": notes,
            }
            results.append(
                BackendResult(
                    target=self._to_numpy(result.target[i], length),
                    residual=self._to_numpy(result.residual[i], length) if result.residual is not None else None,
                    spans=self._predicted_spans(batch, i) if predict_spans else None,
                    extra=extra,
                )
            )
        return results

    @staticmethod
    def _description(prompt: BackendPrompt) -> str:
        """Lowercased free text. Visual-only prompts fall back to Person 2's label."""
        text = (prompt.text or "").strip()
        if not text and prompt.visual is not None:
            text = (prompt.visual.label or "").strip()
        return text.lower()

    def _autocast(self, torch: Any):
        """bfloat16 runs under autocast so float32 inputs (uint8 frames, audio) still work."""
        if self._dtype is torch.float32 or self._device is None:
            return nullcontext()
        return torch.autocast(device_type=self._device.type, dtype=self._dtype)

    def _to_numpy(self, wav: Any, length: int) -> np.ndarray:
        """1-D model output -> float32 numpy of exactly ``length`` samples.

        The codec emits ``ceil(T / hop_length) * hop_length`` samples, so the tail is
        trimmed; the pad branch is defensive only.
        """
        arr = wav.detach().to(self._torch.float32).reshape(-1).cpu().numpy()
        if arr.shape[0] >= length:
            return np.ascontiguousarray(arr[:length], dtype=np.float32)
        out = np.zeros(length, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out

    @staticmethod
    def _predicted_spans(batch: Any, index: int = 0) -> list[tuple[float, float]]:
        """``batch.anchors`` after ``separate`` holds per-item predicted ``["+", s, e]`` lists."""
        anchors = getattr(batch, "anchors", None)
        if not anchors or index >= len(anchors):
            return []
        return [(float(a[1]), float(a[2])) for a in anchors[index] if len(a) >= 3 and a[0] == "+"]


__all__ = ["SamAudioBackend"]
