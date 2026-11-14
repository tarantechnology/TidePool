"""Backend protocol shared by the mock and the SAM Audio implementation.

A backend separates ONE mono 48 kHz signal for ONE prompt (:meth:`Backend.separate`), or
``k`` such windows at once (:meth:`Backend.separate_many`, which defaults to looping).
Everything else (stereo handling, chunking, residual computation, encoding, HTTP) lives in
the engine/API so the real and mock backends stay interchangeable and small.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Anchor = tuple[str, float, float]  # ("+" | "-", start_seconds, end_seconds) relative to the audio passed in


@dataclass
class VisualInput:
    """Resolved, model-ready visual guidance for the audio handed to the backend.

    The engine crops the mixture to ``interval`` (seconds on the source timeline) before
    calling the backend, so the frames the backend decodes for that interval line up with
    the audio it receives. Masks come from exactly one of:

    * ``mask_video_path``: a losslessly encoded video aligned frame-for-frame with
      ``video_path`` where pixel values > 127 are the selected object (generic contract), or
    * ``frame_masks``: Person 2's per-frame instance masks, ``frame_index -> HxW`` array with
      1 = object (decoded from their RLE); frames missing from the dict mean "object absent"
      and get an all-zero mask (never reuse a previous mask).
    """

    video_path: str
    interval: tuple[float, float]
    mask_video_path: str | None = None
    frame_masks: dict[int, np.ndarray] | None = None
    frame_rate: float | None = None  # native frame rate of video_path, if known
    frame_size: tuple[int, int] | None = None  # (width, height) the masks were produced at
    object_id: str | None = None
    label: str | None = None
    source_sha256: str | None = None  # when set, the backend/visual loader verifies video bytes


@dataclass
class BackendPrompt:
    text: str
    anchors: list[Anchor] | None = None
    visual: VisualInput | None = None


@dataclass
class BackendOptions:
    predict_spans: bool = False
    reranking_candidates: int = 1
    seed: int | None = None
    # Independent noise draws to RETURN per window (``BackendResult.candidates``); the engine
    # picks one per window. Distinct from ``reranking_candidates``, which the model ranks
    # internally. Backends that ignore it simply return ``candidates=None``.
    candidates: int = 1
    # Engine-side selection rule ("first" | "overlap" | "mean"); backends do not use it.
    candidate_select: str = "overlap"


@dataclass
class BackendResult:
    target: np.ndarray  # float32 mono, same length as the input
    residual: np.ndarray | None = None  # model residual if the backend produces one
    spans: list[tuple[float, float]] | None = None  # seconds relative to the input, if known
    extra: dict[str, Any] = field(default_factory=dict)
    # K mono targets when options.candidates > 1 (``candidates[0]`` is ``target``) and the
    # matching residuals; None when the backend produced a single draw.
    candidates: list[np.ndarray] | None = None
    candidate_residuals: list[np.ndarray | None] | None = None


class Backend(ABC):
    name: str = "base"
    is_mock: bool = False
    supports_visual: bool = False
    supports_spans: bool = False
    supports_span_prediction: bool = False
    supports_independent_candidates: bool = True
    sample_rate: int = 48_000

    @abstractmethod
    def load(self) -> None:
        """Load weights. Must be idempotent."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Model id, device, dtype and anything useful for /health and /capabilities."""

    @abstractmethod
    def separate(self, mixture: np.ndarray, prompt: BackendPrompt, options: BackendOptions) -> BackendResult:
        """``mixture`` is float32 mono at ``self.sample_rate``. Output must match its length exactly."""

    def separate_many(
        self, mixtures: list[np.ndarray], prompts: list[BackendPrompt], options: BackendOptions
    ) -> list[BackendResult]:
        """Separate ``k`` independent windows and return results in input order.

        This is how the engine expresses "process several chunk windows at once": a backend
        that can put them in ONE model batch (see ``SamAudioBackend``) overrides this and
        gets real GPU-side parallelism without ever entering a shared ``torch.nn.Module``
        from two threads. The default loops over :meth:`separate`, so the mock and any other
        backend keep working unchanged.
        """
        if len(mixtures) != len(prompts):
            raise ValueError(f"separate_many needs one prompt per mixture, got {len(mixtures)} and {len(prompts)}")
        return [self.separate(mixture, prompt, options) for mixture, prompt in zip(mixtures, prompts, strict=True)]

    @property
    def loaded(self) -> bool:
        return True

    def unload(self) -> None:  # pragma: no cover - optional
        return None
