"""Backend protocol shared by the mock and the SAM Audio implementation.

A backend separates ONE mono 48 kHz signal for ONE prompt. Everything else (stereo
handling, chunking, residual computation, encoding, HTTP) lives in the engine/API so the
real and mock backends stay interchangeable and small.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Anchor = tuple[str, float, float]  # ("+" | "-", start_seconds, end_seconds) relative to the audio passed in


@dataclass
class VisualInput:
    """Resolved, model-ready visual guidance for the *segment* being separated.

    ``video_path`` and ``mask_video_path`` are local files (already resolved by media.py).
    ``segment_start``/``segment_end`` are seconds on the source timeline covered by the
    audio handed to the backend, so it can slice frames to match.
    """

    video_path: str
    mask_video_path: str
    segment_start: float
    segment_end: float
    object_id: str | None = None
    label: str | None = None


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


@dataclass
class BackendResult:
    target: np.ndarray  # float32 mono, same length as the input
    residual: np.ndarray | None = None  # model residual if the backend produces one
    spans: list[tuple[float, float]] | None = None  # seconds relative to the input, if known
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    name: str = "base"
    is_mock: bool = False
    supports_visual: bool = False
    supports_spans: bool = False
    supports_span_prediction: bool = False
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

    @property
    def loaded(self) -> bool:
        return True

    def unload(self) -> None:  # pragma: no cover - optional
        return None
