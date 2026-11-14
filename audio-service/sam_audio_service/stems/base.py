"""Stem splitter interface: one stereo mixture in, N named stems out.

Kept separate from the prompted :class:`~sam_audio_service.backends.base.Backend` (one mono
signal and one prompt in, one target out): a splitter produces every stem in a single model
run, so forcing it through ``Backend.separate`` would mean N runs or an output cache.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import Cancelled

# progress(fraction in [0, 1], stage) - called from the inference thread; must be cheap.
Progress = Callable[[float, str], None]


def report(progress: Progress | None, fraction: float, stage: str) -> None:
    if progress is not None:
        progress(min(1.0, max(0.0, fraction)), stage)


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("stem split was cancelled")


class StemSplitter(ABC):
    name: str = "base"
    is_mock: bool = False
    stems: tuple[str, ...] = ()
    sample_rate: int = 48_000  # rate of the arrays passed to and returned from split()

    @abstractmethod
    def load(self) -> None:
        """Load weights. Must be idempotent."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Model id, device and anything useful for /health and /capabilities."""

    @abstractmethod
    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        """``mixture`` is float32 ``(channels, n)`` at ``sample_rate``, 1 or 2 channels.

        Returns one float32 array of exactly the same shape per name in :attr:`stems`.
        ``progress`` is called as work completes, as finely as the model allows; ``cancel``
        is checked between units of work and raises :class:`~sam_audio_service.errors.Cancelled`.
        """

    @property
    def loaded(self) -> bool:
        return True

    def unload(self) -> None:  # pragma: no cover - optional
        return None


def fit_length(samples: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad the last axis to exactly ``n`` samples (resampling round trips drift)."""
    if samples.shape[-1] >= n:
        return samples[..., :n]
    return np.pad(samples, [(0, 0)] * (samples.ndim - 1) + [(0, n - samples.shape[-1])])


# Short-time frame for the consistency weights, the frames per processing block (~11 s at
# 48 kHz), and an energy floor so a frame where every stem is silent splits the error evenly.
CONSISTENCY_FRAME = 2048
CONSISTENCY_BLOCK_FRAMES = 256
ENERGY_FLOOR = 1e-10


def mixture_consistency(mixture: np.ndarray, stems: dict[str, np.ndarray], frame: int = CONSISTENCY_FRAME) -> dict[str, np.ndarray]:
    """Spread ``mixture - sum(stems)`` over the stems in proportion to their short-time energy.

    Afterwards the stems sum to the mixture (up to float32 rounding), so unmuting every track
    reproduces the upload. Weights are per channel and per ``frame`` samples, so a stem that is
    silent in a section (vocals in an instrumental break) receives ~no error there.

    Works in float32 and in place: float32 writable stems are updated and returned as-is (others
    are converted once). Processing runs in ~11 s blocks, so the extra memory is a few
    block-sized buffers, not a copy of every stem.
    """
    out: dict[str, np.ndarray] = {}
    for name, stem in stems.items():
        arr = np.asarray(stem, dtype=np.float32)
        out[name] = arr if arr.flags.writeable else arr.copy()
    if not out:
        return out
    mix = np.asarray(mixture, dtype=np.float32)
    channels, length = mix.shape
    block = frame * CONSISTENCY_BLOCK_FRAMES
    for start in range(0, length, block):
        end = min(start + block, length)
        n = end - start
        frames = -(-n // frame)
        parts = [stem[:, start:end] for stem in out.values()]  # views, updated in place
        error = mix[:, start:end].copy()
        for part in parts:
            error -= part
        energy = np.stack([
            np.square(np.pad(part, ((0, 0), (0, frames * frame - n)))).reshape(channels, frames, frame).sum(axis=-1)
            for part in parts
        ]) + ENERGY_FLOOR  # (stems, channels, frames)
        weights = energy / energy.sum(axis=0, keepdims=True)
        for part, weight in zip(parts, weights, strict=True):
            part += np.repeat(weight, frame, axis=-1)[:, :n].astype(np.float32) * error
    return out
