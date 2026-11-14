"""Stem splitter interface: one stereo mixture in, N named stems out.

Kept separate from the prompted :class:`~sam_audio_service.backends.base.Backend` (one mono
signal and one prompt in, one target out): a splitter produces every stem in a single model
run, so forcing it through ``Backend.separate`` would mean N runs or an output cache.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


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
    def split(self, mixture: np.ndarray) -> dict[str, np.ndarray]:
        """``mixture`` is float32 ``(channels, n)`` at ``sample_rate``, 1 or 2 channels.

        Returns one float32 array of exactly the same shape per name in :attr:`stems`.
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


def mixture_consistency(mixture: np.ndarray, stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Spread ``mixture - sum(stems)`` over the stems in proportion to their energy.

    Afterwards the stems sum to the mixture (up to float32 rounding), so unmuting every
    track reproduces the upload. Weights are per channel; a channel where every stem is
    silent gets the error split equally.
    """
    if not stems:
        return {}
    names = list(stems)
    stack = np.stack([np.asarray(stems[n], dtype=np.float64) for n in names])  # (S, C, T)
    error = np.asarray(mixture, dtype=np.float64) - stack.sum(axis=0)
    energy = np.square(stack).sum(axis=2, keepdims=True)  # (S, C, 1)
    total = energy.sum(axis=0, keepdims=True)
    weights = np.where(total > 0, energy / np.where(total > 0, total, 1.0), 1.0 / len(names))
    fixed = (stack + weights * error[None]).astype(np.float32)
    return {n: fixed[i] for i, n in enumerate(names)}
