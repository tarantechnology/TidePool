"""Chunked overlap-add inference for models that take fixed-length windows (RoFormer).

Model-agnostic and pure numpy: ``model`` maps a batch ``(B, channels, chunk)`` to
``(B, stems, channels, chunk)``, so the windowing can be unit-tested with fake models.

The signal is reflect-padded by ``chunk - step`` on both sides (so the real samples never sit
at a lone window edge), cut into windows every ``step = chunk // overlap`` samples (the last
window zero-padded), run in batches, and overlap-added with a triangular window normalised by
the summed window weights. The window never reaches zero, so the weights are positive
everywhere and an identity model reconstructs the input exactly. No clipping is applied.
Approach after ZFTurbo's Music-Source-Separation-Training demix loop (MIT); written here.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from .base import Progress, check_cancel, report


def triangular_window(size: int) -> np.ndarray:
    """Strictly positive triangle: 1 in the middle, 1/size at the ends."""
    centres = (np.arange(size) + 0.5) / size
    return (1.0 - np.abs(2.0 * centres - 1.0)).astype(np.float32)


def overlap_add(
    x: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    chunk: int,
    overlap: int,
    batch: int = 1,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray:
    """Run ``model`` over ``x`` ``(channels, n)`` and return ``(stems, channels, n)`` float32."""
    if chunk <= 0 or overlap < 1 or batch < 1:
        raise ValueError("chunk must be positive and overlap, batch at least 1")
    channels, n = x.shape
    step = max(1, chunk // overlap)
    border = chunk - step
    if border:
        x = np.pad(x, ((0, 0), (border, border)), mode="reflect" if n > border else "constant")
    length = x.shape[1]
    starts = list(range(0, max(length - border, 1), step))
    window = triangular_window(chunk)
    out: np.ndarray | None = None
    weight = np.zeros(length + chunk, dtype=np.float32)
    for first in range(0, len(starts), batch):
        check_cancel(cancel)
        group = starts[first : first + batch]
        windows = np.zeros((len(group), channels, chunk), dtype=np.float32)
        for i, start in enumerate(group):
            piece = x[:, start : start + chunk]
            windows[i, :, : piece.shape[1]] = piece
        result = np.asarray(model(windows), dtype=np.float32)
        if result.ndim != 4 or result.shape[0] != len(group) or result.shape[2:] != (channels, chunk):
            raise ValueError(f"model returned {result.shape} for a batch of shape {windows.shape}")
        if out is None:
            out = np.zeros((result.shape[1], channels, length + chunk), dtype=np.float32)
        for i, start in enumerate(group):
            out[:, :, start : start + chunk] += result[i] * window
            weight[start : start + chunk] += window
        report(progress, (first + len(group)) / len(starts), "chunks")
    assert out is not None
    return out[:, :, border : border + n] / weight[border : border + n]
