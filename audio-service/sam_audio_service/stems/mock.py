"""MOCK stem splitter: deterministic frequency-band split. NOT source separation.

Lets CI and local development exercise the music-stems contract without weights: each stem
is one band of the mixture's spectrum (bass lowest, drums the cymbal range, and so on). The
bands partition the spectrum, so the stems sum back to the mixture.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..backends.mock import _istft, _stft
from .base import Progress, StemSplitter, check_cancel, report

# (stem, low Hz, high Hz); contiguous so the masks partition the spectrum.
BANDS_4 = (("bass", 0.0, 150.0), ("vocals", 150.0, 3500.0), ("other", 3500.0, 8000.0), ("drums", 8000.0, float("inf")))
BANDS_6 = (
    ("bass", 0.0, 150.0), ("piano", 150.0, 500.0), ("vocals", 500.0, 3000.0),
    ("guitar", 3000.0, 5000.0), ("other", 5000.0, 8000.0), ("drums", 8000.0, float("inf")),
)


class MockStemSplitter(StemSplitter):
    name = "mock"
    is_mock = True

    def __init__(self, six_stems: bool = False, n_fft: int = 2048, hop: int = 512):
        self.bands = BANDS_6 if six_stems else BANDS_4
        self.stems = tuple(name for name, _, _ in self.bands)
        self.model_id = "mock-bands-6s" if six_stems else "mock-bands"
        self.n_fft, self.hop = n_fft, hop
        self._window = np.hanning(n_fft + 1)[:-1]
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "device": "cpu", "dtype": "float32", "license": "n/a (no weights)", "warning": "mock stem splitter: frequency bands, not source separation"}

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        n = mixture.shape[1]
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        out = {name: np.zeros(mixture.shape, dtype=np.float32) for name in self.stems}
        for c in range(mixture.shape[0]):
            check_cancel(cancel)
            report(progress, c / mixture.shape[0], "bands")
            x = np.pad(mixture[c].astype(np.float64), (0, max(0, self.n_fft - n)))
            spec = _stft(x, self.n_fft, self.hop, self._window)
            for name, low, high in self.bands:
                mask = ((freqs >= low) & (freqs < high)).astype(np.float64)
                out[name][c] = _istft(spec * mask[None, :], self.n_fft, self.hop, self._window, len(x))[:n]
        report(progress, 1.0, "done")
        return out
