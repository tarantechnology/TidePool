"""MOCK backend: deterministic DSP stand-in. NOT source separation.

Purpose: let Person 3 and Person 4 integrate against the real HTTP contract, run CI
without a GPU or gated checkpoints, and exercise chunking/residual/stereo logic. It
builds a band-limited version of the mixture whose frequency band is chosen from the
prompt text, honours ``+``/``-`` span anchors and reports energy-based spans. Every
response produced through it is flagged ``model.is_mock = true`` with a warning.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .base import Backend, BackendOptions, BackendPrompt, BackendResult

# Rough, demo-friendly bands so the mock is at least audibly different per prompt.
KEYWORD_BANDS: dict[str, tuple[float, float]] = {
    "siren": (500.0, 2200.0),
    "alarm": (1500.0, 4000.0),
    "speech": (120.0, 3800.0),
    "speaking": (120.0, 3800.0),
    "talking": (120.0, 3800.0),
    "voice": (120.0, 3800.0),
    "man": (85.0, 3000.0),
    "woman": (165.0, 4200.0),
    "person": (120.0, 3800.0),
    "dog": (300.0, 2500.0),
    "bark": (300.0, 2500.0),
    "bird": (2000.0, 9000.0),
    "music": (40.0, 12000.0),
    "guitar": (80.0, 5000.0),
    "drum": (40.0, 400.0),
    "bass": (30.0, 250.0),
    "engine": (30.0, 500.0),
    "car": (30.0, 800.0),
    "traffic": (30.0, 1200.0),
    "rain": (1000.0, 16000.0),
    "wind": (20.0, 600.0),
}


def band_for_prompt(text: str) -> tuple[float, float]:
    words = text.lower().split()
    for word in words:
        for key, band in KEYWORD_BANDS.items():
            if word.startswith(key):
                return band
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    low = 100.0 + (digest[0] / 255.0) * 1500.0
    high = low * (2.0 + (digest[1] / 255.0) * 4.0)
    return low, min(high, 20000.0)


def _stft(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect") if len(x) > pad else np.pad(x, (pad, pad))
    frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(xp[idx] * window, axis=1)


def _istft(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray, length: int) -> np.ndarray:
    frames = spec.shape[0]
    out = np.zeros(n_fft + hop * (frames - 1), dtype=np.float64)
    norm = np.zeros_like(out)
    chunks = np.fft.irfft(spec, n=n_fft, axis=1) * window
    for i in range(frames):
        start = i * hop
        out[start:start + n_fft] += chunks[i]
        norm[start:start + n_fft] += window ** 2
    out = out / np.maximum(norm, 1e-8)
    pad = n_fft // 2
    return out[pad:pad + length]


class MockBackend(Backend):
    name = "mock"
    is_mock = True
    supports_visual = True  # accepts and ignores visual input so the contract path can be tested
    supports_spans = True
    supports_span_prediction = True

    def __init__(self, n_fft: int = 2048, hop: int = 512):
        self.n_fft, self.hop = n_fft, hop
        self._window = np.hanning(n_fft + 1)[:-1].astype(np.float64)
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict[str, Any]:
        return {"model_id": "mock-bandpass", "device": "cpu", "dtype": "float32", "warning": "mock backend: band-pass filter, not source separation"}

    def separate(self, mixture: np.ndarray, prompt: BackendPrompt, options: BackendOptions) -> BackendResult:
        x = mixture.astype(np.float64)
        length = len(x)
        if length == 0:
            return BackendResult(target=mixture.copy(), residual=mixture.copy(), spans=[])
        if length < self.n_fft:
            x = np.pad(x, (0, self.n_fft - length))
        low, high = band_for_prompt(prompt.text or (prompt.visual.label if prompt.visual and prompt.visual.label else "visual"))
        spec = _stft(x, self.n_fft, self.hop, self._window)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        mask = ((freqs >= low) & (freqs <= high)).astype(np.float64)
        target = _istft(spec * mask[None, :], self.n_fft, self.hop, self._window, len(x))[:length]
        if prompt.anchors:
            gate = np.ones(length)
            plus = [a for a in prompt.anchors if a[0] == "+"]
            if plus:
                gate[:] = 0.0
                for _, start, end in plus:
                    gate[int(start * self.sample_rate):int(end * self.sample_rate)] = 1.0
            for token, start, end in prompt.anchors:
                if token == "-":
                    gate[int(start * self.sample_rate):int(end * self.sample_rate)] = 0.0
            target = target * gate
        target = target.astype(np.float32)
        spans = self._energy_spans(target) if options.predict_spans or prompt.anchors else None
        return BackendResult(target=target, residual=(mixture - target).astype(np.float32), spans=spans, extra={"band_hz": [low, high]})

    def _energy_spans(self, target: np.ndarray, frame: int = 4800, threshold_db: float = -40.0) -> list[tuple[float, float]]:
        if len(target) == 0:
            return []
        frames = len(target) // frame
        if frames == 0:
            return [(0.0, len(target) / self.sample_rate)] if np.max(np.abs(target)) > 1e-4 else []
        energy = np.square(target[:frames * frame].reshape(frames, frame)).mean(axis=1)
        peak = energy.max()
        if peak <= 0:
            return []
        active = 10 * np.log10(np.maximum(energy / peak, 1e-12)) > threshold_db
        spans: list[tuple[float, float]] = []
        start = None
        for i, flag in enumerate(active):
            if flag and start is None:
                start = i
            if not flag and start is not None:
                spans.append((start * frame / self.sample_rate, i * frame / self.sample_rate))
                start = None
        if start is not None:
            spans.append((start * frame / self.sample_rate, len(target) / self.sample_rate))
        return [(round(s, 3), round(e, 3)) for s, e in spans]
