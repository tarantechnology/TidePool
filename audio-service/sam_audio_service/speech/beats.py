"""Beats, downbeats and tempo for /v1/beats ("on the beat drop", "make the bass hit on the beat").

``BeatTracker.track(samples, sample_rate)`` takes float32 ``(channels, n)`` and returns
``{"beats": [s], "downbeats": [s]}`` (seconds on the input timeline). Implementations:

* ``mock``      a fixed 120 BPM grid in 4/4 starting at the first non-silent sample. NOT tracking.
* ``beat_this`` CPJKU "Beat This!" (ISMIR 2024), code and weights MIT. No DBN by default
                (AUDIO_BEATS_DBN=true needs madmom). Runs at 22.05 kHz mono internally.
* ``librosa``   librosa.beat.beat_track (ISC). Beats and tempo only: no downbeats.

Tempo and meter are derived the same way for every tracker (:func:`summarise`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BeatTracker(ABC):
    name = "base"
    is_mock = False
    model_id = ""
    license: str | None = None
    has_downbeats = True

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def track(self, samples: np.ndarray, sample_rate: int) -> dict[str, list[float]]: ...

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "is_mock": self.is_mock, "loaded": self.loaded, "license": self.license, "downbeats": self.has_downbeats}

    def unload(self) -> None:
        return None


class MockBeatTracker(BeatTracker):
    name = "mock"
    is_mock = True
    model_id = "mock-120bpm"

    def __init__(self, bpm: float = 120.0, beats_per_bar: int = 4):
        self.bpm, self.beats_per_bar = bpm, beats_per_bar
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def track(self, samples: np.ndarray, sample_rate: int) -> dict[str, list[float]]:
        mono = np.abs(samples).max(axis=0) if samples.size else np.zeros(0)
        loud = np.flatnonzero(mono > 1e-3)
        if not loud.size:
            return {"beats": [], "downbeats": []}
        start, end = loud[0] / sample_rate, samples.shape[1] / sample_rate
        beats = np.arange(start, end, 60.0 / self.bpm)
        return {"beats": beats.tolist(), "downbeats": beats[:: self.beats_per_bar].tolist()}


# Byte-identical copies of the official checkpoints (verified 2026-11-14: SHA-256 below, and every
# tensor equal to the independent safetensors conversion safe-models/beat-this-final0).
MIRRORS: dict[str, tuple[str, str, str]] = {
    "final0": ("Audante/beat-this-final0", "beat_this-final0.ckpt", "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"),
}


def mirror_checkpoint(name: str) -> str:
    import hashlib

    from huggingface_hub import hf_hub_download

    repo, filename, sha256 = MIRRORS[name]
    path = hf_hub_download(repo, filename)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError(f"beat_this mirror checkpoint {repo}/{filename} has sha256 {digest.hexdigest()}, expected {sha256}")
    return path


class BeatThisTracker(BeatTracker):
    name = "beat_this"
    license = "MIT (code and published weights, CPJKU/beat_this)"

    def __init__(self, checkpoint: str = "final0", device: str = "auto", dbn: bool = False):
        self.model_id, self.requested_device, self.device, self.dbn = checkpoint, device, device, dbn
        self.model: Any = None
        self.source: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {**super().info(), "device": self.device, "dbn": self.dbn, "checkpoint_source": self.source}

    def load(self) -> None:
        if self.loaded:
            return
        from beat_this.inference import Audio2Beats

        from ..stems.demucs import resolve_device

        self.device = resolve_device(self.requested_device)
        try:
            self.model = Audio2Beats(checkpoint_path=self.model_id, device=self.device, dbn=self.dbn)
            self.source = "cpjku" if self.model_id in MIRRORS else "path"
        except ValueError:
            # The official checkpoint host (cloud.cp.jku.at) is a Nextcloud share that has been
            # seen returning HTTP 500 for hours; fall back to an unmodified HF mirror, pinned by hash.
            if self.model_id not in MIRRORS:
                raise
            self.model = Audio2Beats(checkpoint_path=mirror_checkpoint(self.model_id), device=self.device, dbn=self.dbn)
            self.source = "hf-mirror:" + MIRRORS[self.model_id][0]

    def track(self, samples: np.ndarray, sample_rate: int) -> dict[str, list[float]]:
        import torch

        self.load()
        with torch.inference_mode():
            beats, downbeats = self.model(np.ascontiguousarray(samples.T, dtype=np.float32), sample_rate)
        return {"beats": np.asarray(beats, dtype=float).tolist(), "downbeats": np.asarray(downbeats, dtype=float).tolist()}

    def unload(self) -> None:
        self.model = None


class LibrosaBeatTracker(BeatTracker):
    name = "librosa"
    model_id = "librosa.beat.beat_track"
    license = "ISC (librosa)"
    has_downbeats = False
    rate = 22_050

    def __init__(self) -> None:
        self._librosa: Any = None

    @property
    def loaded(self) -> bool:
        return self._librosa is not None

    def load(self) -> None:
        if self._librosa is None:
            import librosa

            self._librosa = librosa

    def track(self, samples: np.ndarray, sample_rate: int) -> dict[str, list[float]]:
        self.load()
        librosa = self._librosa
        mono = samples.mean(axis=0).astype(np.float32)
        y = librosa.resample(mono, orig_sr=sample_rate, target_sr=self.rate) if sample_rate != self.rate else mono
        _, beats = librosa.beat.beat_track(y=y, sr=self.rate, units="time")
        return {"beats": np.asarray(beats, dtype=float).tolist(), "downbeats": []}


def summarise(beats: list[float], downbeats: list[float]) -> dict[str, Any]:
    """Tempo (BPM from the median inter-beat interval) and beats per bar (the most common
    count of beats between consecutive downbeats)."""
    tempo = None
    if len(beats) >= 2:
        ibi = float(np.median(np.diff(beats)))
        tempo = round(60.0 / ibi, 2) if ibi > 0 else None
    meter = None
    if len(downbeats) >= 2 and beats:
        b = np.asarray(beats)
        counts = [int(((b >= d0 - 1e-3) & (b < d1 - 1e-3)).sum()) for d0, d1 in zip(downbeats, downbeats[1:])]
        counts = [c for c in counts if c > 0]
        if counts:
            values, freq = np.unique(counts, return_counts=True)
            meter = int(values[np.argmax(freq)])
    return {"tempo_bpm": tempo, "beats_per_bar": meter}


def create_beat_tracker(settings, device: str) -> BeatTracker | None:
    if settings.beats_backend == "mock":
        return MockBeatTracker()
    if settings.beats_backend == "beat_this":
        return BeatThisTracker(settings.beats_model, device=device, dbn=settings.beats_dbn)
    if settings.beats_backend == "librosa":
        return LibrosaBeatTracker()
    return None
