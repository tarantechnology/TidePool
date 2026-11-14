"""Demucs v4 splitter (``htdemucs``, ``htdemucs_ft``, ``htdemucs_6s``).

``demucs`` and ``torch`` are imported inside :meth:`load` / :meth:`split` only, so the
service and its tests run without them. Install with ``pip install -e ".[music]"``.

Demucs works on stereo at ``model.samplerate`` (44.1 kHz). The service speaks 48 kHz, so the
mixture is resampled on the way in and every stem on the way out, then trimmed or padded to
the input length. Mono input is duplicated to stereo and each stem folded back to mono. Long
songs use Demucs's own overlapping-segment loop (``apply_model(split=True)``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import StemSplitter, fit_length


def resolve_device(requested: str) -> str:
    """``auto`` picks cuda > mps > cpu, as the prompted backends do."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)


class DemucsSplitter(StemSplitter):
    name = "demucs"

    def __init__(self, model_name: str = "htdemucs_ft", device: str = "auto", shifts: int = 1, overlap: float = 0.25, segment: float | None = None):
        self.model_name = model_name
        self.requested_device = device
        self.device = device
        self.shifts, self.overlap, self.segment = shifts, overlap, segment
        self.model: Any = None
        # Known before load so /capabilities can list them without importing torch.
        self.stems = ("drums", "bass", "other", "vocals", "guitar", "piano") if model_name.endswith("6s") else ("drums", "bass", "other", "vocals")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {"model_id": self.model_name, "device": self.device, "dtype": "float32", "shifts": self.shifts, "overlap": self.overlap, "segment": self.segment}

    def load(self) -> None:
        if self.loaded:
            return
        from demucs.pretrained import get_model

        self.device = resolve_device(self.requested_device)
        model = get_model(self.model_name)
        model.eval()
        self.stems = tuple(model.sources)
        self.model = model

    def split(self, mixture: np.ndarray) -> dict[str, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("Demucs is not loaded")
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        channels, n = mixture.shape
        stereo = np.repeat(mixture, 2, axis=0) if channels == 1 else mixture
        # Demucs's own CLI normalises by the mono reference's mean/std; the model was trained so.
        ref = stereo.mean(axis=0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        wav = torch.from_numpy(np.ascontiguousarray((stereo - mean) / std, dtype=np.float32))
        wav = convert_audio(wav, self.sample_rate, self.model.samplerate, self.model.audio_channels)
        with torch.inference_mode():
            out = apply_model(
                self.model, wav[None], shifts=self.shifts, split=True, overlap=self.overlap,
                progress=False, device=self.device, segment=self.segment,
            )[0]  # (sources, channels, frames) at model.samplerate
            out = convert_audio(out, self.model.samplerate, self.sample_rate, 2)
        stems = to_numpy(out).astype(np.float32) * std + mean
        stems = fit_length(stems, n)
        if channels == 1:
            stems = stems.mean(axis=1, keepdims=True)
        return {name: np.ascontiguousarray(stems[i]) for i, name in enumerate(self.model.sources)}

    def unload(self) -> None:
        self.model = None
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
