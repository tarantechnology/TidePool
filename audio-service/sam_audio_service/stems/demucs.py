"""Demucs v4 splitter (``htdemucs``, ``htdemucs_ft``, ``htdemucs_6s``).

``demucs`` and ``torch`` are imported inside :meth:`load` / :meth:`split` only, so the
service and its tests run without them. Install with ``pip install -e ".[music]"``.

Demucs works on stereo at ``model.samplerate`` (44.1 kHz). The service speaks 48 kHz, so the
mixture is resampled on the way in and every stem on the way out, then trimmed or padded to
the input length. Mono input is duplicated to stereo and each stem folded back to mono. Long
songs use Demucs's own overlapping-segment loop (``apply_model(split=True)``).
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .base import Progress, StemSplitter, check_cancel, fit_length, report
from .overlap import overlap_add


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

    def __init__(self, model_name: str = "htdemucs_ft", device: str = "auto", shifts: int = 1, overlap: float = 0.25, segment: float | None = None,
                 batch: int = 1):
        self.model_name = model_name
        self.batch = batch  # segments per forward; > 1 = our batched overlap-add (single-model bags, shifts 0 only)
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
        return {"model_id": self.model_name, "device": self.device, "dtype": "float32", "license": "MIT (Demucs code and htdemucs* weights)",
                "shifts": self.shifts, "overlap": self.overlap, "segment": self.segment, "batch": self.batch}

    def load(self) -> None:
        if self.loaded:
            return
        from demucs.pretrained import get_model

        self.device = resolve_device(self.requested_device)
        model = get_model(self.model_name)
        model.eval()
        self.stems = tuple(model.sources)
        self.model = model

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        """Progress is reported at start and end only, and cancel is checked only before the
        model runs: ``apply_model`` has no per-segment callback (just a tqdm bar)."""
        if not self.loaded:
            raise RuntimeError("Demucs is not loaded")
        check_cancel(cancel)
        report(progress, 0.0, "demucs")
        channels, n = mixture.shape
        stereo = np.repeat(mixture, 2, axis=0) if channels == 1 else mixture
        stems = fit_length(self.separate(stereo, self.sample_rate, out_rate=self.sample_rate), n)
        if channels == 1:
            stems = stems.mean(axis=1, keepdims=True)
        report(progress, 1.0, "demucs")
        return {name: np.ascontiguousarray(stems[i]) for i, name in enumerate(self.model.sources)}

    def separate(self, stereo: np.ndarray, rate: int, out_rate: int | None = None) -> np.ndarray:
        """Stereo ``(2, n)`` at ``rate`` -> ``(sources, 2, m)`` float32 at ``out_rate`` (default:
        the model's rate, so a caller already at 44.1 kHz skips both resamples). Resampling runs
        on the model's device."""
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        out_rate = out_rate or self.model.samplerate
        # Demucs's own CLI normalises by the mono reference's mean/std; the model was trained so.
        ref = stereo.mean(axis=0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        wav = torch.from_numpy(np.ascontiguousarray((stereo - mean) / std, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            wav = convert_audio(wav, rate, self.model.samplerate, self.model.audio_channels)
            if self._batched():
                out = torch.from_numpy(self._run_batched(to_numpy(wav).astype(np.float32, copy=False))).to(self.device)
            else:
                out = apply_model(
                    self.model, wav[None], shifts=self.shifts, split=True, overlap=self.overlap,
                    progress=False, device=self.device, segment=self.segment,
                )[0]  # (sources, channels, frames) at model.samplerate
            out = convert_audio(out, self.model.samplerate, out_rate, 2)
        # Undo the normalisation; the mean goes in once overall (split across stems) so the
        # stems still sum to the mix, as the Phase 0 benchmark runner does.
        return to_numpy(out).astype(np.float32) * std + mean / len(self.model.sources)

    def _batched(self) -> bool:
        """Batched segments only where they are equivalent to ``apply_model``'s loop: one model
        in the bag (htdemucs, htdemucs_6s; not the per-source htdemucs_ft bag) and no shifts."""
        return self.batch > 1 and self.shifts == 0 and len(getattr(self.model, "models", ())) == 1

    def _run_batched(self, wav: np.ndarray) -> np.ndarray:
        """``apply_model(split=True)`` with ``batch`` segments per forward: windows of the model's
        segment length every ``(1 - overlap)`` of it, triangle-weighted overlap-add (as Demucs)."""
        import torch

        model = self.model.models[0].to(self.device)  # apply_model moves the model itself; here we must
        weights = np.asarray(self.model.weights[0], dtype=np.float32)[:, None, None]  # per source
        length = int(round((self.segment or float(model.segment)) * model.samplerate))
        step = max(1, int((1 - self.overlap) * length))

        def run(windows: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return to_numpy(model(torch.from_numpy(windows).to(self.device))) * weights

        return overlap_add(wav, run, length, 1, batch=self.batch, step=step)

    def unload(self) -> None:
        self.model = None
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
