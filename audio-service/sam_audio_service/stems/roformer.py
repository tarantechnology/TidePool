"""Native multi-stem BS-RoFormer splitter (e.g. ZFTurbo's 4-stem MUSDB checkpoint).

The model is built from the training YAML with the ``bs_roformer`` package (lucidrains,
MIT) and the checkpoint from ``AUDIO_STEM_ROFORMER_CKPT`` (ZFTurbo
Music-Source-Separation-Training weights are MIT). ``torch``, ``bs_roformer`` and ``yaml`` are
imported in :meth:`load` / :meth:`split` only; install with ``pip install -e ".[music-roformer]"``.

Inference: 48 kHz -> the config's sample rate, fixed-length windows overlap-added by
:func:`~sam_audio_service.stems.overlap.overlap_add` (``audio.chunk_size``, step =
chunk // ``AUDIO_STEM_ROFORMER_OVERLAP``), back to 48 kHz, trimmed to the input length.
"""
from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ..audio_io import resample
from .base import Progress, StemSplitter, check_cancel, fit_length, report
from .demucs import resolve_device, to_numpy
from .overlap import overlap_add
from .vocabulary import canonical_stem

log = logging.getLogger("sam_audio_service")

# Loading with strict=False tolerates renamed buffers between bs_roformer versions; more
# missing parameters than this means the YAML and checkpoint do not describe the same model.
MAX_MISSING_FRACTION = 0.02


def _tuples(value: Any) -> Any:
    """YAML gives lists; BSRoformer expects tuples (e.g. ``freqs_per_bands``)."""
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def constructor_kwargs(cls: type, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``config`` filtered to what ``cls.__init__`` accepts. Returns ``(kwargs, dropped keys)``."""
    params = inspect.signature(cls).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: _tuples(v) for k, v in config.items()}, []
    kwargs = {k: _tuples(v) for k, v in config.items() if k in params}
    return kwargs, sorted(set(config) - set(kwargs))


def unwrap_state(state: Any) -> dict[str, Any]:
    """Checkpoints come as a bare state dict or wrapped in ``state_dict`` / ``model``."""
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            return state[key]
    return state


class RoformerSplitter(StemSplitter):
    name = "roformer"

    def __init__(self, checkpoint: str, config: str, device: str = "auto", overlap: int = 2, fp16: bool = False, batch: int = 1,
                 resampler: Callable[[np.ndarray, int, int], np.ndarray] = resample):
        self.checkpoint, self.config_path = checkpoint, config
        self.requested_device = self.device = device
        self.overlap, self.fp16, self.batch = overlap, fp16, batch
        self.resample = resampler
        self.model: Any = None
        self.model_rate = 44_100
        self.chunk = 0
        self.instruments: list[str] = []
        self.stems = ("vocals", "drums", "bass", "other")  # replaced by the config's instruments on load

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {"model_id": self.checkpoint.rsplit("/", 1)[-1], "device": self.device, "dtype": "float16" if self.fp16 else "float32",
                "license": "MIT (ZFTurbo BS-RoFormer weights; bs_roformer code)", "chunk_size": self.chunk or None,
                "overlap": self.overlap, "batch": self.batch, "sample_rate_model": self.model_rate}

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        import yaml
        from bs_roformer import BSRoformer

        with open(self.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kwargs, dropped = constructor_kwargs(BSRoformer, cfg["model"])
        if dropped:
            log.info("BSRoformer does not take %s from %s; ignored", ", ".join(dropped), self.config_path)
        model = BSRoformer(**kwargs)
        state = unwrap_state(torch.load(self.checkpoint, map_location="cpu", weights_only=True))
        missing, unexpected = model.load_state_dict(state, strict=False)
        total = max(1, len(model.state_dict()))
        if len(missing) > MAX_MISSING_FRACTION * total:
            raise RuntimeError(f"{len(missing)} of {total} parameters missing from {self.checkpoint} (e.g. {list(missing)[:3]}); "
                               "the YAML and checkpoint do not match")
        if missing or unexpected:
            log.warning("RoFormer checkpoint: %d missing, %d unexpected keys", len(missing), len(unexpected))
        self.device = resolve_device(self.requested_device)
        model = model.to(self.device).eval()
        self.model = model.half() if self.fp16 else model
        self.model_rate = int(cfg.get("audio", {}).get("sample_rate", 44_100))
        self.chunk = int(cfg["audio"]["chunk_size"])
        self.instruments = list(cfg["training"]["instruments"])
        self.stems = tuple(dict.fromkeys(canonical_stem(i) or "other" for i in self.instruments))

    def _run(self, batch: np.ndarray) -> np.ndarray:
        import torch

        dtype = torch.float16 if self.fp16 else torch.float32
        with torch.inference_mode():
            y = self.model(torch.from_numpy(batch).to(self.device, dtype=dtype))
        y = to_numpy(y.float() if hasattr(y, "float") else y)
        return y[:, None] if y.ndim == 3 else y  # a single-stem model returns (B, C, T)

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("RoFormer is not loaded")
        check_cancel(cancel)
        channels, n = mixture.shape
        stereo = np.repeat(mixture, 2, axis=0) if channels == 1 else mixture
        x = self.resample(np.ascontiguousarray(stereo, dtype=np.float32), self.sample_rate, self.model_rate)
        report(progress, 0.0, "chunks")
        est = overlap_add(x, self._run, self.chunk, self.overlap, self.batch, progress=progress, cancel=cancel)
        if est.shape[0] != len(self.instruments):
            raise RuntimeError(f"model returned {est.shape[0]} stems; the config lists {self.instruments}")
        out: dict[str, np.ndarray] = {}
        for i, instrument in enumerate(self.instruments):
            stem = fit_length(self.resample(est[i], self.model_rate, self.sample_rate), n)
            if channels == 1:
                stem = stem.mean(axis=0, keepdims=True)
            name = canonical_stem(instrument) or "other"  # unknown instruments fold into other
            out[name] = out[name] + stem if name in out else np.ascontiguousarray(stem, dtype=np.float32)
        return out

    def unload(self) -> None:
        self.model = None
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
