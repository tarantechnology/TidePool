"""Native multi-stem BS-RoFormer splitter (e.g. ZFTurbo's 4-stem MUSDB checkpoint).

The model is built from the training YAML with the ``bs_roformer`` package (lucidrains,
MIT) and the checkpoint from ``AUDIO_STEM_ROFORMER_CKPT`` (ZFTurbo
Music-Source-Separation-Training weights are MIT). ``torch``, ``bs_roformer`` and ``yaml`` are
imported in :meth:`load` / :meth:`split` only; install with ``pip install -e ".[music-roformer]"``.

Inference: 48 kHz -> the config's sample rate, fixed-length windows overlap-added by
:func:`~sam_audio_service.stems.overlap.overlap_add` (``audio.chunk_size``, step =
chunk // ``AUDIO_STEM_ROFORMER_OVERLAP``), back to 48 kHz, trimmed to the input length.

``fp16`` runs the forward pass under CUDA autocast (weights stay float32; STFT/iSTFT stay
float32) and lets scaled-dot-product attention use the flash kernel: bs_roformer only enables
flash on A100 and falls back to math attention elsewhere (A40, L4...), which is most of the
cost. Single-target checkpoints (``num_stems: 1`` + ``training.target_instrument``, e.g. the
ViperX vocal model) return the target plus its complement (mixture - target) as ``other``.
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


def config_loader(yaml: Any) -> type:
    """``yaml.SafeLoader`` that also reads ``!!python/tuple`` (audio-separator's ViperX YAML
    uses it for ``freqs_per_bands``); no other python tags are allowed."""

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    return Loader


def unwrap_state(state: Any) -> dict[str, Any]:
    """Checkpoints come as a bare state dict or wrapped in ``state_dict`` / ``model``."""
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            return state[key]
    return state


def enable_flash(model: Any) -> int:
    """Allow the flash SDPA kernel in every bs_roformer ``Attend`` (it only does so on A100).
    Flash needs fp16/bf16 inputs, i.e. autocast; math / mem-efficient remain as fallbacks."""
    changed = 0
    for module in model.modules():
        config = getattr(module, "cuda_config", None)
        if config is not None and hasattr(config, "_replace") and "enable_flash" in config._fields:
            module.cuda_config = config._replace(enable_flash=True, enable_mem_efficient=True)
            changed += 1
    return changed


class FastRotary:
    """Drop-in for ``rotary_embedding_torch.RotaryEmbedding.rotate_queries_or_keys`` as
    bs_roformer's ``Attention`` calls it (full-width rotary, no xpos, offset 0): the same
    float32 rotation, as one complex multiply with cos/sin cached per sequence length. The
    library recomputes cos/sin and runs ~10 separate float32 elementwise kernels per call over
    every query and key tensor, which is a third of ViperX's inference time on an A40."""

    def __init__(self, rotary: Any):
        self.rotary = rotary
        self.original = rotary.rotate_queries_or_keys
        self.cache: dict[tuple[int, Any], Any] = {}

    @staticmethod
    def supports(rotary: Any) -> bool:
        return (hasattr(rotary, "freqs") and not getattr(rotary, "use_xpos", False) and not getattr(rotary, "learned_freq", False)
                and getattr(rotary, "interpolate_factor", 1.0) == 1.0 and getattr(rotary, "freqs_for", "lang") == "lang")

    def rotate_queries_or_keys(self, t: Any, seq_dim: int | None = None, offset: int = 0, scale: Any = None) -> Any:
        import torch

        n, d = t.shape[-2], t.shape[-1]
        if seq_dim not in (None, -2) or offset or scale is not None or 2 * self.rotary.freqs.shape[-1] != d or d % 2:
            return self.original(t, seq_dim=seq_dim, offset=offset, scale=scale)
        key = (n, t.device)
        rot = self.cache.get(key)
        if rot is None:
            with torch.autocast(t.device.type, enabled=False):
                angles = torch.outer(torch.arange(n, device=t.device, dtype=torch.float32), self.rotary.freqs.detach().float())
                rot = self.cache[key] = torch.polar(torch.ones_like(angles), angles)  # (n, d/2) complex64
        with torch.autocast(t.device.type, enabled=False):
            pairs = torch.view_as_complex(t.float().unflatten(-1, (d // 2, 2)).contiguous())
            return torch.view_as_real(pairs * rot).flatten(-2).to(t.dtype)


def fast_rotary(model: Any) -> int:
    """Route every rotary embedding that bs_roformer's attention uses through :class:`FastRotary`
    (patched on the instance; embeddings shared between layers are patched once). Returns how
    many embeddings were patched."""
    patched = 0
    for module in model.modules():
        rotary = getattr(module, "rotary_embed", None)
        if rotary is None or getattr(rotary, "_fast_rotary", False) or not FastRotary.supports(rotary):
            continue
        rotary.rotate_queries_or_keys = FastRotary(rotary).rotate_queries_or_keys
        rotary._fast_rotary = True
        patched += 1
    return patched


class RoformerSplitter(StemSplitter):
    name = "roformer"

    def __init__(self, checkpoint: str, config: str, device: str = "auto", overlap: int = 2, fp16: bool = False, batch: int = 1,
                 resampler: Callable[[np.ndarray, int, int], np.ndarray] = resample, license: str = "MIT (ZFTurbo BS-RoFormer weights; bs_roformer code)",
                 compile: bool = False):
        self.checkpoint, self.config_path = checkpoint, config
        self.requested_device = self.device = device
        self.overlap, self.fp16, self.batch = overlap, fp16, batch
        self.resample = resampler
        self.model: Any = None
        self.model_rate = 44_100
        self.chunk = 0
        self.instruments: list[str] = []
        self.complement: str | None = None  # single-target models: stem name for mixture - target
        self.license = license
        self.compile = compile  # torch.compile (CUDA only): fixed batch shape, compiled on the first split
        self.stems = ("vocals", "drums", "bass", "other")  # replaced by the config's instruments on load

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {"model_id": self.checkpoint.rsplit("/", 1)[-1], "device": self.device, "dtype": "float16" if self.fp16 else "float32",
                "license": self.license, "chunk_size": self.chunk or None,
                "overlap": self.overlap, "batch": self.batch, "compiled": self.compile, "sample_rate_model": self.model_rate}

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        import yaml
        from bs_roformer import BSRoformer

        with open(self.config_path, encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=config_loader(yaml))  # noqa: S506 - SafeLoader plus python/tuple
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
        self.model = model.to(self.device).eval()
        if self.device.startswith("cuda"):
            fast_rotary(model)
            if self.fp16:
                enable_flash(model)
        if self.compile and self.device.startswith("cuda"):
            self.model = torch.compile(model, dynamic=False)
        self.model_rate = int(cfg.get("audio", {}).get("sample_rate", 44_100))
        self.chunk = int(cfg["audio"]["chunk_size"])
        self.instruments = list(cfg["training"]["instruments"])
        target = cfg["training"].get("target_instrument")
        if int(cfg["model"].get("num_stems", len(self.instruments))) == 1 and target:
            self.instruments = [target]
            self.complement = "other"
        names = [canonical_stem(i) or "other" for i in self.instruments] + ([self.complement] if self.complement else [])
        self.stems = tuple(dict.fromkeys(names))

    def _run(self, batch: np.ndarray) -> np.ndarray:
        import torch

        n = batch.shape[0]
        if self.compile and n < self.batch:  # keep one input shape so the compiled graph is reused
            batch = np.concatenate([batch, np.zeros((self.batch - n, *batch.shape[1:]), dtype=batch.dtype)])
        autocast = self.fp16 and self.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            y = self.model(torch.from_numpy(batch).to(self.device))
        y = to_numpy(y.float() if hasattr(y, "float") else y)[:n]
        return y[:, None] if y.ndim == 3 else y  # a single-stem model returns (B, C, T)

    def separate(self, x: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        """Stereo ``x`` ``(2, n)`` at ``model_rate`` -> ``{stem: (2, n)}`` at ``model_rate``."""
        if not self.loaded:
            raise RuntimeError("RoFormer is not loaded")
        est = overlap_add(np.ascontiguousarray(x, dtype=np.float32), self._run, self.chunk, self.overlap, self.batch, progress=progress, cancel=cancel)
        if est.shape[0] != len(self.instruments):
            raise RuntimeError(f"model returned {est.shape[0]} stems; the config lists {self.instruments}")
        out: dict[str, np.ndarray] = {}
        for i, instrument in enumerate(self.instruments):
            name = canonical_stem(instrument) or "other"  # unknown instruments fold into other
            out[name] = out[name] + est[i] if name in out else est[i]
        if self.complement:
            out[self.complement] = out.get(self.complement, 0) + (x - est.sum(axis=0))
        return out

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("RoFormer is not loaded")
        check_cancel(cancel)
        channels, n = mixture.shape
        stereo = np.repeat(mixture, 2, axis=0) if channels == 1 else mixture
        x = self.resample(np.ascontiguousarray(stereo, dtype=np.float32), self.sample_rate, self.model_rate)
        report(progress, 0.0, "chunks")
        out: dict[str, np.ndarray] = {}
        for name, est in self.separate(x, progress=progress, cancel=cancel).items():
            stem = fit_length(self.resample(est, self.model_rate, self.sample_rate), n)
            if channels == 1:
                stem = stem.mean(axis=0, keepdims=True)
            out[name] = np.ascontiguousarray(stem, dtype=np.float32)
        return out

    def unload(self) -> None:
        self.model = None
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
