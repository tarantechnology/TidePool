"""Speech enhancement for /v1/enhance: DeepFilterNet3 (denoise a dialogue/vocal layer).

``Enhancer.enhance(samples_48k, atten_lim_db)`` takes float32 ``(channels, n)`` at 48 kHz (the
model's native rate; the caller resamples around it) and returns the same shape. Channels are
processed independently as a batch. Implementations:

* ``mock``          a fixed spectral gate (STFT magnitude below a per-bin noise floor is
                    attenuated). Deterministic, CPU, for tests. NOT a learned model.
* ``deepfilternet`` DeepFilterNet3 (Schröter et al. 2023). Code MIT OR Apache-2.0; the
                    pretrained models ship in the same repository under the same licence.
                    The ``deepfilternet`` package provides the PyTorch model; its Rust
                    feature library is replaced by :mod:`.libdf_shim` on Python >= 3.12.

``atten_lim_db`` is DeepFilterNet's attenuation limit: the output is
``lim * noisy + (1 - lim) * enhanced`` in the STFT domain with ``lim = 10^(-dB/20)``, so 6 dB is
gentle, 12-20 dB moderate and ``None`` removes as much noise as the model wants.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

log = logging.getLogger("sam_audio_service")

RATE = 48_000


class Enhancer(ABC):
    name = "base"
    is_mock = False
    model_id = ""
    license: str | None = None
    sample_rate = RATE

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def enhance(self, samples: np.ndarray, atten_lim_db: float | None) -> np.ndarray: ...

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "is_mock": self.is_mock, "loaded": self.loaded, "license": self.license, "sample_rate": self.sample_rate}

    def unload(self) -> None:
        return None


def _limit(noisy: np.ndarray, enhanced: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
    if atten_lim_db is None or atten_lim_db <= 0:
        return enhanced
    lim = 10 ** (-abs(atten_lim_db) / 20)
    return lim * noisy + (1 - lim) * enhanced


class MockEnhancer(Enhancer):
    """Per-bin spectral gate: bins below 3x their median magnitude over time are muted."""

    name = "mock"
    is_mock = True
    model_id = "mock-spectral-gate"
    n_fft = 1024

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def enhance(self, samples: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
        n_fft, hop = self.n_fft, self.n_fft // 2
        window = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)
        out = np.zeros_like(samples)
        n = samples.shape[1]
        padded = np.pad(samples, ((0, 0), (n_fft, n_fft + n_fft)))
        frames = (padded.shape[1] - n_fft) // hop + 1
        idx = np.arange(frames)[:, None] * hop + np.arange(n_fft)[None, :]
        for c in range(samples.shape[0]):
            spec = np.fft.rfft(padded[c][idx] * window, axis=-1)
            mag = np.abs(spec)
            floor = 3.0 * np.median(mag, axis=0, keepdims=True)
            gated = spec * (mag > floor)
            spec = _limit(spec, gated, atten_lim_db)
            seg = np.fft.irfft(spec, n=n_fft, axis=-1) * window
            acc = np.zeros(padded.shape[1])
            for t in range(frames):
                acc[t * hop: t * hop + n_fft] += seg[t]
            out[c] = acc[n_fft: n_fft + n]
        return out.astype(np.float32)


class DeepFilterNetEnhancer(Enhancer):
    name = "deepfilternet"
    license = "MIT OR Apache-2.0 (code and pretrained DeepFilterNet3 weights, Rikorose/DeepFilterNet)"
    context_seconds = 2.0  # GRU warm-up on each side of a chunk

    def __init__(self, model: str = "DeepFilterNet3", device: str = "auto", chunk_seconds: float = 60.0):
        self.model_id, self.requested_device, self.device = model, device, device
        self.chunk_seconds = chunk_seconds
        self.model: Any = None
        self.df_state: Any = None
        self.libdf: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {**super().info(), "device": self.device, "libdf": self.libdf, "chunk_seconds": self.chunk_seconds}

    def load(self) -> None:
        if self.loaded:
            return
        from . import libdf_shim

        self.libdf = libdf_shim.install()
        import torch
        from df import config as df_config  # noqa: F401 - initialises df's global config
        from df.enhance import init_df
        from loguru import logger as df_logger

        from ..stems.demucs import resolve_device

        df_logger.remove()  # DeepFilterNet logs every step at INFO through loguru
        self.device = resolve_device(self.requested_device)
        # init_df places the model on df.modules.get_device() (cuda if available); move it after.
        model, df_state, _ = init_df(model_base_dir=self.model_id, post_filter=False, log_level="ERROR", log_file=None)
        self.model = model.to(self.device).eval()
        self.df_state = df_state
        self._torch = torch

    def _run(self, x: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
        """One window, ``(channels, n)`` -> same; mirrors df.enhance.enhance (pad + delay crop)."""
        import torch
        from df.enhance import df_features
        from df.utils import as_complex

        state = self.df_state
        n = x.shape[1]
        fft, hop = state.fft_size(), state.hop_size()
        audio = torch.from_numpy(np.pad(x, ((0, 0), (0, fft))).astype(np.float32))
        nb_df = getattr(self.model, "nb_df", getattr(self.model, "df_bins", 96))
        spec, erb_feat, spec_feat = df_features(audio, state, nb_df, device=self.device)
        with torch.inference_mode():
            enhanced = self.model(spec.clone(), erb_feat, spec_feat)[0]
        enhanced = as_complex(enhanced.squeeze(1).float().cpu()).numpy()
        noisy = as_complex(spec.squeeze(1).float().cpu()).numpy()
        out = state.synthesis(_limit(noisy, enhanced, atten_lim_db))
        d = fft - hop
        return np.ascontiguousarray(out[:, d: n + d], dtype=np.float32)

    def enhance(self, samples: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
        self.load()
        n = samples.shape[1]
        chunk = int(self.chunk_seconds * RATE)
        if not chunk or n <= chunk:
            return self._run(samples, atten_lim_db)
        ctx = int(self.context_seconds * RATE)
        out = np.zeros_like(samples, dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            a, b = max(0, start - ctx), min(n, end + ctx)
            y = self._run(samples[:, a:b], atten_lim_db)
            out[:, start:end] = y[:, start - a: start - a + (end - start)]
        return out

    def unload(self) -> None:
        self.model = self.df_state = None


def create_enhancer(settings, device: str) -> Enhancer | None:
    if settings.enhance_backend == "mock":
        return MockEnhancer()
    if settings.enhance_backend == "deepfilternet":
        return DeepFilterNetEnhancer(settings.enhance_model, device=device, chunk_seconds=settings.enhance_chunk_seconds)
    return None
