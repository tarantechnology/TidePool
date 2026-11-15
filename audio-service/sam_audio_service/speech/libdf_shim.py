"""NumPy re-implementation of DeepFilterNet's Rust feature library (``libdf``, DeepFilterLib 0.5.6).

Why: DeepFilterLib publishes wheels for CPython 3.8-3.11 only and pins numpy < 2, while this
service runs on Python 3.12 / numpy 2 next to torch 2.8. The DeepFilterNet *model* is plain
PyTorch (the ``deepfilternet`` package); only the STFT and the feature normalisation live in
Rust. This module provides exactly the functions ``df`` imports (``DF``, ``erb``, ``erb_inv``,
``erb_norm``, ``unit_norm``, ``unit_norm_init``) with the same maths as libDF/src
(MIT/Apache-2.0, Hendrik Schröter), checked against the real wheel on Python 3.11
(benchmarks/speech.py ``check-libdf``: max abs difference ~1e-6 on every output).

:func:`install` registers it as ``libdf`` (only when the real one is not importable) and stubs
``torchaudio.backend.common`` that ``df.io`` still imports (removed from torchaudio 2.x).
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np

MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)


def _freq2erb(f: float) -> float:
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def _erb2freq(e: float) -> float:
    return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)


def erb_widths(sr: int, fft_size: int, nb_bands: int, min_nb_freqs: int) -> np.ndarray:
    nyq = sr // 2
    freq_width = sr / fft_size
    lo, hi = _freq2erb(0.0), _freq2erb(float(nyq))
    step = (hi - lo) / nb_bands
    widths = [0] * nb_bands
    prev, over = 0, 0
    for i in range(1, nb_bands + 1):
        fb = int(round(_erb2freq(lo + i * step) / freq_width))
        n = fb - prev - over
        if n < min_nb_freqs:
            over = min_nb_freqs - n
            n = min_nb_freqs
        else:
            over = 0
        widths[i - 1] = n
        prev = fb
    widths[-1] += 1  # fft_size / 2 + 1 bins
    too_large = sum(widths) - (fft_size // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    return np.asarray(widths, dtype=np.uint64)


def vorbis_window(size: int) -> np.ndarray:
    half = size // 2
    i = np.arange(size, dtype=np.float64)
    s = np.sin(0.5 * np.pi * (i + 0.5) / half)
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


class DF:
    """Offline STFT state with libDF's framing: frame ``t`` covers samples
    ``[(t + 1) * hop - fft_size, (t + 1) * hop)`` (zeros before the start), Vorbis window, and
    ``wnorm = 2 * hop / fft_size**2`` applied on analysis; synthesis is windowed overlap-add,
    delayed by ``fft_size - hop`` samples exactly as in Rust."""

    def __init__(self, sr: int, fft_size: int, hop_size: int, nb_bands: int, min_nb_erb_freqs: int = 1):
        self._sr, self._fft, self._hop = int(sr), int(fft_size), int(hop_size)
        self._widths = erb_widths(self._sr, self._fft, int(nb_bands), int(min_nb_erb_freqs))
        self.window = vorbis_window(self._fft)
        self.wnorm = 1.0 / (self._fft ** 2 / (2 * self._hop))

    def sr(self) -> int:
        return self._sr

    def fft_size(self) -> int:
        return self._fft

    def hop_size(self) -> int:
        return self._hop

    def erb_widths(self) -> np.ndarray:
        return self._widths.copy()

    def analysis(self, audio: np.ndarray, reset: bool = True) -> np.ndarray:
        """``[C, n]`` float -> ``[C, n // hop, fft // 2 + 1]`` complex64."""
        audio = np.asarray(audio, dtype=np.float32)
        channels, n = audio.shape
        frames = n // self._hop
        pad = self._fft - self._hop
        x = np.concatenate([np.zeros((channels, pad), np.float32), audio[:, : frames * self._hop]], axis=1)
        idx = np.arange(frames)[:, None] * self._hop + np.arange(self._fft)[None, :]
        segs = x[:, idx] * self.window  # [C, T, fft]
        spec = np.fft.rfft(segs, axis=-1) * self.wnorm
        return spec.astype(np.complex64)

    def synthesis(self, spec: np.ndarray, reset: bool = True) -> np.ndarray:
        """``[C, T, F]`` complex -> ``[C, T * hop]`` float32 (overlap-add, libDF delay)."""
        spec = np.asarray(spec)
        channels, frames, _ = spec.shape
        segs = np.fft.irfft(spec, n=self._fft, axis=-1) * self._fft  # libDF's inverse is unnormalised
        segs = (segs * self.window).astype(np.float32)  # [C, T, fft]
        hop, k = self._hop, self._fft // self._hop
        out = np.zeros((channels, (frames + k - 1) * hop), np.float32)
        for j in range(k):  # frame t's j-th hop-sized piece lands at hop-block t + j
            out[:, j * hop: j * hop + frames * hop] += segs[:, :, j * hop:(j + 1) * hop].reshape(channels, frames * hop)
        # Rust emits, at frame t, block t of the overlap-add sum (later blocks are still in memory).
        return out[:, : frames * hop]


def _band_mean(spec_power: np.ndarray, widths: np.ndarray) -> np.ndarray:
    edges = np.concatenate([[0], np.cumsum(widths.astype(np.int64))])
    sums = np.add.reduceat(spec_power, edges[:-1], axis=-1)
    return sums / widths.astype(np.float32)


def erb(spec: np.ndarray, erb_fb: np.ndarray, db: bool = True) -> np.ndarray:
    """``[C, T, F]`` complex -> ``[C, T, B]`` mean band power (dB: ``10 log10(x + 1e-10)``)."""
    power = (np.abs(spec) ** 2).astype(np.float32)
    out = _band_mean(power, np.asarray(erb_fb))
    if db:
        out = 10.0 * np.log10(out + 1e-10)
    return out.astype(np.float32)


def erb_inv(gains: np.ndarray, erb_fb: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(gains, dtype=np.float32), np.asarray(erb_fb).astype(np.int64), axis=-1)


def _ema(x: np.ndarray, alpha: float, init: np.ndarray) -> np.ndarray:
    """``state_t = x_t * (1 - alpha) + state_{t-1} * alpha`` along axis -2 (time)."""
    try:
        from scipy.signal import lfilter
    except ImportError:  # pragma: no cover - scipy ships with the speech extra
        state, out = np.broadcast_to(init, x.shape[:-2] + x.shape[-1:]).astype(np.float64), np.empty_like(x, dtype=np.float64)
        for t in range(x.shape[-2]):
            state = x[..., t, :] * (1 - alpha) + state * alpha
            out[..., t, :] = state
        return out
    zi = np.broadcast_to(alpha * init, x.shape[:-2] + x.shape[-1:])[..., None, :]
    y, _ = lfilter([1 - alpha], [1, -alpha], x.astype(np.float64), axis=-2, zi=zi)
    return y


def erb_norm(erb_feat: np.ndarray, alpha: float, state: np.ndarray | None = None) -> np.ndarray:
    bands = erb_feat.shape[-1]
    init = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], bands) if state is None else np.asarray(state, np.float64)
    mean = _ema(erb_feat, alpha, init)
    return ((erb_feat - mean) / 40.0).astype(np.float32)


def unit_norm_init(num_freq_bins: int) -> np.ndarray:
    return np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], num_freq_bins).astype(np.float32)[None, :]


def unit_norm(spec: np.ndarray, alpha: float, state: np.ndarray | None = None) -> np.ndarray:
    freqs = spec.shape[-1]
    init = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], freqs) if state is None else np.asarray(state, np.float64).reshape(-1)
    mean = _ema(np.abs(spec), alpha, init)
    return (spec / np.sqrt(mean)).astype(np.complex64)


def install() -> str:
    """Make ``import libdf`` and ``df.io`` work. Returns which ``libdf`` is in use."""
    try:
        import libdf

        source = "shim" if getattr(libdf, "_IS_SHIM", False) else "deepfilterlib"
    except ImportError:
        mod = types.ModuleType("libdf")
        mod.DF, mod.erb, mod.erb_inv, mod.erb_norm, mod.unit_norm, mod.unit_norm_init = DF, erb, erb_inv, erb_norm, unit_norm, unit_norm_init
        mod._IS_SHIM = True
        mod.__all__ = ["DF", "erb", "erb_inv", "erb_norm", "unit_norm", "unit_norm_init"]
        sys.modules["libdf"] = mod
        source = "shim"
    try:
        import torchaudio.backend.common  # noqa: F401
    except Exception:  # noqa: BLE001 - removed in torchaudio 2.x; df.io only uses it for a type hint
        backend = sys.modules.get("torchaudio.backend") or types.ModuleType("torchaudio.backend")
        common = types.ModuleType("torchaudio.backend.common")

        class AudioMetaData:  # noqa: D401 - placeholder type
            pass

        common.AudioMetaData = AudioMetaData
        backend.common = common
        sys.modules["torchaudio.backend"] = backend
        sys.modules["torchaudio.backend.common"] = common
    return source
