"""Pure-numpy DSP used by music analysis, rendering and mix-in.

Nothing here needs scipy or librosa, so the whole generate pipeline (with the built-in synth and
the rule-based composer) runs in CI on CPU. Heavier optional backends (beat_this, Basic Pitch,
FluidSynth, AMT, MusicGen) plug in on top.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-10


def as_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=0) if x.ndim == 2 else x


def _smooth(m: int) -> int:
    """Smallest integer >= m whose only prime factors are 2, 3 and 5 (fast FFT sizes)."""
    best = 1 << max(0, int(np.ceil(np.log2(max(m, 1)))))
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < m:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def fast_len(n: int) -> int:
    return _smooth(n)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resample along the last axis. soxr when installed; else an FFT resample whose
    sizes are padded to fast (2/3/5-smooth) lengths: a 4-min song at an awkward length took ~3 s
    per call with plain numpy FFTs, ~0.2 s padded."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    n_in = x.shape[-1]
    n_out = int(round(n_in * sr_out / sr_in))
    try:
        import soxr

        flat = x.reshape(-1, n_in)
        out = np.stack([soxr.resample(c, sr_in, sr_out, quality="HQ") for c in flat])
        out = out[:, :n_out] if out.shape[1] >= n_out else np.pad(out, ((0, 0), (0, n_out - out.shape[1])))
        return out.reshape(x.shape[:-1] + (n_out,)).astype(np.float32)
    except ImportError:
        pass
    g = int(np.gcd(int(sr_in), int(sr_out)))
    a, b = int(sr_in) // g, int(sr_out) // g
    m = _smooth(-(-n_in // a))
    n_pad, n_pad_out = m * a, m * b
    spec = np.fft.rfft(x, n=n_pad, axis=-1)
    keep = min(spec.shape[-1], n_pad_out // 2 + 1)
    out_spec = np.zeros(x.shape[:-1] + (n_pad_out // 2 + 1,), dtype=np.complex64)
    out_spec[..., :keep] = spec[..., :keep]
    return (np.fft.irfft(out_spec, n=n_pad_out, axis=-1)[..., :n_out] * (n_pad_out / n_pad)).astype(np.float32)


def frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    pad = n_fft // 2
    x = np.pad(np.asarray(x, dtype=np.float32), (pad, pad))
    n = 1 + max(0, (len(x) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """|STFT| as (frames, bins); frame t is centred on sample t*hop."""
    win = np.hanning(n_fft).astype(np.float32)
    out = []
    fr = frames(x, n_fft, hop)
    for i in range(0, len(fr), 2048):  # bound memory on long songs
        out.append(np.abs(np.fft.rfft(fr[i:i + 2048] * win, axis=1)).astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, n_fft // 2 + 1), np.float32)


def stft(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    win = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)
    return np.fft.rfft(frames(x, n_fft, hop) * win, axis=1)


def istft(spec: np.ndarray, n: int, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Inverse of :func:`stft` (sqrt-Hann analysis/synthesis, weighted overlap-add)."""
    win = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)
    fr = np.fft.irfft(spec, n=n_fft, axis=1) * win
    total = n_fft + hop * (len(fr) - 1)
    out = np.zeros(total, np.float32)
    norm = np.zeros(total, np.float32)
    for i, f in enumerate(fr):
        out[i * hop:i * hop + n_fft] += f
        norm[i * hop:i * hop + n_fft] += win ** 2
    out = out / np.maximum(norm, 1e-3)
    pad = n_fft // 2
    return out[pad:pad + n] if len(out) >= pad + n else np.pad(out[pad:], (0, pad + n - len(out)))


def rms_envelope(x: np.ndarray, sr: int, hop_s: float = 0.01, win_s: float = 0.05) -> np.ndarray:
    x = as_mono(x)
    hop, win = max(1, int(sr * hop_s)), max(2, int(sr * win_s))
    fr = frames(x, win, hop)
    return np.sqrt(np.mean(fr ** 2, axis=1) + EPS).astype(np.float32)


def onset_envelope(x: np.ndarray, sr: int, hop: int = 512, n_fft: int = 1024, fmin: float = 0, fmax: float | None = None) -> np.ndarray:
    """Log-magnitude spectral flux, half-wave rectified, mean over bins (optionally a band)."""
    mag = stft_mag(as_mono(x), n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= fmin) & (freqs <= (fmax or sr / 2))
    logm = np.log1p(100 * mag[:, band])
    flux = np.maximum(0, np.diff(logm, axis=0, prepend=logm[:1])).mean(axis=1)
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")  # local mean removal
    return np.maximum(flux, 0).astype(np.float32)


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    size = fast_len(max(n, 2))
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:n].astype(np.float32)


def fft_eq(x: np.ndarray, sr: int, gain_fn) -> np.ndarray:
    """Zero-phase static EQ: multiply the spectrum by gain_fn(freqs) (linear gain)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    size = fast_len(n + 2048)  # zero-padded to a fast FFT size (also avoids wrap-around at the ends)
    spec = np.fft.rfft(x, n=size, axis=-1)
    g = gain_fn(np.fft.rfftfreq(size, 1 / sr)).astype(np.float32)
    return np.fft.irfft(spec * g, n=size, axis=-1)[..., :n].astype(np.float32)


def highpass_gain(f: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    return 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order))


def lowpass_gain(f: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    return 1 / np.sqrt(1 + (f / fc) ** (2 * order))


def peak_gain(f: np.ndarray, fc: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Log-frequency bell: gain_db at fc, half-width ~ 1/q octaves."""
    oct_ = np.log2(np.maximum(f, 1.0) / fc)
    return 10 ** (gain_db * np.exp(-0.5 * (oct_ * q * 2.355 / 2) ** 2) / 20)


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def attack_release(env: np.ndarray, rate: float, attack_s: float, release_s: float) -> np.ndarray:
    """One-pole attack/release follower on a (short) control-rate envelope."""
    a = np.exp(-1 / max(1, attack_s * rate))
    r = np.exp(-1 / max(1, release_s * rate))
    out = np.empty_like(env)
    y = 0.0
    for i, v in enumerate(env):
        c = a if v > y else r
        y = c * y + (1 - c) * v
        out[i] = y
    return out


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Approximate BS.1770 K-weighting (high shelf +4 dB above ~1.5 kHz, highpass ~60 Hz)."""
    def g(f):
        shelf = 10 ** (4 / 20 * (1 / (1 + (1500 / np.maximum(f, 1.0)) ** 2)))
        return shelf * highpass_gain(f, 60.0, 2)
    return fft_eq(x, sr, g)


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness (BS.1770-style: 400 ms blocks, -70 LUFS absolute, -10 LU relative)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.shape[-1] < sr // 2:
        x = np.pad(x, ((0, 0), (0, sr // 2 - x.shape[-1])))
    kw = np.stack([k_weight(c, sr) for c in x])
    block, hop = int(0.4 * sr), int(0.1 * sr)
    n = 1 + (kw.shape[1] - block) // hop
    if n <= 0:
        return -70.0
    power = np.array([np.sum(np.mean(kw[:, i * hop:i * hop + block] ** 2, axis=1)) for i in range(n)])
    lk = -0.691 + 10 * np.log10(power + EPS)
    gated = power[lk > -70]
    if gated.size == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(gated.mean() + EPS) - 10
    final = power[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10((final.mean() if final.size else gated.mean()) + EPS))


def db(x: float) -> float:
    return float(20 * np.log10(max(abs(x), EPS)))


def midi_to_hz(m) -> np.ndarray:
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def hz_to_midi(f) -> np.ndarray:
    return 69 + 12 * np.log2(np.maximum(np.asarray(f, dtype=np.float64), 1e-6) / 440.0)


def estimate_f0(x: np.ndarray, sr: int, fmin: float = 25.0, fmax: float = 2500.0) -> float | None:
    """Median-frame autocorrelation f0 (YIN-like cumulative mean normalised difference) of a sample."""
    x = as_mono(x)
    n = len(x)
    if n < sr // 10:
        return None
    seg = x[int(n * 0.15):int(n * 0.15) + min(n // 2, sr)]  # skip the attack
    win = int(sr / fmin) * 2
    ests = []
    for start in range(0, max(1, len(seg) - win), max(1, win // 2)):
        f = seg[start:start + win]
        if len(f) < win:
            break
        tau_max = int(sr / fmin)
        tau_min = max(2, int(sr / fmax))
        w = win - tau_max
        d = np.array([np.sum((f[:w] - f[t:t + w]) ** 2) for t in range(tau_max)])
        cmnd = d[1:] * np.arange(1, tau_max) / (np.cumsum(d[1:]) + EPS)
        cmnd = np.concatenate([[1.0], cmnd])
        cand = np.where(cmnd[tau_min:] < 0.15)[0]
        tau = (cand[0] + tau_min) if cand.size else int(np.argmin(cmnd[tau_min:]) + tau_min)
        while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
        if 1 <= tau < tau_max - 1:  # parabolic interpolation
            a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
            denom = a - 2 * b + c
            tau = tau + (0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0)
        ests.append(sr / tau)
    return float(np.median(ests)) if ests else None
