"""Small numpy DSP kit for the procedural SFX pack: oscillators, FFT filters, envelopes, loudness.

numpy only (no scipy), so the pack builds anywhere the backend runs. Everything is
deterministic for a given random generator; the pack seeds one per sound id.
"""
import math

import numpy as np

SR = 48000


def n_of(seconds):
    return max(1, int(round(seconds * SR)))


def times(n):
    return np.arange(n) / SR


def midi(note):
    return 440.0 * 2 ** ((note - 69) / 12)


# ---------------------------------------------------------------- envelopes

def points(n, pts):
    """Piecewise-linear envelope through (seconds, value) points, held at the ends."""
    xs = np.array([p[0] for p in pts], dtype=float) * SR
    ys = np.array([p[1] for p in pts], dtype=float)
    return np.interp(np.arange(n), xs, ys)


def decay(n, tau, attack=0.002):
    """Exponential decay with a short linear attack, so nothing starts with a click."""
    t = times(n)
    env = np.exp(-t / max(tau, 1e-4))
    a = max(1, int(attack * SR))
    env[:a] *= np.linspace(0, 1, a, endpoint=False)
    return env


def adsr(n, a, d, s, r):
    total = n / SR
    release_start = max(a + d, total - r)
    return points(n, [(0, 0), (a, 1), (a + d, s), (release_start, s), (total, 0)])


def bump(n, center, width):
    """A raised-cosine bump (seconds), zero outside center±width."""
    t = times(n)
    x = np.clip((t - center) / width, -1, 1)
    return 0.5 * (1 + np.cos(np.pi * x)) * (np.abs(t - center) < width)


def smooth_noise(n, rng, rate_hz, depth=1.0):
    """Slowly varying noise in [-depth, depth]: random control points at rate_hz, cosine-interpolated."""
    count = max(2, int(n / SR * rate_hz) + 2)
    ctrl = rng.uniform(-1, 1, count)
    x = np.arange(n) / SR * rate_hz
    i = np.minimum(x.astype(int), count - 2)
    frac = x - i
    w = 0.5 - 0.5 * np.cos(np.pi * frac)
    return depth * (ctrl[i] * (1 - w) + ctrl[i + 1] * w)


# ---------------------------------------------------------------- oscillators

def const(value, n):
    return np.full(n, float(value))


def glide(f0, f1, n, curve='exp', power=1.0):
    x = np.linspace(0, 1, n) ** power
    if curve == 'exp':
        return f0 * (f1 / f0) ** x
    return f0 + (f1 - f0) * x


def phase(freq):
    return 2 * np.pi * np.cumsum(freq) / SR


def sine(freq, start_phase=0.0):
    return np.sin(phase(freq) + start_phase)


def harmonic(freq, amp_fn, kmax=48, ceiling=None):
    """Band-limited additive oscillator. amp_fn(k) gives harmonic k's amplitude; harmonics
    above `ceiling` (default 0.45*SR) fade out per sample, so glides never alias."""
    ceiling = ceiling or 0.45 * SR
    ph = phase(freq)
    top = int(min(kmax, max(1, ceiling / max(float(np.min(freq)), 1.0))))
    out = np.zeros_like(ph)
    for k in range(1, top + 1):
        a = amp_fn(k)
        if a == 0:
            continue
        hk = k * freq
        mask = np.clip((ceiling - hk) / (0.05 * ceiling), 0, 1)
        out += a * mask * np.sin(k * ph)
    return out


def saw(freq, kmax=48, ceiling=None):
    return harmonic(freq, lambda k: 1.0 / k, kmax, ceiling) * 0.6


def square(freq, kmax=48, ceiling=None):
    return harmonic(freq, lambda k: (1.0 / k) if k % 2 else 0.0, kmax, ceiling) * 0.8


def triangle(freq, kmax=24):
    return harmonic(freq, lambda k: ((-1) ** ((k - 1) // 2)) / (k * k) if k % 2 else 0.0, kmax) * 0.9


def noise(n, rng):
    return rng.standard_normal(n)


def partials(n, base, ratios, amps, decays, rng=None, detune=0.0):
    """Inharmonic struck partials (bells, metal, glass)."""
    t = times(n)
    out = np.zeros(n)
    for r, a, d in zip(ratios, amps, decays):
        f = base * r * (1 + (rng.uniform(-detune, detune) if rng is not None and detune else 0))
        if f >= 0.45 * SR:
            continue
        out += a * np.sin(2 * np.pi * f * t + (rng.uniform(0, 2 * np.pi) if rng is not None else 0)) * decay(n, d, 0.001)
    return out


# ---------------------------------------------------------------- filters (FFT domain)

def _freqs(m):
    return np.fft.rfftfreq(m, 1 / SR)


def fft_filter(x, response):
    """Zero-phase filter with |H(f)| = response(freqs). Pads to avoid circular wrap."""
    x = np.asarray(x, dtype=float)
    pad = min(len(x), SR // 2)
    m = len(x) + pad
    spec = np.fft.rfft(x, m, axis=0)
    h = response(_freqs(m))
    if spec.ndim == 2:
        h = h[:, None]
    return np.fft.irfft(spec * h, m, axis=0)[:len(x)]


def lowpass(x, fc, order=2):
    return fft_filter(x, lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order)))


def highpass(x, fc, order=2):
    return fft_filter(x, lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-6)) ** (2 * order)))


def bandpass(x, lo, hi, order=2):
    return lowpass(highpass(x, lo, order), hi, order)


def resonances(f, peaks):
    """Sum of Lorentzian peaks [(center, bandwidth, gain)] evaluated at frequencies f."""
    out = np.zeros_like(f, dtype=float)
    for c, bw, g in peaks:
        out = out + g / (1 + ((f - c) / bw) ** 2)
    return out


def formant(x, peaks, floor=0.02):
    return fft_filter(x, lambda f: resonances(f, peaks) + floor)


def stft_filter(x, mask_fn, frame=1024, hop=256):
    """Time-varying filter: mask_fn(freqs[1,bins], t_seconds[frames,1]) -> gains[frames,bins]."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    padded = np.concatenate([np.zeros(frame), x, np.zeros(frame + hop)])
    count = (len(padded) - frame) // hop + 1
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    window = np.hanning(frame)
    frames = padded[idx] * window
    spec = np.fft.rfft(frames, axis=1)
    centers = (hop * np.arange(count) + frame / 2 - frame) / SR
    gains = mask_fn(_freqs(frame)[None, :], centers[:, None])
    out_frames = np.fft.irfft(spec * gains, frame, axis=1) * window
    out = np.zeros(len(padded))
    norm = np.zeros(len(padded))
    for i in range(count):
        s = i * hop
        out[s:s + frame] += out_frames[i]
        norm[s:s + frame] += window ** 2
    out = out / np.maximum(norm, 1e-3)
    return out[frame:frame + n]


def gauss_band(f, center, width_oct):
    """Gaussian bump on a log-frequency axis (width in octaves)."""
    lf = np.log2(np.maximum(f, 1.0))
    return np.exp(-0.5 * ((lf - np.log2(center)) / width_oct) ** 2)


# ---------------------------------------------------------------- effects

def softclip(x, drive):
    return np.tanh(drive * x) / math.tanh(drive)


def tremolo(n, rate, depth):
    rate = np.asarray(rate, dtype=float) if np.ndim(rate) else const(rate, n)
    return 1 - depth * (0.5 + 0.5 * np.sin(phase(rate)))


def reverb(x, rng, seconds=1.2, wet=0.25, damp=6000.0, predelay=0.01, stereo=True):
    """Convolution with a synthesized impulse response (decaying filtered noise)."""
    x = np.asarray(x, dtype=float)
    mono = x if x.ndim == 1 else x.mean(axis=1)
    m = n_of(seconds)
    t = times(m)
    tau = seconds / 6.9
    outs = []
    for _ in range(2 if stereo else 1):
        ir = noise(m, rng) * np.exp(-t / tau)
        ir = lowpass(ir, damp, 1)
        ir[:n_of(predelay)] = 0
        ir /= max(np.sqrt(np.sum(ir ** 2)), 1e-9)
        size = len(mono) + m
        tail = np.fft.irfft(np.fft.rfft(mono, size) * np.fft.rfft(ir, size), size)
        outs.append(tail)
    wet_sig = np.stack(outs, axis=1) if stereo else outs[0]
    dry = to_stereo(x) if stereo else (x if x.ndim == 1 else x.mean(axis=1))
    dry = np.concatenate([dry, np.zeros((m,) + dry.shape[1:])])
    peak_dry = max(np.max(np.abs(dry)), 1e-9)
    wet_sig = wet_sig / max(np.max(np.abs(wet_sig)), 1e-9) * peak_dry
    return (1 - wet) * dry + wet * wet_sig


def to_stereo(x, pan=0.0):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return x
    angle = (pan + 1) * np.pi / 4
    return np.stack([x * math.cos(angle) * math.sqrt(2), x * math.sin(angle) * math.sqrt(2)], axis=1)


def pan_curve(x, pans):
    """Stereo from mono with a per-sample pan array in [-1, 1]."""
    angle = (np.asarray(pans) + 1) * np.pi / 4
    return np.stack([x * np.cos(angle), x * np.sin(angle)], axis=1) * math.sqrt(2)


def silence(seconds, channels=1):
    n = n_of(seconds)
    return np.zeros(n) if channels == 1 else np.zeros((n, channels))


def place(dst, src, at, gain=1.0):
    """Mix src into dst starting at `at` seconds, growing dst as needed. Handles mono/stereo."""
    src = np.asarray(src, dtype=float)
    if dst.ndim != src.ndim:
        dst, src = to_stereo(dst), to_stereo(src)
    start = max(0, int(round(at * SR)))
    end = start + len(src)
    if end > len(dst):
        dst = np.concatenate([dst, np.zeros((end - len(dst),) + dst.shape[1:])])
    dst[start:end] += gain * src
    return dst


def concat(*parts):
    parts = [np.asarray(p, dtype=float) for p in parts]
    if any(p.ndim == 2 for p in parts):
        parts = [to_stereo(p) for p in parts]
    return np.concatenate(parts)


# ---------------------------------------------------------------- loudness and finishing

_K1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
_K2 = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def _biquad_mag(f, b, a):
    z = np.exp(-1j * 2 * np.pi * f / SR)
    return np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z))


def loudness_lufs(x):
    """ITU-R BS.1770 integrated loudness (K-weighting in the FFT domain, 400 ms gated blocks).
    Clips shorter than one block are measured as a single block."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not x.size or not np.any(x):
        return -120.0
    weighted = fft_filter(x, lambda f: _biquad_mag(f, *_K1) * _biquad_mag(f, *_K2))
    block, hop = int(0.4 * SR), int(0.1 * SR)
    if len(weighted) <= block:
        powers = np.array([np.sum(np.mean(weighted ** 2, axis=0))])
    else:
        starts = range(0, len(weighted) - block + 1, hop)
        powers = np.array([np.sum(np.mean(weighted[s:s + block] ** 2, axis=0)) for s in starts])
    lk = -0.691 + 10 * np.log10(np.maximum(powers, 1e-12))
    gated = powers[lk > -70]
    if not len(gated):
        return -120.0
    relative = -0.691 + 10 * math.log10(float(np.mean(gated))) - 10
    gated = powers[(lk > -70) & (lk > relative)]
    return round(-0.691 + 10 * math.log10(float(np.mean(gated))), 2)


def true_peak(x):
    """4x-oversampled peak estimate (linear)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    m = len(x)
    spec = np.fft.rfft(x, axis=0)
    up = np.fft.irfft(spec, 4 * m, axis=0) * 4
    return float(max(np.max(np.abs(up)), np.max(np.abs(x))))


def db(value):
    return 20 * math.log10(max(value, 1e-9))


def finish(x, target_lufs=-16.0, ceiling_db=-1.0, tail_db=-60.0):
    """DC/sub-rumble removal, silence trim, fades, loudness target, true-peak ceiling.

    Returns float32 samples shaped [frames, channels]."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = highpass(x, 22, 2)
    level = np.max(np.abs(x), axis=1)
    peak = float(np.max(level)) if len(level) else 0.0
    if peak <= 0:
        raise ValueError('synthesized sound is silent')
    active = np.flatnonzero(level > peak * 10 ** (tail_db / 20))
    start = max(0, int(active[0]) - n_of(0.002))
    end = min(len(x), int(active[-1]) + n_of(0.02))
    x = x[start:end]
    fade_in, fade_out = min(len(x) // 4, n_of(0.002)), min(len(x) // 4, n_of(0.015))
    if fade_in:
        x[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    if fade_out:
        x[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]
    gain = 10 ** ((target_lufs - loudness_lufs(x)) / 20)
    ceiling = 10 ** (ceiling_db / 20)
    tp = true_peak(x) * gain
    if tp > ceiling:
        gain *= ceiling / tp
    return (x * gain).astype(np.float32)


def smooth(x, seconds):
    """Moving Hann-window average with edge padding (for control curves such as pitch)."""
    k = max(3, n_of(seconds) | 1)
    kernel = np.hanning(k)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(x, dtype=float), k // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')
