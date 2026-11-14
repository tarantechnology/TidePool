"""Chat-driven layer effects: render bounded effect chains, ducking and loudness as candidates.

Rendering is CPU-only: ffmpeg filters for the standard effects and numpy for what ffmpeg lacks
(convolution reverb with synthesized impulse responses, echo taps, waveshapers, bitcrush, noise,
ring modulation and the output limiter). Every chain renders from the layer's dry audio, so a chip
can be removed or adjusted later by re-rendering without it. Length, sample rate and channels are
preserved, except for the explicit whole-song speed op.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import fx_schema as S
from .candidates import revisions, safe_path
from .core import probe, project_dir, uid
from .media import export
from .visual import file_sha256

TRUE_PEAK_CEILING_DB = -1.0
LOUDNESS_RISE_WARN_DB = 6.0
TIMEOUT = 600


# ---------------------------------------------------------------- audio I/O

def decode(path, sample_rate=None, channels=None):
    """Float32 (samples, channels) at the file's own rate and channel count unless asked."""
    info = probe(path)
    sr = sample_rate or info['sample_rate']; ch = channels or info['channels']
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-f', 'f32le', '-acodec', 'pcm_f32le',
                          '-ar', str(sr), '-ac', str(ch), 'pipe:1'], capture_output=True, timeout=TIMEOUT)
    if out.returncode: raise RuntimeError('Could not decode audio: ' + out.stderr.decode(errors='replace')[-500:])
    return np.frombuffer(out.stdout, np.float32).reshape(-1, ch).copy(), sr


def write_wav(path, x, sr):
    """IEEE-float WAV (what the rest of the pipeline writes with pcm_f32le)."""
    x = np.ascontiguousarray(x, dtype='<f4'); n, ch = x.shape
    data = x.tobytes()
    header = (b'RIFF' + struct.pack('<I', 4 + 26 + 12 + 8 + len(data)) + b'WAVE'
              + b'fmt ' + struct.pack('<IHHIIHHH', 18, 3, ch, sr, sr * ch * 4, ch * 4, 32, 0)
              + b'fact' + struct.pack('<II', 4, n)
              + b'data' + struct.pack('<I', len(data)))
    with open(path, 'wb') as f:
        f.write(header); f.write(data)


@lru_cache(maxsize=1)
def ffmpeg_filters():
    try:
        text = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(m.group(1) for m in re.finditer(r'^\s*\S+\s+(\w+)\s+\S+->\S+', text, re.M))


def ffmpeg_pass(x, sr, filters):
    ch = x.shape[1]
    out = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', str(ch), '-i', 'pipe:0', '-af', ','.join(filters),
                          '-f', 'f32le', '-ar', str(sr), '-ac', str(ch), 'pipe:1'], input=x.tobytes(), capture_output=True, timeout=TIMEOUT)
    if out.returncode: raise RuntimeError('Effect rendering failed: ' + out.stderr.decode(errors='replace')[-500:])
    y = np.frombuffer(out.stdout, np.float32).reshape(-1, ch)
    return fit(y, len(x))


def fit(y, n):
    """Exactly n samples: effects with tails (chorus, echo) are trimmed, short ones padded."""
    if len(y) >= n: return np.array(y[:n], np.float32)
    return np.concatenate([y, np.zeros((n - len(y), y.shape[1]), np.float32)])


def loudness(x, sr):
    """Integrated loudness (LUFS, EBU R128) via ffmpeg's ebur128; -70 for silence."""
    out = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-f', 'f32le', '-ar', str(sr), '-ac', str(x.shape[1]), '-i', 'pipe:0',
                          '-af', 'ebur128', '-f', 'null', '-'], input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True, timeout=TIMEOUT)
    found = re.findall(r'I:\s*(-?\d+(?:\.\d+)?|-?inf)\s*LUFS', out.stderr.decode(errors='replace'))
    if out.returncode or not found: raise RuntimeError('Could not measure loudness.')
    value = float(found[-1])
    return max(-70.0, value) if math.isfinite(value) else -70.0


# ---------------------------------------------------------------- peaks and limiting

def _phase_kernels(taps=16):
    j = np.arange(-taps // 2 + 1, taps // 2 + 1)
    kernels = []
    for p in (.25, .5, .75):
        k = np.sinc(j - p) * np.kaiser(taps, 6.0)
        kernels.append((k / k.sum()).astype(np.float32))
    return kernels, taps // 2 - 1


_KERNELS = _phase_kernels()


def peak_envelope(x):
    """Per-sample peak including inter-sample peaks (4x oversampled estimate), max over channels."""
    kernels, lead = _KERNELS
    env = np.abs(x).max(axis=1)
    for c in range(x.shape[1]):
        col = x[:, c]
        for k in kernels:
            y = np.convolve(col, k[::-1], mode='full')[lead:lead + len(col)]
            np.maximum(env, np.abs(y), out=env)
    return env


def true_peak_db(x):
    if not len(x): return -120.0
    peak = float(peak_envelope(x).max())
    return 20 * math.log10(peak) if peak > 1e-9 else -120.0


def limit(x, sr, ceiling_db=TRUE_PEAK_CEILING_DB):
    """Look-ahead peak limiter with linked channels. Gains are computed per 32-sample block, take
    the minimum of neighbouring blocks, then attack/release ramps that only lower them, so the
    interpolated gain never exceeds what any sample needs. A no-op below the ceiling."""
    ceiling = 10 ** ((ceiling_db - .05) / 20)
    env = peak_envelope(x)
    if env.max() <= ceiling: return x
    need = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))
    block = 32; n = len(x); count = -(-n // block)
    padded = np.ones(count * block, np.float32); padded[:n] = need
    g = padded.reshape(count, block).min(axis=1)
    g = np.minimum(g, np.minimum(np.r_[g[1:], 1.0], np.r_[1.0, g[:-1]]))
    k = np.arange(count, dtype=np.float64)
    release = block / (.08 * sr); attack = block / (.003 * sr)
    g = np.minimum(g, k * release + np.minimum.accumulate(g - k * release))  # Rise at most `release` per block.
    rk = k[::-1]; gr = g[::-1]
    g = np.minimum(g, (rk * attack + np.minimum.accumulate(gr - rk * attack))[::-1])  # Fall early (look-ahead).
    centers = k * block + (block - 1) / 2
    gain = np.interp(np.arange(n), centers, g).astype(np.float32)
    y = x * gain[:, None]
    over = peak_envelope(y).max()
    if over > ceiling: y *= np.float32(ceiling / over)
    return y


def guard(x, sr):
    if not np.isfinite(x).all(): raise ValueError('Effect rendering produced invalid samples; the layer was not changed.')
    return limit(x, sr)


# ---------------------------------------------------------------- numpy DSP

def fft_convolve(x, h):
    """First len(x) samples of x * h by overlap-add (float32)."""
    n, m = len(x), len(h)
    size = 1 << max(16, int(math.ceil(math.log2(2 * m))))
    step = size - m + 1
    H = np.fft.rfft(h.astype(np.float32), size)
    out = np.zeros(n + size, np.float32)
    for start in range(0, n, step):
        seg = x[start:start + step]
        out[start:start + size] += np.fft.irfft(np.fft.rfft(seg, size) * H, size).astype(np.float32)
    return out[:n]


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()[:8], 'little')


def impulse_response(sr, size, decay, predelay_ms, damping, seed, channels):
    """Procedural IR: band-split exponentially decaying noise (highs decay faster with damping),
    a diffuse build-up and sparse early reflections. Unit energy per channel; no recorded IRs."""
    length = int(min(10.0, decay * 1.2 + .05) * sr)
    t = np.arange(length, dtype=np.float32) / sr
    rng = np.random.default_rng(seed)
    crossover = 4500 - 3000 * damping
    freqs = np.fft.rfftfreq(length, 1 / sr)
    lowpass = 1 / (1 + (freqs / crossover) ** 4)
    env_low = np.exp(-6.91 * t / decay)
    env_high = np.exp(-6.91 * t / max(.05, decay * (1 - .75 * damping)))
    build = 1 - np.exp(-t / (.002 + .02 * size))
    pre = int(predelay_ms / 1000 * sr)
    out = []
    for c in range(channels):
        noise = rng.standard_normal(length).astype(np.float32)
        low = np.fft.irfft(np.fft.rfft(noise) * lowpass, length).astype(np.float32)
        tail = (low * env_low + (noise - low) * env_high) * build
        taps = 6 + int(size * 10)
        times = np.sort(rng.uniform(.001, .005 + .075 * size, taps))
        gains = rng.uniform(.2, .6, taps) * np.where(rng.random(taps) < .5, -1, 1) * np.linspace(1, .4, taps)
        er = np.zeros(length, np.float32)
        np.add.at(er, np.minimum(length - 1, (times * sr).astype(int)), gains.astype(np.float32))
        ir = tail / max(1e-9, float(np.sqrt((tail ** 2).sum()))) * .8 + er / max(1e-9, float(np.sqrt((er ** 2).sum()))) * .6
        ir = np.concatenate([np.zeros(pre, np.float32), ir.astype(np.float32)])
        out.append(ir / max(1e-9, float(np.sqrt((ir ** 2).sum()))))
    return out


def reverb(x, sr, p, seed):
    irs = impulse_response(sr, p['size'], p['decay'], p['predelay'], p['damping'], seed, min(2, x.shape[1]))
    wet = np.stack([fft_convolve(x[:, c], irs[min(c, len(irs) - 1)]) for c in range(x.shape[1])], axis=1)
    angle = p['mix'] * math.pi / 2
    return (x * math.cos(angle) + wet * math.sin(angle)).astype(np.float32)


def delay(x, sr, p):
    n = len(x); step = max(1, int(round(p['time'] / 1000 * sr)))
    wet = np.zeros_like(x); level = 1.0; k = 1
    stereo = x.shape[1] == 2 and p['pingpong'] >= .5
    mono = x.mean(axis=1) if stereo else None
    while k * step < n and level >= 1e-3 and k <= 200:
        shift = k * step
        if stereo: wet[shift:, (k - 1) % 2] += level * mono[:n - shift]
        else: wet[shift:] += level * x[:n - shift]
        level *= p['feedback']; k += 1
        if p['feedback'] <= 0: break
    return (x + p['mix'] * wet).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def _boxcar(x, width):
    if width <= 1: return x
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1]), np.float64), x.astype(np.float64)]), axis=0)
    y = (c[width:] - c[:-width]) / width
    return fit(np.vstack([np.repeat(y[:1], width // 2, axis=0), y]).astype(np.float32), len(x))


def _matched(dry, wet):
    a, b = _rms(dry), _rms(wet)
    return wet * np.float32(a / b) if b > 1e-9 and a > 1e-9 else wet


def saturation(x, p):
    k = 1 + p['drive'] * 9
    wet = _matched(x, np.tanh(k * x) / math.tanh(k))
    return (x * (1 - p['mix']) + wet * p['mix']).astype(np.float32)


def distortion(x, sr, p):
    k = 1 + p['drive'] * 40
    shaped = np.arctan(k * x) / math.atan(k)
    cutoff = 1500 + p['tone'] * 9000
    wet = _matched(x, _boxcar(shaped.astype(np.float32), max(1, int(round(sr / cutoff / 2)))))
    return (x * (1 - p['mix']) + wet * p['mix']).astype(np.float32)


def bitcrush(x, p):
    levels = 2 ** (int(p['bits']) - 1)
    wet = np.round(x * levels) / levels
    ds = int(p['downsample'])
    if ds > 1: wet = wet[(np.arange(len(x)) // ds) * ds]
    return (x * (1 - p['mix']) + wet * p['mix']).astype(np.float32)


def noise(x, sr, p, seed):
    """Vinyl-style hiss plus sparse crackle, scaled to the layer's level."""
    rng = np.random.default_rng(seed); n = len(x)
    ref = max(_rms(x), .01)
    hiss = _boxcar(rng.standard_normal((n, 1)).astype(np.float32), 3)[:, 0] * (.08 * p['amount'] * ref)
    clicks = np.zeros(n, np.float32)
    count = int(p['amount'] * 8 * n / sr)
    if count:
        at = rng.integers(0, n, count)
        np.add.at(clicks, at, (rng.uniform(.5, 3, count) * p['amount'] * ref * np.where(rng.random(count) < .5, -1, 1)).astype(np.float32))
        clicks = np.convolve(clicks, np.exp(-np.arange(24) / 4).astype(np.float32))[:n]
    return (x + (hiss + clicks)[:, None]).astype(np.float32)


def ring_mod(x, sr, p):
    carrier = np.sin(2 * np.pi * p['frequency'] * np.arange(len(x)) / sr).astype(np.float32) * np.float32(math.sqrt(2))
    return (x * (1 - p['mix']) + x * carrier[:, None] * p['mix']).astype(np.float32)


def fade(x, sr, p):
    gain = np.ones(len(x), np.float32); n = len(x)
    a = min(n, int(p['fade_in'] * sr)); b = min(n, int(p['fade_out'] * sr))
    if a: gain[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b: gain[n - b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return x * gain[:, None]


# ---------------------------------------------------------------- ffmpeg mapping

def ffmpeg_filter(effect, sr, channels, notes):
    """The ffmpeg filter for an ffmpeg-rendered effect type, built only from clamped numbers.
    Returns None for numpy-rendered types, '' for a no-op."""
    t = effect['type']; p = effect['params']; nyq = sr / 2 * .95
    if t == 'eq': return f"equalizer=f={min(p['frequency'], nyq):.2f}:t=q:w={p['q']:.3f}:g={p['gain']:.2f}" if p['gain'] else ''
    if t == 'low_shelf': return f"bass=g={p['gain']:.2f}:f={p['frequency']:.2f}:w=0.6" if p['gain'] else ''
    if t == 'high_shelf': return f"treble=g={p['gain']:.2f}:f={min(p['frequency'], nyq):.2f}:w=0.6" if p['gain'] else ''
    if t == 'highpass': return f"highpass=f={min(p['frequency'], nyq):.2f}:poles=2"
    if t == 'lowpass': return f"lowpass=f={min(p['frequency'], nyq):.2f}:poles=2" if p['frequency'] < nyq else ''
    if t == 'bandpass':
        low, high = min(p['low'], nyq / 2), min(p['high'], nyq)
        return ','.join([f'highpass=f={low:.2f}'] * 2 + ([f'lowpass=f={high:.2f}'] * 2 if high < nyq else []))
    if t == 'compressor':
        return (f"acompressor=threshold={max(.000977, 10 ** (p['threshold'] / 20)):.6f}:ratio={p['ratio']:.3f}:attack={max(.01, p['attack']):.3f}"
                f":release={p['release']:.3f}:makeup={min(64, 10 ** (p['makeup'] / 20)):.4f}")
    if t == 'chorus':
        if not p['mix']: return ''
        r, d, m = p['rate'], p['depth'], p['mix']
        return f"chorus=0.8:0.9:25|40:{.5 * m:.3f}|{.4 * m:.3f}:{r:.3f}|{r * 1.31:.3f}:{1 + 3 * d:.3f}|{1 + 2.5 * d:.3f}"
    if t == 'flanger': return f"flanger=delay=2:depth={p['depth'] * 8:.3f}:regen={p['feedback'] * 90:.2f}:width=71:speed={max(.1, p['rate']):.3f}"
    if t == 'phaser': return f"aphaser=in_gain=0.6:out_gain=0.8:delay={1 + p['depth'] * 4:.3f}:decay={min(.95, p['feedback'] + .05):.3f}:speed={p['rate']:.3f}:type=t"
    if t == 'tremolo': return f"tremolo=f={p['rate']:.3f}:d={p['depth']:.3f}" if p['depth'] else ''
    if t == 'wobble': return f"vibrato=f={p['rate']:.3f}:d={max(.001, p['depth'] * .5):.3f}" if p['depth'] else ''
    if t == 'gain': return f"volume={p['gain']:.2f}dB" if p['gain'] else ''
    if t in ('autopan', 'width'):
        if channels != 2:
            notes.append(f"{S.EFFECT_LABELS[t]} needs a stereo layer; skipped on a mono layer."); return ''
        if t == 'autopan': return f"apulsator=mode=sine:hz={p['rate']:.3f}:amount={p['depth']:.3f}" if p['depth'] else ''
        return f"extrastereo=m={p['amount']:.3f}:c=false" if p['amount'] != 1 else ''
    if t == 'pitch':
        if not p['semitones']: return ''
        ratio = 2 ** (p['semitones'] / 12)
        if 'rubberband' in ffmpeg_filters():
            return f"rubberband=pitch={ratio:.6f}:formant={'preserved' if p['formant'] else 'shifted'}"
        notes.append('Pitch uses a resample-and-stretch fallback (rubberband is unavailable); formants shift with the pitch.')
        return f'asetrate={sr * ratio:.3f},aresample={sr},atempo={1 / ratio:.6f}'
    return None


def apply_chain(x, sr, chain, seed, notes):
    pending = []
    def flush(y):
        if pending:
            y = ffmpeg_pass(y, sr, pending); pending.clear()
        return y
    for i, effect in enumerate(chain):
        f = ffmpeg_filter(effect, sr, x.shape[1], notes)
        if f is not None:
            if f: pending.append(f)
            continue
        x = flush(x); t = effect['type']; p = effect['params']
        if t == 'reverb': x = reverb(x, sr, p, _seed(seed, i))
        elif t == 'delay': x = delay(x, sr, p)
        elif t == 'limiter': x = limit(x, sr, p['ceiling'])
        elif t == 'saturation': x = saturation(x, p)
        elif t == 'distortion': x = distortion(x, sr, p)
        elif t == 'bitcrush': x = bitcrush(x, p)
        elif t == 'noise': x = noise(x, sr, p, _seed(seed, i))
        elif t == 'ring_mod': x = ring_mod(x, sr, p)
        elif t == 'fade': x = fade(x, sr, p)
        else: raise ValueError(f'Unsupported effect {t}.')
    return flush(x)


def blend(dry, wet, sr, interval):
    """Wet only inside [start, end], with short linear crossfades just inside the edges."""
    n = len(dry); a = int(round(interval['start'] * sr)); b = min(n, int(round(interval['end'] * sr)))
    if a >= n or b <= a: return dry
    xf = max(1, min(int(.03 * sr), (b - a) // 4))
    mask = np.zeros(n, np.float32); mask[a:b] = 1
    mask[a:a + xf] = np.linspace(0, 1, xf, dtype=np.float32)
    mask[b - xf:b] = np.linspace(1, 0, xf, dtype=np.float32)
    return dry + (wet - dry) * mask[:, None]


def change_speed(x, sr, rate):
    """Turntable-style speed change (tempo and pitch together); the only length-changing op."""
    y = ffmpeg_pass_free(x, sr, [f'asetrate={sr * rate:.4f}', f'aresample={sr}'])
    return fit(y, int(round(len(x) / rate)))


def ffmpeg_pass_free(x, sr, filters):
    ch = x.shape[1]
    out = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', str(ch), '-i', 'pipe:0', '-af', ','.join(filters),
                          '-f', 'f32le', '-ar', str(sr), '-ac', str(ch), 'pipe:1'], input=x.tobytes(), capture_output=True, timeout=TIMEOUT)
    if out.returncode: raise RuntimeError('Speed change failed: ' + out.stderr.decode(errors='replace')[-500:])
    return np.frombuffer(out.stdout, np.float32).reshape(-1, ch)


def net_speed(items):
    rate = 1.0
    for it in items: rate *= it.get('speed') or 1.0
    return rate


def render(x, sr, items, notes):
    """Dry audio through every chip in order, then the output guard."""
    for it in items:
        if it.get('speed'): x = change_speed(x, sr, it['speed'])
        if not it['chain']: continue
        before = loudness(x, sr) if it.get('loudness_cap_db') is not None else None
        y = apply_chain(x, sr, it['chain'], it['id'], notes)
        if it.get('match_level'):
            # Band-limited "character" presets keep the layer's level instead of getting much quieter.
            a, b = _rms(x), _rms(y)
            if a > 1e-6 and b > 1e-9: y = y * np.float32(min(4.0, a / b))
        if it.get('interval'): y = blend(x, y, sr, it['interval'])
        if before is not None and before > -70:
            rise = loudness(y, sr) - before
            if rise > it['loudness_cap_db']:
                y = y * np.float32(10 ** (-(rise - it['loudness_cap_db']) / 20))
                notes.append(f"{it['label']} was turned down {rise - it['loudness_cap_db']:.1f} dB so it cannot get dangerously loud.")
        x = y
    return guard(x, sr)


# ---------------------------------------------------------------- per-layer records

def record(row):
    """The layer's applied chips, valid only while the layer still holds the audio they rendered."""
    try: rec = json.loads(row.get('fx') or 'null')
    except (TypeError, ValueError): rec = None
    if isinstance(rec, dict) and rec.get('path') == row.get('path') and rec.get('dry_path') and isinstance(rec.get('items'), list): return rec
    return None


def public(row):
    rec = record(row)
    if not rec or not rec['items']: return None
    keys = ('id', 'label', 'preset', 'scope', 'interval', 'speed', 'kind')
    return {'items': [{k: it.get(k) for k in keys} for it in rec['items']]}


def dry_path(row):
    rec = record(row)
    return rec['dry_path'] if rec else None


def _scale_times(row, ratio):
    if abs(ratio - 1) < 1e-9: return
    row['start_offset'] = float(row.get('start_offset') or 0) * ratio
    points = json.loads(row.get('automation') or '[]')
    row['automation'] = json.dumps([{**p, 'time': p['time'] * ratio} for p in points])
    ranges = json.loads(row.get('mute_ranges') or '[]')
    row['mute_ranges'] = json.dumps([[a * ratio, b * ratio] for a, b in ranges])


# ---------------------------------------------------------------- ducking

def _interp_curve(points, t, default=1.0):
    if not points: return np.full_like(t, default)
    xs = np.array([p[0] for p in points]); vs = np.array([p[1] for p in points])
    return np.interp(t, xs, vs)


def duck_keyframes(source_x, source_sr, source_offset, mute_ranges, volume, timeline, existing, params, budget=64):
    """Gain keyframes that lower a layer while the source (voice) is active. Returns (points, regions, threshold)."""
    mono = source_x.mean(axis=1) * volume
    hop = max(1, int(.02 * source_sr)); frames = len(mono) // hop
    if frames < 2: raise ValueError('The voice layer is too short to duck against.')
    level = 10 * np.log10(np.mean(np.square(mono[:frames * hop].reshape(frames, hop), dtype=np.float64), axis=1) + 1e-12)
    times = np.arange(frames) * hop / source_sr + source_offset
    for a, b in mute_ranges: level[(times >= a) & (times <= b)] = -120
    audible = level[level > -80]
    if not len(audible): raise ValueError('The voice layer is silent; nothing to duck against.')
    threshold = params['threshold_db'] if params['threshold_db'] is not None else max(-55.0, float(np.percentile(audible, 95)) - 18)
    active = level > threshold
    regions = []; start = None
    for i, on in enumerate(active):
        if on and start is None: start = times[i]
        if not on and start is not None: regions.append([start, times[i]]); start = None
    if start is not None: regions.append([start, times[-1] + hop / source_sr])
    attack, release = params['attack_ms'] / 1000, params['release_ms'] / 1000
    regions = [r for r in regions if r[1] - r[0] >= .06]
    def merge(gap):
        out = []
        for r in regions:
            if out and r[0] - out[-1][1] <= gap: out[-1][1] = max(out[-1][1], r[1])
            else: out.append(list(r))
        return out
    gap = attack + release + .3
    regions = merge(gap)
    regions = [r for r in regions if r[1] - r[0] >= .15]
    if not regions: raise ValueError('No speech or vocals were loud enough to duck under. Try a lower threshold.')
    existing = sorted((float(p['time']), float(p['value'])) for p in existing)
    if len(existing) > budget - 8: raise ValueError('This layer already has too many volume keyframes to add ducking.')
    while 4 * len(regions) + len(existing) > budget:
        gaps = [regions[i + 1][0] - regions[i][1] for i in range(len(regions) - 1)]
        i = int(np.argmin(gaps)); regions[i][1] = regions[i + 1][1]; del regions[i + 1]
    duck = 10 ** (-params['depth_db'] / 20)
    curve = []
    for a, b in regions:
        curve += [(max(0.0, a - attack), 1.0), (max(0.0, a), duck), (min(timeline, b), duck), (min(timeline, b + release), 1.0)]
    clean = []
    for t, v in curve:
        if clean and t <= clean[-1][0] + .005: clean[-1] = (clean[-1][0], min(clean[-1][1], v))
        else: clean.append((t, v))
    grid = sorted({round(t, 4) for t, _ in clean} | {round(t, 4) for t, _ in existing})
    grid = [t for i, t in enumerate(grid) if i == 0 or t - grid[i - 1] > .005]
    tt = np.array(grid)
    values = _interp_curve(clean, tt) * (_interp_curve(existing, tt) if existing else 1.0)
    points = [{'time': float(t), 'value': float(min(2.0, max(0.0, round(v, 4))))} for t, v in zip(tt, values)]
    return points, regions, threshold


# ---------------------------------------------------------------- candidate build

def _check_snapshot(pid, snapshot):
    source = safe_path(pid, snapshot['source'])
    if file_sha256(source) != snapshot['source_sha256']: raise ValueError('Source changed; generate another preview.')
    for row in snapshot['rows']:
        if file_sha256(safe_path(pid, row['path'])) != snapshot['stem_hashes'][row['id']]: raise ValueError('A layer changed; generate another preview.')
    return source


def _timeline(rows):
    return max(float(r.get('start_offset') or 0) + float(r['duration']) for r in rows)


def build(project, payload, candidate_id):
    pid = project['id']; snapshot = payload['_candidate']; spec = payload['fx']
    source = _check_snapshot(pid, snapshot)
    rows = [dict(r) for r in snapshot['rows']]
    if not rows:
        rows = [{'id': uid(), 'project_id': pid, 'name': 'Original audio', 'path': str(source), 'duration': probe(source)['duration'],
                 'start_offset': 0, 'volume': 1, 'muted': 0, 'solo': 0, 'automation': '[]', 'mute_ranges': '[]', 'revision': 0}]
    directory = project_dir(pid) / 'stems'; directory.mkdir(parents=True, exist_ok=True)
    created = []
    def new_file():
        path = directory / (uid() + '.wav'); created.append(path); return path
    notes: list[str] = list(spec.get('notes') or [])
    try:
        original = new_file()
        if snapshot['rows']: export(snapshot['rows'], source, original, 'wav', False)
        else: shutil.copyfile(source, original)
        by_id = {r['id']: r for r in rows}
        op = spec['op']
        provenance = {'kind': 'fx', 'op': op, 'stem_revisions': revisions(snapshot['rows']), 'source_sha256': snapshot['source_sha256']}
        if op == 'duck':
            result = _duck(rows, by_id, spec, provenance)
        else:
            result = _render_rows(rows, by_id, spec, provenance, original, new_file, notes)
        mix = new_file(); export(rows, source, mix, 'wav', False)
        mix_x, mix_sr = decode(mix)
        mix_peak = true_peak_db(mix_x)
        provenance['mix_true_peak_db'] = round(mix_peak, 2)
        if op == 'loudness':
            provenance['loudness']['after_lufs'] = round(loudness(mix_x, mix_sr), 1) if spec['targets'] == ['mix'] else None
        if mix_peak > TRUE_PEAK_CEILING_DB + .05 and op != 'duck':
            notes.append(f'The combined mix peaks at {mix_peak:.1f} dBTP; each layer is limited, but lower a layer or normalize loudness before export.')
        target = result.get('target') or mix
        return {'target': {'path': str(target)}, 'mix': {'path': str(mix)}, 'original': {'path': str(original)}, 'snapshot': snapshot,
                'proposed': rows,
                'inference': {'model': {'backend': 'fx-dsp', 'is_mock': False},
                              'warnings': list(dict.fromkeys(notes + ['Effects render into the layer when applied; Undo restores the previous audio.']))},
                'provenance': provenance}
    except Exception:
        for path in created: path.unlink(missing_ok=True)
        raise


def _duck(rows, by_id, spec, provenance):
    source = by_id.get(spec['duck']['source'])
    if not source: raise ValueError('The voice layer no longer exists.')
    if source['muted']: raise ValueError('The voice layer is muted; unmute it to duck against it.')
    targets = [by_id[t] for t in spec['targets'] if t in by_id and t != source['id']]
    if not targets: raise ValueError('Choose at least one music layer to duck.')
    x, sr = decode(source['path'], channels=1)
    params = spec['duck']['params']; timeline = _timeline(rows)
    summary = []
    for row in targets:
        points, regions, threshold = duck_keyframes(x, sr, float(source.get('start_offset') or 0), json.loads(source['mute_ranges'] or '[]'),
                                                    float(source['volume']), timeline, json.loads(row['automation'] or '[]'), params)
        row['automation'] = json.dumps(points)
        summary.append({'id': row['id'], 'name': row['name'], 'keyframes': len(points), 'regions': len(regions)})
    provenance.update(source={'id': source['id'], 'name': source['name']}, targets=summary, params={**params, 'threshold_db': round(threshold, 1)},
                      label=f"Ducked under {source['name']}", steps=[f"Lower {', '.join(r['name'] for r in targets)} by {params['depth_db']:g} dB while {source['name']} is active"])
    return {}


def _render_rows(rows, by_id, spec, provenance, original, new_file, notes):
    op = spec['op']
    if spec['targets'] == ['mix']: targets = list(rows)
    else: targets = [by_id[t] for t in spec['targets'] if t in by_id]
    if not targets: raise ValueError('Layer no longer exists.')
    records = {r['id']: record(r) or {'version': 1, 'dry_path': r['path'], 'items': [], 'path': None} for r in rows}
    plans = {}
    if op == 'chain':
        new = spec['item']
        if new.get('speed'): targets = list(rows)  # A speed change keeps every layer on one timeline.
        scope = 'mix' if spec['targets'] == ['mix'] or new.get('speed') else 'layer'
        for row in targets: plans[row['id']] = records[row['id']]['items'] + [{**new, 'scope': scope}]
        provenance.update(item={**new, 'scope': scope}, label=new['label'], steps=[e['type'] for e in new['chain']])
    elif op in ('adjust', 'remove'):
        wanted = spec.get('item_id'); changed = None
        if wanted:
            changed = next((it for r in rows for it in records[r['id']]['items'] if it['id'] == wanted), None)
        else:
            # The named preset if the user said which, else the most recent chip on the chosen layers.
            pool = [it for r in targets for it in records[r['id']]['items']]
            if spec.get('preset'): changed = next((it for it in reversed(pool) if it.get('preset') == spec['preset']), None)
            if changed is None and not spec.get('preset'): changed = next((it for it in reversed(pool) if it.get('kind') != 'loudness' or op == 'remove'), None)
        if not changed: raise ValueError('There is no applied effect to change on that layer yet.')
        target_id = changed['id']
        for row in rows:
            items = records[row['id']]['items']
            if not any(it['id'] == target_id for it in items): continue
            if op == 'remove': plans[row['id']] = [it for it in items if it['id'] != target_id]
            else: plans[row['id']] = [S.adjust_item(it, spec['direction']) if it['id'] == target_id else it for it in items]
        after = next((it for items in plans.values() for it in items if it['id'] == target_id), None)
        provenance.update(item=after or changed, removed=changed if op == 'remove' else None,
                          label=(f"Removed {changed['label']}" if op == 'remove' else after['label']))
    elif op == 'loudness':
        goal = spec['lufs']
        if spec['targets'] == ['mix']:
            x, sr = decode(original)
            measured = loudness(x, sr); peak = true_peak_db(x)
            if measured <= -69: raise ValueError('The mix is silent; there is nothing to normalize.')
            delta = goal - measured
            if len(rows) > 1 and delta > TRUE_PEAK_CEILING_DB - .1 - peak:
                capped = TRUE_PEAK_CEILING_DB - .1 - peak
                notes.append(f'Peaks limit the gain: reaches about {measured + capped:.1f} LUFS instead of {goal:g} LUFS without limiting the mix.')
                delta = capped
            measurements = {r['id']: (measured, peak) for r in rows}
            targets = list(rows); scope = 'mix'
        else:
            measurements = {}
            for row in targets:
                x, sr = decode(row['path']); measurements[row['id']] = (loudness(x, sr), true_peak_db(x))
            scope = 'layer'; delta = None
        loud_id = uuid_short()
        for row in targets:
            items = records[row['id']]['items']
            old = next((it for it in items if it.get('kind') == 'loudness'), None)
            old_gain = old['chain'][0]['params']['gain'] if old else 0.0
            d = delta if delta is not None else goal - measurements[row['id']][0]
            if delta is None and measurements[row['id']][0] <= -69: raise ValueError(f"{row['name']} is silent; there is nothing to normalize.")
            gain = min(12.0, max(-24.0, old_gain + d))
            new = {'id': loud_id, 'label': f'Loudness {goal:g} LUFS', 'preset': None, 'kind': 'loudness', 'scope': scope, 'interval': None, 'speed': None,
                   'chain': [{'type': 'gain', 'params': {'gain': round(gain, 2)}}]}
            plans[row['id']] = [it for it in items if it.get('kind') != 'loudness'] + [new]
        provenance.update(loudness={'target_lufs': goal, 'before_lufs': round(next(iter(measurements.values()))[0], 1) if scope == 'mix' else None,
                                    'layers': {by_id[k]['name']: round(v[0], 1) for k, v in measurements.items()} if scope == 'layer' else None},
                          label=f'Loudness {goal:g} LUFS', steps=['Measure integrated loudness (EBU R128)', 'Apply gain', 'Limit true peaks to -1 dBTP'])
    else:
        raise ValueError('Unsupported effect operation.')
    if not plans: raise ValueError('Nothing to change on the chosen layers.')

    def work(row):
        rec = records[row['id']]; items = plans[row['id']]; local: list[str] = []
        old_rate = net_speed(rec['items']); new_rate = net_speed(items)
        try: safe_path(row['project_id'], rec['dry_path'])
        except ValueError as exc: raise ValueError(f"The unprocessed audio of {row['name']} is no longer available; effects cannot be re-rendered.") from exc
        if not items:
            path = rec['dry_path']; fx = None
            dry_info = probe(path); duration = dry_info['duration']
            return row, path, fx, duration, old_rate, new_rate, local, None
        dry = rec['dry_path']
        dst = new_file()
        x, sr = decode(dry)
        expected = len(x) if abs(new_rate - 1) < 1e-9 else int(round(len(x) / new_rate))
        y = render(x, sr, items, local)
        if len(y) != expected or y.shape[1] != x.shape[1]: raise ValueError('Effects changed the audio timeline.')
        write_wav(dst, y, sr)
        rise = None
        try:
            current, csr = decode(row['path'])
            before = loudness(current, csr); after = loudness(y, sr)
            rise = after - before
            if before > -70 and rise > LOUDNESS_RISE_WARN_DB:
                local.append(f"{row['name']} gets {rise:.1f} dB louder; check your volume before playing it back.")
        except RuntimeError:
            pass
        return row, str(dst), {'version': 1, 'dry_path': dry, 'items': items, 'path': str(dst)}, len(y) / sr, old_rate, new_rate, local, rise

    with ThreadPoolExecutor(max_workers=min(4, len(plans))) as pool:
        results = list(pool.map(work, [r for r in rows if r['id'] in plans]))
    changed_rows = []
    for row, path, record_, duration, old_rate, new_rate, local, rise in results:
        row['path'] = path; row['fx'] = json.dumps(record_) if record_ else None; row['duration'] = duration
        _scale_times(row, old_rate / new_rate)
        notes.extend(local)
        changed_rows.append({'id': row['id'], 'name': row['name'], 'loudness_change_db': round(rise, 1) if rise is not None else None})
    provenance['targets'] = changed_rows
    if any(abs(r[5] - r[4]) > 1e-9 for r in results):
        rate = results[0][5]
        provenance['speed'] = rate
        notes.append(f'This changes the song speed to {rate:g}x (tempo and pitch together), so the timeline is now {"longer" if rate < 1 else "shorter"}.')
    return {'target': results[0][1] if len(results) == 1 else None}


def uuid_short():
    return uid()[:12]


def export_duration(stems, original_duration):
    """Mix length: the source's, unless a whole-song speed chip changed the layers' length."""
    if not any(abs(net_speed((record(s) or {}).get('items', [])) - 1) > 1e-9 for s in stems): return original_duration
    return max(float(s.get('start_offset') or 0) + float(s['duration']) for s in stems)
