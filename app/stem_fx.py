"""Time-ranged effects on one stem: deepen (pitch down), impact (bass + loudness, limited) and
disorient (autopan + flanger + echo) and muffle (low-pass, as if heard through a wall or mask). The effect is rendered over the whole stem with FFmpeg and
blended in only inside [start, end] with short fades, so the rest of the layer is untouched and
the timeline stays sample aligned."""
from pathlib import Path
import numpy as np
import soundfile as sf
from .core import run, uid

EFFECTS = ('deepen', 'impact', 'disorient', 'muffle')
FADE = 0.03

def chain(effect, amount, sr):
    amount = min(max(amount, 0.0), 1.0)
    if effect == 'deepen':
        # amount 1 = one octave down; rubberband keeps duration and formants follow the pitch (darker, bigger voice)
        return f'rubberband=pitch={2 ** (-amount):.4f}:formant=shifted,bass=g={4 * amount:.1f}:f=150'
    if effect == 'impact':
        return f'bass=g={6 + 10 * amount:.1f}:f=90:w=0.8,volume={1 + 2 * amount:.2f},alimiter=limit=0.97:attack=1:release=60'
    if effect == 'disorient':
        return (f'aformat=channel_layouts=stereo,apulsator=hz={1 + 5 * amount:.2f}:amount={0.5 + 0.5 * amount:.2f},'
                f'flanger=delay={2 + 8 * amount:.1f}:depth={2 + 8 * amount:.1f}:speed=0.8,'
                f'aecho=0.8:0.7:{int(60 + 120 * amount)}|{int(140 + 200 * amount)}:0.35|0.25,volume={1 + amount:.2f},alimiter=limit=0.97')
    if effect == 'muffle':
        return f'lowpass=f={1200 - 900 * amount:.0f}:p=2,lowpass=f={1200 - 900 * amount:.0f},volume={1 - 0.4 * amount:.2f}'
    raise ValueError(f'Unknown effect {effect!r}; use one of {", ".join(EFFECTS)}')

def apply_effect(path, effect, start, end, amount=0.7):
    """Write a new WAV next to ``path`` and return its path."""
    src = Path(path)
    dry, sr = sf.read(src, dtype='float32', always_2d=True)
    dur = len(dry) / sr
    if not (0 <= start < end <= dur + 1e-6): raise ValueError(f'Range must be within 0..{dur:.2f} s')
    wet_path = src.with_name(uid() + '.wet.wav')
    try:
        run('ffmpeg', '-v', 'error', '-y', '-i', str(src), '-af', chain(effect, amount, sr), '-ar', str(sr), str(wet_path))
        wet, _ = sf.read(wet_path, dtype='float32', always_2d=True)
    finally:
        wet_path.unlink(missing_ok=True)
    ch = max(dry.shape[1], wet.shape[1])
    dry = np.repeat(dry, ch // dry.shape[1], axis=1) if dry.shape[1] != ch else dry
    wet = np.repeat(wet, ch // wet.shape[1], axis=1) if wet.shape[1] != ch else wet
    wet = np.pad(wet, ((0, max(0, len(dry) - len(wet))), (0, 0)))[:len(dry)]
    t = np.arange(len(dry)) / sr
    mask = np.clip(np.minimum((t - start) / FADE, (end - t) / FADE), 0, 1)[:, None]
    out = dry * (1 - mask) + wet * mask
    dest = src.with_name(uid() + '.wav')
    sf.write(dest, out, sr, subtype='FLOAT')
    return dest
