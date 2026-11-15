"""Moon pull: symbolic rules that hold for the whole clip ("whenever X happens, sound Y does Z").

The editor turns a rule into moments on the timeline. Gain responses (swell, hush) stay in the
editor as automation. The responses that change the audio itself live here:

* effect over many ranges at once (deepen, impact, disorient, muffle from app.stem_fx);
* echo: copy a motif sound (for example the loudest thunder boom) onto every moment;
* revert: put back the file a Moon pull change replaced, so a rule can be edited and re-applied;
* transcript: timed lines of a layer, so a rule can follow what a character says.

All times in requests are timeline seconds; they are converted to each stem's own time here.
"""
import json
import os
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from .core import run, uid
from .stem_fx import chain

FADE = 0.03
MAX_RANGES = 64
#: Above this a mixed sample is bent back softly instead of clipping.
KNEE = 0.85


def _read(path):
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    return data, sr


def _match_channels(a, ch):
    if a.shape[1] == ch: return a
    if a.shape[1] == 1: return np.repeat(a, ch, axis=1)
    return a.mean(axis=1, keepdims=True).repeat(ch, axis=1)


def soft_limit(x):
    """Identity below KNEE, then a tanh curve that never reaches 1."""
    mag = np.abs(x)
    over = mag > KNEE
    if not over.any(): return x
    out = x.copy()
    out[over] = np.sign(x[over]) * np.minimum(KNEE + (1 - KNEE) * np.tanh((mag[over] - KNEE) / (1 - KNEE)), 0.999)
    return out


def to_stem_ranges(ranges, offset, duration):
    """Timeline ranges to the stem's own time, clipped to the stem. Ranges that miss it are dropped."""
    out = []
    for start, end in ranges:
        if not (np.isfinite(start) and np.isfinite(end)) or end <= start: raise ValueError('Each range needs start < end')
        a, b = max(0.0, start - offset), min(duration, end - offset)
        if b - a > 0.01: out.append((a, b))
    if not out: raise ValueError('None of the moments fall inside this layer')
    return sorted(out)


def range_mask(n, sr, ranges, fade=FADE):
    """1 inside any range, 0 outside, with linear fades of ``fade`` seconds inside each edge."""
    t = np.arange(n) / sr
    mask = np.zeros(n, dtype=np.float32)
    for start, end in ranges:
        m = np.clip(np.minimum((t - start) / fade, (end - t) / fade), 0, 1)
        np.maximum(mask, m, out=mask)
    return mask[:, None]


def write_next(src, out, sr):
    dest = Path(src).with_name(uid() + '.wav')
    sf.write(dest, out, sr, subtype='FLOAT')
    return dest


def apply_effect_ranges(path, effect, ranges, amount=0.7):
    """Like app.effects.apply_effect, for many ranges in one render. Returns the new WAV path."""
    src = Path(path)
    dry, sr = _read(src)
    wet_path = src.with_name(uid() + '.wet.wav')
    try:
        run('ffmpeg', '-v', 'error', '-y', '-i', str(src), '-af', chain(effect, amount, sr), '-ar', str(sr), str(wet_path))
        wet, _ = _read(wet_path)
    finally:
        wet_path.unlink(missing_ok=True)
    ch = max(dry.shape[1], wet.shape[1])
    dry, wet = _match_channels(dry, ch), _match_channels(wet, ch)
    wet = np.pad(wet, ((0, max(0, len(dry) - len(wet))), (0, 0)))[:len(dry)]
    mask = range_mask(len(dry), sr, ranges)
    return write_next(src, np.clip(dry * (1 - mask) + wet * mask, -1, 1), sr)


def apply_echo(path, source_path, source_range, times, gain=1.0):
    """Mix the snippet ``source_range`` of ``source_path`` into ``path`` at each time in ``times``.

    Times and the source range are in each file's own seconds. The snippet fades in over 10 ms and
    out over the last quarter (at most 0.6 s), so a boom lands cleanly and dies away naturally.
    """
    dest_audio, sr = _read(path)
    src_audio, src_sr = _read(source_path)
    a, b = source_range
    b = min(b, len(src_audio) / src_sr)  # rounding can put the end a hair past the last sample
    if not (0 <= a < b): raise ValueError('The motif sound must lie inside its layer')
    snip = src_audio[int(a * src_sr):int(b * src_sr)]
    if src_sr != sr:
        idx = np.arange(int(len(snip) * sr / src_sr)) * (src_sr / sr)
        snip = np.stack([np.interp(idx, np.arange(len(snip)), snip[:, c]) for c in range(snip.shape[1])], axis=1).astype(np.float32)
    if len(snip) < int(0.05 * sr): raise ValueError('The motif sound is too short')
    ch = max(dest_audio.shape[1], snip.shape[1])
    dest_audio, snip = _match_channels(dest_audio, ch), _match_channels(snip, ch)
    n = len(snip)
    env = np.ones(n, dtype=np.float32)
    fin, fout = int(0.01 * sr), min(int(0.6 * sr), n // 4)
    env[:fin] = np.linspace(0, 1, fin, endpoint=False)
    if fout > 0: env[-fout:] = np.linspace(1, 0, fout)
    snip = snip * env[:, None] * float(gain)
    out = dest_audio.copy()
    placed = 0
    for t in times:
        start = int(round(t * sr))
        if start >= len(out) or start + n <= 0: continue
        lo, hi = max(0, start), min(len(out), start + n)
        out[lo:hi] += snip[lo - start:hi - start]
        placed += 1
    if placed == 0: raise ValueError('None of the moments fall inside this layer')
    return write_next(path, soft_limit(out), sr)


def loudest_window(path, seconds=2.0):
    """Start and end, in the file's own seconds, of the loudest ``seconds`` long stretch."""
    audio, sr = _read(path)
    mono = audio.mean(axis=1)
    hop = max(1, int(0.05 * sr))
    energy = np.add.reduceat(mono ** 2, np.arange(0, len(mono), hop))
    w = max(1, int(seconds / 0.05))
    if len(energy) <= w: return 0.0, len(mono) / sr
    sums = np.convolve(energy, np.ones(w), mode='valid')
    i = int(np.argmax(sums))
    # Every window holding the whole sound ties, so start where it actually rises, a little
    # early so the attack is kept.
    part = energy[i:i + w]
    i += int(np.argmax(part >= 0.1 * part.max()))
    start = max(0.0, i * hop / sr - 0.1)
    return round(start, 3), round(min(len(mono) / sr, start + seconds), 3)


# Revert: every Moon pull change leaves a note saying which file it replaced.

def note_path(dest): return Path(str(dest) + '.moonpull.json')


def write_note(dest, previous_path, rule=None):
    note_path(dest).write_text(json.dumps({'previous': str(previous_path), 'rule': rule}))


def previous_of(path):
    note = note_path(path)
    if not note.exists(): return None
    prev = json.loads(note.read_text()).get('previous')
    return prev if prev and Path(prev).exists() else None


# Transcript: timed lines of a layer, from OpenAI's transcription API.

class TranscriptError(Exception):
    pass


def _key():
    from .agent import AgentError, _key as key
    try: return key()
    except AgentError as exc: raise TranscriptError(str(exc)) from exc


def parse_transcript(data, offset):
    def clean(items, field):
        out = []
        for it in items or []:
            try: s, e, text = float(it['start']), float(it['end']), str(it[field]).strip()
            except (KeyError, TypeError, ValueError): continue
            if text and e >= s: out.append({'start': round(s + offset, 3), 'end': round(e + offset, 3), 'text': text})
        return out
    return {'text': str(data.get('text', '')).strip(), 'lines': clean(data.get('segments'), 'text'), 'words': clean(data.get('words'), 'word')}


def transcribe(path, offset=0.0):
    """Lines and words of a layer, in timeline seconds. Cached next to the audio file, so a
    changed layer (new file) is transcribed again."""
    cache = Path(str(path) + '.transcript.json')
    if cache.exists():
        try: return parse_transcript(json.loads(cache.read_text()), offset)
        except (ValueError, OSError): pass
    small = Path(path).with_name(uid() + '.transcribe.mp3')
    try:
        run('ffmpeg', '-v', 'error', '-y', '-i', str(path), '-ac', '1', '-ar', '16000', '-b:a', '48k', str(small))
        with small.open('rb') as f:
            try:
                response = httpx.post('https://api.openai.com/v1/audio/transcriptions',
                    headers={'Authorization': 'Bearer ' + _key()},
                    data={'model': os.getenv('OPENAI_TRANSCRIBE_MODEL', 'whisper-1'), 'response_format': 'verbose_json',
                          'timestamp_granularities[]': ['segment', 'word']},
                    files={'file': ('layer.mp3', f, 'audio/mpeg')}, timeout=180)
            except httpx.HTTPError as exc: raise TranscriptError('Could not reach OpenAI to read the lines. Please retry.') from exc
    finally:
        small.unlink(missing_ok=True)
    if response.is_error:
        if response.status_code in (401, 403): raise TranscriptError('OpenAI rejected the API key or model access.')
        if response.status_code == 429: raise TranscriptError('OpenAI rate limit or quota exceeded.')
        raise TranscriptError(f'Transcription failed (HTTP {response.status_code}).')
    data = response.json()
    try: cache.write_text(json.dumps(data))
    except OSError: pass
    return parse_transcript(data, offset)


# Layer: a sound (an uploaded clip or a generated effect) that plays during every moment, on a new
# layer of its own, so "whenever this person is on screen, their music plays" stays mixable.

LAYER_FADE_IN = 0.08
LAYER_FADE_OUT = 0.4


def place_sound(sound_path, ranges, total, dest, sr=48000, gain=1.0, loop=True):
    """Write ``total`` seconds of audio to ``dest``: silence, with the sound starting at each range.

    The sound restarts at every range (a theme that answers each appearance). It loops to fill a
    range longer than itself when ``loop`` is set, and is cut short with a fade where the range ends.
    Ranges are in the new file's own seconds (timeline seconds, since it starts at 0).
    """
    snd, snd_sr = _read(sound_path)
    if snd_sr != sr:
        idx = np.arange(int(len(snd) * sr / snd_sr)) * (snd_sr / sr)
        snd = np.stack([np.interp(idx, np.arange(len(snd)), snd[:, c]) for c in range(snd.shape[1])], axis=1).astype(np.float32)
    if len(snd) < int(0.05 * sr): raise ValueError('The sound is too short')
    snd = _match_channels(snd, 2)
    n_total = int(round(total * sr))
    if n_total <= 0: raise ValueError('The clip has no length')
    out = np.zeros((n_total, 2), dtype=np.float32)
    placed = 0
    for start, end in ranges:
        a, b = max(0, int(round(start * sr))), min(n_total, int(round(end * sr)))
        if b - a < int(0.05 * sr): continue
        n = b - a
        if loop and n > len(snd):
            piece = np.tile(snd, (int(np.ceil(n / len(snd))), 1))[:n]
        else:
            piece = snd[:n].copy()
        m = len(piece)
        env = np.ones(m, dtype=np.float32)
        fin, fout = min(int(LAYER_FADE_IN * sr), m // 4), min(int(LAYER_FADE_OUT * sr), m // 3)
        if fin > 0: env[:fin] = np.linspace(0, 1, fin, endpoint=False)
        # Only fade out where the range cuts the sound off; a sound that ends by itself keeps its tail.
        if fout > 0 and (loop or n < len(snd)): env[-fout:] = np.minimum(env[-fout:], np.linspace(1, 0, fout))
        out[a:a + m] += piece * env[:, None] * float(gain)
        placed += 1
    if placed == 0: raise ValueError('None of the moments fall inside the clip')
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, soft_limit(out), sr, subtype='FLOAT')
    return Path(dest)
