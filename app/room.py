"""Making room for a boosted layer, as the export mixes it.

A layer pushed past 100% mostly drives the mix into the limiter, which holds its absolute level, so on its own a
500% voice ends up only about 12 dB over the rest. So while a boosted layer is actually making sound, every other
layer steps back by the same factor: at level g the others play at 1/g. At 500% a voice then sits about 26 dB over
the rest while it talks, and the scene comes back between its lines.

The editor preview applies the same rule (frontend/src/audio/room.ts); keep the constants and steps in step.
"""
import json
import math
import numpy as np

HOP = 0.01            # activity is measured in 10 ms hops
HOLD_HOPS = 25        # stay active 250 ms after the sound stops, so short pauses do not pump
ATTACK_STEP = 0.5     # reach full room in 20 ms
RELEASE_STEP = 0.04   # give it back over 250 ms
SILENT_DB = -60.0
FLOOR_DB = -45.0
BELOW_LOUD_DB = 24.0


def activity(samples, rate):
    """0..1 per hop: how much this layer is sounding (1 = clearly sounding). samples is (frames, channels)."""
    hop = max(1, round(rate * HOP))
    mono = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1).mean(axis=1)
    n = len(mono) // hop
    if not n: return np.zeros(0)
    db = 20 * np.log10(np.sqrt(np.mean(np.square(mono[:n * hop].reshape(n, hop)), axis=1)) + 1e-12)
    voiced = np.sort(db[db > SILENT_DB])
    if not voiced.size: return np.zeros(n)
    threshold = max(FLOOR_DB, voiced[math.floor(0.95 * (voiced.size - 1))] - BELOW_LOUD_DB)
    out = np.zeros(n); a = 0.0; last_on = -10**9
    for k, on in enumerate(db > threshold):
        if on: last_on = k
        a = min(1.0, a + ATTACK_STEP) if k - last_on <= HOLD_HOPS else max(0.0, a - RELEASE_STEP)
        out[k] = a
    return out


def level(stem, times):
    """Fader x volume curve x mute ranges at timeline times (keyframes are linear, held flat past both ends)."""
    points = json.loads(stem['automation'] or '[]')
    curve = np.interp(times, [float(p['time']) for p in points], [float(p['value']) for p in points]) if points else np.ones_like(times)
    out = float(stem['volume']) * curve
    for start, end in json.loads(stem['mute_ranges'] or '[]'):
        out[(times >= float(start)) & (times <= float(end))] = 0
    return out


def peak_level(stem):
    points = json.loads(stem['automation'] or '[]')
    return float(stem['volume']) * max([float(p['value']) for p in points] or [1.0])


def room_tracks(active, read, length):
    """For each audible layer, None if nothing needs to make room for, else its room factor every HOP seconds of
    the timeline (index k is time k*HOP), covering `length` seconds. read(path) gives (samples, rate)."""
    boosted = [i for i, s in enumerate(active) if peak_level(s) > 1]
    if not boosted: return [None] * len(active)
    times = np.arange(math.ceil(length / HOP) + 1) * HOP
    push = {}
    for i in boosted:
        s = active[i]; samples, rate = read(s['path'])
        a = activity(samples, rate)
        k = np.floor((times - float(s['start_offset'])) / HOP + 1e-9).astype(int)
        act = np.where((k >= 0) & (k < a.size), a[np.clip(k, 0, max(a.size - 1, 0))] if a.size else 0, 0)
        push[i] = np.maximum(level(s, times) - 1, 0) * act
    out = []
    for i in range(len(active)):
        others = [push[j] for j in boosted if j != i]
        out.append(None if not others else 1 / (1 + np.max(others, axis=0)))
    return out
