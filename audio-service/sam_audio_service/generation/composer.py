"""Rule-based arranger: writes a part for (instrument, role) that follows the analysed chords,
sits on the beat grid, stays in the instrument's register and keeps out of the vocal's way.

Roles: pad (voice-led sustained chords), counter_melody (sustains under a busy vocal, answers in
its gaps), lead (plays only in vocal gaps), ostinato (arpeggiator over chord tones), bassline
(roots/fifths/approach notes, locks to the kick when a drum stem was analysed, funk/rock/gentle
templates), riff (a one-bar motif repeated and transposed per chord), stabs (short voiced chords
on off-beats), chords (piano/guitar comping: bass note + voiced right hand, or strums).

Deterministic for a given seed. Input is the JSON analysis (``music_analysis.MusicAnalyzer``), so
a client can also send an edited analysis back.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..music_analysis import QUALITIES, key_at, scale_pcs
from .vocabulary import PartSpec


@dataclass
class PNote:
    start: float
    end: float
    pitch: int
    velocity: int
    articulation: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 4), "end": round(self.end, 4), "pitch": int(self.pitch), "velocity": int(self.velocity), "articulation": self.articulation}


DISSONANT = {1, 11, 6}  # minor 2nd / major 7th / tritone against the vocal note sounding at the same time


class Context:
    def __init__(self, analysis: dict[str, Any], windows: list[tuple[float, float]]):
        self.a = analysis
        self.duration = float(analysis["duration"])
        self.beats = np.asarray(analysis["beats"], dtype=np.float64)
        if len(self.beats) < 2:
            self.beats = np.arange(0, self.duration, 0.5)
        self.period = float(np.median(np.diff(self.beats)))
        # Extend the grid by one beat so the last bar has an end.
        self.grid = np.r_[self.beats, self.beats[-1] + self.period]
        down = np.asarray(analysis.get("downbeats") or self.beats[::4], dtype=np.float64)
        self.downbeats = down
        self.chords = [c for c in analysis.get("chords", [])]
        self.key = analysis.get("key") or {"pc": 0, "mode": "major"}
        self.scale = sorted(scale_pcs(self.key))
        self._scales = [(float(s["start"]), float(s["end"]), sorted(scale_pcs(s["key"]))) for s in analysis.get("sections") or [] if s.get("key")]
        mel = analysis.get("melody_midi") or []
        self.melody = [(float(n["start"]), float(n["end"]), int(n["pitch"])) for n in mel]
        self.mel_starts = np.array([m[0] for m in self.melody]) if self.melody else np.zeros(0)
        self.vocal_activity = analysis.get("vocal_activity") or []
        self.windows = windows
        self.swing = float((analysis.get("groove") or {}).get("swing", 0.5))
        self.kicks = np.asarray((analysis.get("groove") or {}).get("kick_times") or [], dtype=np.float64)
        sec = analysis.get("sections") or []
        self.sections = sec
        reg = analysis.get("register") or {}
        self.vocal_range = None
        if "vocals" in reg:
            self.vocal_range = (int(reg["vocals"]["low"]), int(reg["vocals"]["high"]))
        elif self.melody:
            p = np.array([m[2] for m in self.melody])
            self.vocal_range = (int(np.percentile(p, 10)), int(np.percentile(p, 90)))
        self.crowded = [(int(v["low"]), int(v["high"])) for k, v in reg.items() if k not in ("vocals", "bass", "drums")]

    # -- lookups
    def scale_at(self, t: float) -> list[int]:
        for a, b, sc in self._scales:
            if a <= t < b:
                return sc
        return self.scale

    def chord_at(self, t: float) -> tuple[int, str] | None:
        for c in self.chords:
            if c["start"] <= t < c["end"] and c.get("root_pc") is not None:
                return int(c["root_pc"]), c.get("quality") or "maj"
        # Nearest preceding real chord (N regions keep the previous harmony).
        prev = [c for c in self.chords if c["start"] <= t and c.get("root_pc") is not None]
        if prev:
            return int(prev[-1]["root_pc"]), prev[-1].get("quality") or "maj"
        nxt = [c for c in self.chords if c.get("root_pc") is not None]
        return (int(nxt[0]["root_pc"]), nxt[0].get("quality") or "maj") if nxt else None

    def chord_pcs(self, t: float, sevenths: bool = False) -> list[int]:
        ch = self.chord_at(t)
        if ch is None:
            return [self.key.get("pc", 0), (self.key.get("pc", 0) + 4) % 12, (self.key.get("pc", 0) + 7) % 12]
        r, q = ch
        pcs = [(r + i) % 12 for i in QUALITIES.get(q, (0, 4, 7))]
        if sevenths:
            pcs.append((r + (10 if q == "min" or (r + 10) % 12 in self.scale_at(t) else 11)) % 12)
        return pcs

    def vocal_at(self, a: float, b: float) -> list[int]:
        return [p for s, e, p in self.melody if s < b and a < e]

    def vocal_busy(self, a: float, b: float) -> float:
        if self.melody:
            covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e, _ in self.melody)
        else:
            covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e in self.vocal_activity)
        return covered / max(1e-6, b - a)

    def near_vocal_onset(self, t: float, tol: float = 0.07) -> bool:
        return bool(self.mel_starts.size and np.min(np.abs(self.mel_starts - t)) < tol)

    def in_windows(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.windows)

    def window_end(self, t: float) -> float:
        for a, b in self.windows:
            if a <= t < b:
                return b
        return self.duration

    def energy_at(self, t: float) -> float:
        for s in self.sections:
            if s["start"] <= t < s["end"]:
                return float(s.get("energy", 0.7))
        return 0.7

    def bars(self) -> list[tuple[float, float, int]]:
        """(start, end, first_beat_index) for every bar, from the downbeats (4/4)."""
        out = []
        idx = [int(np.argmin(np.abs(self.beats - d))) for d in self.downbeats]
        if not idx or idx[0] > 0:
            # Pickup: a partial bar from beat 0 to the first downbeat.
            idx = [0] + idx
        idx = sorted(set(idx))
        for i, b0 in enumerate(idx):
            b1 = idx[i + 1] if i + 1 < len(idx) else min(len(self.grid) - 1, b0 + 4)
            if b1 <= b0:
                continue
            out.append((float(self.grid[b0]), float(self.grid[b1]), b0))
        return out

    def subdivisions(self, b0: int, b1: int, k: int) -> list[float]:
        """Grid points (k per beat) between beat indices b0..b1, swung for k == 2 or 4."""
        pts = []
        for i in range(b0, b1):
            a, b = float(self.grid[i]), float(self.grid[min(i + 1, len(self.grid) - 1)])
            for j in range(k):
                frac = j / k
                if self.swing > 0.52 and k == 2 and j == 1:
                    frac = self.swing
                elif self.swing > 0.52 and k == 4 and j % 2 == 1:
                    frac = j / k + (self.swing - 0.5) / 2
                pts.append(a + frac * (b - a))
        return pts


def plan_register(spec: PartSpec, ctx: Context) -> tuple[int, int]:
    """Working range: the instrument range, moved off the vocal register where possible."""
    lo, hi = spec.low, spec.high
    if spec.role == "bassline":
        return lo, min(hi, lo + 19)
    if ctx.vocal_range is None:
        mid = (lo + hi) // 2
        return max(lo, mid - 10), min(hi, mid + 10)
    vlo, vhi = ctx.vocal_range
    below = (lo, min(hi, vlo - 1))
    above = (max(lo, vhi + 2), hi)
    span = lambda r: r[1] - r[0]  # noqa: E731
    center = (lo + hi) / 2
    if spec.role in ("pad", "chords", "stabs"):
        # Accompaniment sits under a high (e.g. female) voice, but above a low one: voicings
        # below ~A3 get muddy and collide with the bass.
        if span(below) >= 10 and below[1] >= 57:
            return max(lo, below[1] - 16), below[1]
        if span(above) >= 10:
            return above[0], min(hi, above[0] + 17)
        if span(below) >= 10:
            return max(lo, below[1] - 16), below[1]
        return lo, hi
    prefer_above = center >= (vlo + vhi) / 2 - 2
    first, second = (above, below) if prefer_above else (below, above)
    for r in (first, second):
        if span(r) >= 9:
            return (r[0], min(r[1], r[0] + 16)) if r is above else (max(r[0], r[1] - 16), r[1])
    return lo, hi


def _crowd_penalty(p: int, ctx: Context) -> float:
    return sum(0.15 for lo, hi in ctx.crowded if lo <= p <= hi)


def voice_chord(pcs: list[int], lo: int, hi: int, n: int, prev: list[int] | None, ctx: Context, a: float, b: float) -> list[int]:
    """Choose an n-voice voicing of the chord in [lo, hi] with minimal motion from ``prev``."""
    cands = [p for p in range(lo, hi + 1) if p % 12 in pcs]
    vocal = ctx.vocal_at(a, b)
    best, best_cost = None, 1e9
    target = (lo + hi) / 2
    for combo in itertools.combinations(cands, n):
        span = combo[-1] - combo[0]
        if span > 19 or span < 5:
            continue
        gaps = np.diff(combo)
        if np.any(gaps < 2) or np.any(gaps > 9):
            continue
        if combo[0] < 52 and gaps[0] < 5:  # low thirds get muddy
            continue
        got = {p % 12 for p in combo}
        if pcs[0] not in got or (len(pcs) > 1 and pcs[1] not in got):
            continue
        cost = 0.0
        if prev and len(prev) == n:
            cost += sum(abs(x - y) for x, y in zip(combo, prev))
        else:
            cost += 2 * abs(np.mean(combo) - target)
        cost += sum(_crowd_penalty(p, ctx) for p in combo)
        for v in vocal:
            for p in combo:
                iv = abs(p - v) % 12
                if iv in DISSONANT and abs(p - v) < 13:
                    cost += 2.0
                if p == v:
                    cost += 1.0
        if combo[0] % 12 != pcs[0]:
            cost += 1.5  # prefer root position, inversions still allowed for smooth motion
        if cost < best_cost:
            best, best_cost = list(combo), cost
    if best is None:  # range too narrow: stack chord tones upwards from the lowest candidate
        base = cands[0] if cands else lo
        best = sorted({base + ((pc - base) % 12) for pc in pcs})[:n]
    return best


def _vel(base: float, ctx: Context, t: float, spread: float = 30) -> int:
    return int(np.clip(base + spread * (ctx.energy_at(t) - 0.6), 25, 120))


def compose(spec: PartSpec, analysis: dict[str, Any], windows: list[tuple[float, float]], seed: int = 0, humanize: bool = True) -> list[PNote]:
    ctx = Context(analysis, windows)
    rng = np.random.default_rng(seed)
    role = spec.role
    fn = {"pad": _pad, "counter_melody": _line, "lead": _line, "ostinato": _ostinato, "bassline": _bassline,
          "riff": _riff, "stabs": _stabs, "chords": _chords}.get(role, _pad)
    notes = fn(spec, ctx, rng)
    notes = [n for n in notes if ctx.in_windows(n.start)]
    for n in notes:
        n.end = min(n.end, ctx.window_end(n.start), ctx.duration)
        n.pitch = int(np.clip(n.pitch, spec.low, spec.high))
    notes = [n for n in notes if n.end - n.start > 0.03]
    if humanize:
        jitter = 0.012 if role in ("pad",) else 0.007
        for n in notes:
            d = float(np.clip(rng.normal(0, jitter), -2.5 * jitter, 2.5 * jitter))
            n.start = max(0.0, n.start + d)
            n.end = max(n.start + 0.03, n.end + d)
            n.velocity = int(np.clip(n.velocity + rng.normal(0, 5), 20, 125))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


# --------------------------------------------------------------------------- roles

def _chord_spans(ctx: Context, max_len_bars: int = 1) -> list[tuple[float, float]]:
    """Chord segments split at bar lines (so long chords re-articulate every bar)."""
    bars = ctx.bars()
    edges = sorted({0.0, ctx.duration} | {round(c["start"], 4) for c in ctx.chords} | {round(b[0], 4) for b in bars[::max_len_bars]})
    spans = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= 0.25 * ctx.period:
            spans.append((a, b))
        elif spans:
            spans[-1] = (spans[-1][0], b)
    return spans


def _pad(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    lo, hi = plan_register(spec, ctx)
    n_voices = 3 if spec.family in ("strings", "brass", "winds") and spec.high - spec.low < 50 else 4
    if spec.family == "strings" and spec.instrument == "strings":
        n_voices = 4
    bars_per = 2 if spec.density < 0.35 else 1
    prev = None
    out: list[PNote] = []
    sevenths = spec.style in ("jazz", "lofi", "ambient")
    for a, b in _chord_spans(ctx, bars_per):
        if not ctx.in_windows(a):
            prev = None
            continue
        pcs = ctx.chord_pcs(a, sevenths)
        voicing = voice_chord(pcs, lo, hi, min(n_voices, max(3, len(pcs))), prev, ctx, a, b)
        held = prev is not None and voicing == prev and out and abs(out[-1].end - a) < 0.05 and spec.density < 0.5
        vel = _vel(62 + 20 * spec.density, ctx, a)
        for p in voicing:
            if held:  # same chord continues: extend instead of re-attacking (smoother pads)
                for n in out[-len(voicing):]:
                    n.end = b + 0.05
                break
            out.append(PNote(a, b + 0.05, p, vel, "sustain"))
        prev = voicing
    return out


def _scale_cands(ctx: Context, lo: int, hi: int, t: float | None = None) -> list[int]:
    sc = ctx.scale_at(t) if t is not None else ctx.scale
    return [p for p in range(lo, hi + 1) if p % 12 in sc]


def _choose_pitch(ctx: Context, cands: list[int], prev: int | None, center: float, t0: float, t1: float, chord: list[int], strong: bool, rng) -> int:
    vocal = ctx.vocal_at(t0, t1)
    best, best_cost = cands[0], 1e9
    for p in cands:
        cost = 0.0
        is_ct = p % 12 in chord
        if strong and not is_ct:
            cost += 4.0
        elif not is_ct:
            cost += 0.8
        if prev is not None:
            leap = abs(p - prev)
            cost += 0.35 * leap + (3.0 if leap > 7 else 0.0) + (1.2 if leap == 0 else 0.0)
        cost += 0.08 * abs(p - center)
        for v in vocal:
            if abs(p - v) % 12 in DISSONANT and abs(p - v) < 14:
                cost += 5.0
            if abs(p - v) < 3:
                cost += 2.0  # too close to the voice: masks it
        cost += rng.uniform(0, 0.6)
        if cost < best_cost:
            best, best_cost = p, cost
    return best


def _line(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    """counter_melody: long tones under a busy vocal, motion in its gaps; lead: only in the gaps."""
    lo, hi = plan_register(spec, ctx)
    center = (lo + hi) / 2
    out: list[PNote] = []
    prev: int | None = None
    lead = spec.role == "lead"
    for bi, (a, b, b0) in enumerate(ctx.bars()):
        if not ctx.in_windows(a):
            prev = None
            continue
        cands = _scale_cands(ctx, lo, hi, a)
        busy = ctx.vocal_busy(a, b)
        if busy > 0.45 and (lead or spec.density < 0.3):
            continue  # leave the voice alone
        if busy > 0.45:
            # Sustain a chord tone for the bar (or two halves when the chord changes mid-bar).
            halves = [(a, (a + b) / 2), ((a + b) / 2, b)] if ctx.chord_at(a) != ctx.chord_at((a + b) / 2 + 1e-3) else [(a, b)]
            for h0, h1 in halves:
                chord = ctx.chord_pcs(h0)
                p = _choose_pitch(ctx, [c for c in cands if c % 12 in chord] or cands, prev, center, h0, h1, chord, True, rng)
                out.append(PNote(h0, h1 + 0.03, p, _vel(64, ctx, h0), "sustain"))
                prev = p
            continue
        # Vocal gap: a phrase on the 8th (or quarter) grid, avoiding vocal onsets.
        k = 2 if spec.density >= 0.4 else 1
        pts = ctx.subdivisions(b0, min(b0 + 4, len(ctx.grid) - 1), k)
        if not pts:
            continue
        step_len = (b - a) / max(1, len(pts))
        play_p = 0.35 + 0.55 * spec.density
        i = 0
        while i < len(pts):
            t = pts[i]
            strong = (i % k == 0)
            if (i > 0 and rng.random() > play_p) or (ctx.near_vocal_onset(t) and i > 0):
                i += 1
                continue
            # Note length: 1-3 steps, longer at phrase ends.
            span = 1 + int(rng.random() < 0.35) + int(i >= len(pts) - 2)
            t_end = pts[i + span] if i + span < len(pts) else b
            chord = ctx.chord_pcs(t)
            p = _choose_pitch(ctx, cands, prev, center, t, t_end, chord, strong, rng)
            out.append(PNote(t, t_end - 0.02 * step_len, p, _vel(72 + (8 if strong else 0), ctx, t)))
            prev = p
            i += span
    return out


ARP_PATTERNS = {
    "up": [0, 1, 2, 3, 4, 3, 2, 1],
    "updown": [0, 1, 2, 3, 2, 1],
    "80s": [0, 2, 1, 3, 2, 4, 3, 1],
    "gentle": [0, 1, 2, 1],
}


def _ostinato(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    lo, hi = plan_register(spec, ctx)
    k = 4 if spec.density >= 0.6 or spec.style == "80s" else 2
    pattern = ARP_PATTERNS["80s" if spec.style == "80s" else "gentle" if spec.density < 0.35 else "up"]
    gate = 0.5 if spec.articulation in ("spiccato", "pizzicato") or spec.family in ("synth", "plucked", "mallets") else 0.85
    out: list[PNote] = []
    step = 0
    bars = ctx.bars()
    for a, b, b0 in bars:
        if not ctx.in_windows(a):
            continue
        b1 = min(b0 + 4, len(ctx.grid) - 1)
        pts = ctx.subdivisions(b0, b1, k)
        for j, t in enumerate(pts):
            nxt = pts[j + 1] if j + 1 < len(pts) else b
            pcs = ctx.chord_pcs(t, spec.style in ("ambient", "jazz"))
            base = lo + ((pcs[0] - lo) % 12)
            if base + 12 > hi:
                base -= 12
            tones = sorted({base + ((pc - base) % 12) for pc in pcs})
            ladder = tones + [x + 12 for x in tones] + [x + 24 for x in tones]
            p = ladder[pattern[step % len(pattern)] % len(ladder)]
            while p > hi:
                p -= 12
            accent = 14 if j % k == 0 else 0
            out.append(PNote(t, t + gate * (nxt - t), p, _vel(64 + accent, ctx, t), spec.articulation))
            step += 1
    return out


BASS_TEMPLATES = {
    # 16 steps per bar: (step, degree, length_steps). degree: 'R' root, '5' fifth, 'O' octave, 'b7', 'A' approach.
    "gentle": [(0, "R", 8), (8, "R", 8)],
    "default": [(0, "R", 4), (4, "R", 2), (6, "5", 2), (8, "R", 4), (12, "5", 2), (14, "A", 2)],
    "rock": [(i, "R", 2) for i in range(0, 16, 2)],
    "funk": [(0, "R", 2), (3, "O", 1), (4, "R", 1), (6, "R", 2), (9, "5", 1), (10, "b7", 2), (12, "O", 1), (14, "R", 1), (15, "A", 1)],
    "jazz": [(0, "R", 4), (4, "3", 4), (8, "5", 4), (12, "A", 4)],
}


def _bassline(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    lo, hi = spec.low, min(spec.high, spec.low + 19)
    style = spec.style if spec.style in BASS_TEMPLATES else ("gentle" if spec.density < 0.35 else "funk" if spec.density > 0.75 else "default")
    tmpl = BASS_TEMPLATES[style]
    out: list[PNote] = []
    prev_root = None
    bars = ctx.bars()
    for bi, (a, b, b0) in enumerate(bars):
        if not ctx.in_windows(a):
            continue
        b1 = min(b0 + 4, len(ctx.grid) - 1)
        pts = ctx.subdivisions(b0, b1, 4)
        if len(pts) < 16:
            pts = pts + [b] * (16 - len(pts))
        pts = pts[:16] + [b]
        for step, deg, length in tmpl:
            t = pts[step]
            t_end = pts[min(16, step + length)]
            ch = ctx.chord_at(t + 1e-3)
            if ch is None:
                continue
            r, q = ch
            root = lo + ((r - lo) % 12)
            if prev_root is not None and abs(root - prev_root) > 7 and root - 12 >= lo:
                root = root - 12 if root - prev_root > 0 else root
            if deg == "R":
                p = root
            elif deg == "5":
                p = root + 7 if root + 7 <= hi else root - 5
            elif deg == "O":
                p = root + 12 if root + 12 <= hi else root
            elif deg == "3":
                p = root + (3 if q == "min" else 4)
            elif deg == "b7":
                p = root + 10 if root + 10 <= hi else root - 2
            else:  # approach the next chord's root chromatically / by step
                nxt = ctx.chord_at(t_end + 1e-3)
                target = lo + (((nxt[0] if nxt else r) - lo) % 12)
                sc = ctx.scale_at(t)
                p = target - 1 if (target - 1) % 12 in sc else target + 2 if (target + 2) % 12 in sc else target - 2
            prev_root = root
            ghost = deg in ("O",) and style == "funk"
            out.append(PNote(t, t_end - 0.01, int(p), _vel(70 - (14 if ghost else 0) + (12 if step == 0 else 0), ctx, t)))
    # Lock to the kick: move notes that land within 60 ms of a kick onto it.
    if ctx.kicks.size:
        for n in out:
            j = int(np.argmin(np.abs(ctx.kicks - n.start)))
            if abs(ctx.kicks[j] - n.start) < 0.06:
                shift = ctx.kicks[j] - n.start
                n.start += shift
                n.end += shift
    return out


def _riff(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    lo, hi = plan_register(spec, ctx)
    # One-bar motif on the 16th grid: (step, scale-degree offset from the chord root, length).
    pent_major, pent_minor = [0, 2, 4, 7, 9], [0, 3, 5, 7, 10]
    n_hits = int(3 + 5 * spec.density)
    steps = sorted({0} | set(int(s) for s in rng.choice(np.arange(1, 16), size=min(15, n_hits - 1), replace=False)))
    motif = []
    for i, s in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else 16
        motif.append((s, int(rng.integers(0, 5)) if i else 0, max(1, min(4, nxt - s))))
    out: list[PNote] = []
    for a, b, b0 in ctx.bars():
        if not ctx.in_windows(a):
            continue
        b1 = min(b0 + 4, len(ctx.grid) - 1)
        pts = ctx.subdivisions(b0, b1, 4)
        if len(pts) < 16:
            continue
        pts = pts[:16] + [b]
        for s, deg, length in motif:
            t = pts[s]
            ch = ctx.chord_at(t + 1e-3)
            if ch is None:
                continue
            r, q = ch
            pent = pent_minor if q == "min" else pent_major
            base = lo + ((r - lo) % 12)
            if s % 4 == 0 or length >= 2:  # strong steps and held notes: chord tones
                p = base + (0, 3 if q == "min" else 4, 7, 12)[deg % 4]
            else:  # quick passing notes: pentatonic over the chord
                p = base + pent[deg % 5]
                if (p % 12) not in ctx.scale_at(t):
                    p = base + 7
            while p > hi:
                p -= 12
            out.append(PNote(t, pts[min(16, s + length)] - 0.015, p, _vel(78 + (10 if s % 4 == 0 else 0), ctx, t)))
    return out


def _stabs(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    lo, hi = plan_register(spec, ctx)
    out: list[PNote] = []
    prev = None
    rhythm = [(6, 2), (14, 2)] if spec.density < 0.5 else [(0, 2), (6, 2), (10, 1), (14, 2)]
    for a, b, b0 in ctx.bars():
        if not ctx.in_windows(a):
            continue
        b1 = min(b0 + 4, len(ctx.grid) - 1)
        pts = ctx.subdivisions(b0, b1, 4)
        if len(pts) < 16:
            continue
        pts = pts[:16] + [b]
        for s, ln in rhythm:
            t = pts[s]
            pcs = ctx.chord_pcs(t + 1e-3)
            voicing = voice_chord(pcs, lo, hi, 3, prev, ctx, t, pts[s + ln])
            prev = voicing
            for p in voicing:
                out.append(PNote(t, t + 0.6 * (pts[s + ln] - t), p, _vel(84, ctx, t), "staccato"))
    return out


def _chords(spec: PartSpec, ctx: Context, rng) -> list[PNote]:
    """Keys: left-hand root + right-hand voicing; guitar: strummed voicings."""
    out: list[PNote] = []
    lo, hi = plan_register(spec, ctx)
    rh_lo, rh_hi = max(lo, spec.low + 12), hi
    prev = None
    guitar = spec.family == "guitar"
    if spec.density < 0.4:
        rhythm = [(0, 8), (8, 8)]
    elif spec.density < 0.7:
        rhythm = [(0, 6), (6, 4), (12, 4)]
    else:
        rhythm = [(0, 3), (3, 3), (6, 2), (8, 3), (11, 3), (14, 2)]
    for a, b, b0 in ctx.bars():
        if not ctx.in_windows(a):
            continue
        b1 = min(b0 + 4, len(ctx.grid) - 1)
        pts = ctx.subdivisions(b0, b1, 4)
        if len(pts) < 16:
            continue
        pts = pts[:16] + [b]
        for i, (s, ln) in enumerate(rhythm):
            t, t_end = pts[s], pts[min(16, s + ln)]
            pcs = ctx.chord_pcs(t + 1e-3, spec.style in ("jazz", "lofi"))
            n_v = 4 if guitar else 3
            voicing = voice_chord(pcs, rh_lo, rh_hi, n_v, prev, ctx, t, t_end)
            prev = voicing
            vel = _vel(66 + (10 if s == 0 else 0), ctx, t)
            for j, p in enumerate(voicing):
                offset = 0.012 * j if guitar else 0.0  # down-strum
                if guitar and i % 2 == 1:
                    offset = 0.010 * (len(voicing) - 1 - j)  # up-strum
                out.append(PNote(t + offset, t_end - 0.02, p, vel - (6 if guitar and i % 2 else 0)))
            if not guitar and s in (0, 8):
                root = spec.low + ((pcs[0] - spec.low) % 12)
                if root + 12 < voicing[0] - 4:
                    root += 12
                out.append(PNote(t, pts[min(16, s + 8)] - 0.02, root, vel - 4))
    return out
