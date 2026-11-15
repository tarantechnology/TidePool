"""Music analysis for generated parts: tempo/beats/downbeats, key, chords, sections, vocal melody,
vocal activity, groove, register occupancy and a rough room estimate.

The analysis is stem-aware. Each separated track is used for what it is best evidence of:

* ``drums``  -> beat/tempo onset evidence, swing, kick times, density per section
* ``bass``   -> chord roots (Basic Pitch notes when a transcriber is loaded, else low-band chroma)
* ``vocals`` -> melody MIDI (transcriber), where the voice is active, the space (decay tails)
* ``other`` / ``guitar`` / ``piano`` / ``keys`` / ``strings`` ... -> harmony chroma and register
  occupancy (which pitch ranges are already crowded)

With only a mix the same outputs are estimated from the mix (melody is then omitted).

Everything here is numpy only. Optional model backends plug in through two small protocols:
``BeatBackend.track(mono, sr) -> (beats, downbeats)`` (beat_this on the GPU host) and
``Transcriber.transcribe(mono, sr) -> list[Note]`` (Basic Pitch). Without them the numpy beat
tracker (Ellis-style dynamic programming) and chroma evidence are used, and the response says so.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from .generation import dsp

log = logging.getLogger("sam_audio_service")

ANALYSIS_SR = 22050
HOP = 512
PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7)}
# Krumhansl-Kessler key profiles.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # natural minor; harmonic V handled by the chord set

HARMONY_STEMS = ("other", "guitar", "piano", "keys", "strings", "synth", "organ", "instrumental", "accompaniment", "brass", "winds")


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int = 80

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 4), "end": round(self.end, 4), "pitch": int(self.pitch), "velocity": int(self.velocity)}


class BeatBackend(Protocol):
    name: str

    def track(self, mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]: ...


class Transcriber(Protocol):
    name: str

    def transcribe(self, mono: np.ndarray, sr: int) -> list[Note]: ...


# --------------------------------------------------------------------------- helpers

def _canonical_stem(name: str) -> str:
    n = name.lower().strip()
    for key in ("vocal", "voice", "sing"):
        if key in n:
            return "vocals"
    for key, canon in (("drum", "drums"), ("perc", "drums"), ("bass", "bass"), ("guitar", "guitar"), ("piano", "piano"), ("key", "keys"), ("string", "strings"), ("synth", "synth")):
        if key in n:
            return canon
    return n if n in HARMONY_STEMS else "other"


def pick_peaks(env: np.ndarray, min_distance: int, threshold: float) -> np.ndarray:
    peaks = []
    last = -min_distance
    for i in np.argsort(-env):
        if env[i] < threshold:
            break
        if all(abs(i - p) >= min_distance for p in peaks):
            peaks.append(i)
    del last
    return np.array(sorted(peaks), dtype=int)


def _local_peaks(env: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Local maxima above ``threshold`` at least ``min_distance`` frames apart (left to right)."""
    out = []
    for i in range(1, len(env) - 1):
        if env[i] >= threshold and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            if out and i - out[-1] < min_distance:
                if env[i] > env[out[-1]]:
                    out[-1] = i
                continue
            out.append(i)
    return np.array(out, dtype=int)


def chroma_matrix(sr: int, n_fft: int, fmin: float, fmax: float, sigma: float = 0.35) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    m = np.zeros((len(freqs), 12), np.float32)
    ok = (freqs >= fmin) & (freqs <= fmax)
    midi = dsp.hz_to_midi(freqs[ok])
    nearest = np.round(midi)
    w = np.exp(-0.5 * ((midi - nearest) / sigma) ** 2)
    m[np.where(ok)[0], (nearest.astype(int) % 12)] = w
    return m


def chroma(x: np.ndarray, sr: int, fmin: float, fmax: float, n_fft: int = 4096, hop: int = HOP) -> np.ndarray:
    mag = dsp.stft_mag(x, n_fft, hop)
    c = np.power(mag, 0.6) @ chroma_matrix(sr, n_fft, fmin, fmax)  # scale-invariant, keeps harmonics lower than log
    return c.astype(np.float32)  # (frames, 12), not normalised (energy carries the N decision)


def beat_sync(feat: np.ndarray, beat_frames: np.ndarray, reduce=np.mean) -> np.ndarray:
    """Average frame features between consecutive beat frames: (beats, dim)."""
    edges = np.concatenate([beat_frames, [len(feat)]])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = int(a), int(max(b, a + 1))
        out.append(reduce(feat[a:min(b, len(feat))], axis=0) if a < len(feat) else np.zeros(feat.shape[1]))
    return np.array(out, dtype=np.float32)


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


# --------------------------------------------------------------------------- beats

class NumpyBeatTracker:
    """Ellis (2007) dynamic-programming beat tracker on a spectral-flux onset envelope."""

    name = "numpy-dp"

    def track(self, mono: np.ndarray, sr: int, onset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        env = onset if onset is not None else dsp.onset_envelope(mono, sr, HOP)
        fps = sr / HOP
        tempo = estimate_tempo(env, fps)
        period = fps * 60 / tempo
        env_n = _gauss_smooth(env, max(1.0, period / 32)) / (env.std() + 1e-9)
        score = env_n.copy()
        back = np.full(len(env), -1)
        lo, hi = int(round(period / 2)), int(round(period * 2))
        alpha = 100.0
        for t in range(len(env)):
            a, b = t - hi, t - lo
            if b <= 0:
                continue
            prev = np.arange(max(0, a), b)
            penalty = -alpha * np.log((t - prev) / period) ** 2
            k = int(np.argmax(score[prev] + penalty))
            score[t] = env_n[t] + score[prev[k]] + penalty[k]
            back[t] = prev[k]
        # Start backtracking from the best-scoring frame among the last period.
        tail = np.arange(max(0, len(env) - int(period)), len(env))
        t = int(tail[np.argmax(score[tail])]) if len(tail) else 0
        beats = []
        while t >= 0:
            beats.append(t)
            t = back[t]
        beats = np.array(beats[::-1], dtype=np.float64) / fps
        return beats, np.array([], dtype=np.float64)  # downbeats chosen by the analyser


def _gauss_smooth(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2)
    return np.convolve(x, k / k.sum(), mode="same")


def estimate_tempo(env: np.ndarray, fps: float, bpm_range=(60.0, 200.0), prior_bpm: float = 110.0) -> float:
    env = _gauss_smooth(env)  # spiky onsets split autocorrelation energy across neighbouring lags
    env = env - env.mean()
    n = len(env)
    if n < 8:
        return prior_bpm
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(env, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n]
    lags = np.arange(1, n)
    bpm = 60 * fps / lags
    ok = (bpm >= bpm_range[0]) & (bpm <= bpm_range[1])
    weight = np.exp(-0.5 * (np.log2(bpm[ok] / prior_bpm) / 0.9) ** 2)
    strength = ac[1:][ok] * weight
    best = lags[ok][int(np.argmax(strength))]
    # Backbeat patterns (kick / snare alternating) make the two-beat lag win; prefer the faster
    # metrical level when its own periodicity is still strong.
    half = int(round(best / 2))
    if half >= 1 and 60 * fps / half <= bpm_range[1] and 60 * fps / best < 90:
        h = ac[max(1, half - 1):half + 2].max()
        if h > 0.35 * ac[best]:
            best = int(half - 1 + np.argmax(ac[max(1, half - 1):half + 2]))
    # Parabolic refinement of the lag.
    if 1 < best < n - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        d = a - 2 * b + c
        best = best + (0.5 * (a - c) / d if abs(d) > 1e-12 else 0)
    return float(60 * fps / best)


# --------------------------------------------------------------------------- analysis

class MusicAnalyzer:
    def __init__(self, beat_backend: BeatBackend | None = None, transcriber: Transcriber | None = None):
        self.beat_backend = beat_backend
        self.transcriber = transcriber

    def analyze(self, mix: np.ndarray | None, sr: int, stems: dict[str, np.ndarray] | None = None, *, melody: bool = True) -> dict[str, Any]:
        stems = {(_canonical_stem(k)): v for k, v in (stems or {}).items() if v is not None}
        if mix is None:
            if not stems:
                raise ValueError("analysis needs a mix or at least one stem")
            n = max(v.shape[-1] for v in stems.values())
            mix = sum(np.pad(dsp.as_mono(v), (0, n - v.shape[-1])) for v in stems.values())
        # Transcriptions (ONNX, releases the GIL) run in the background while the numpy analysis runs.
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=3)
        mono = dsp.resample(dsp.as_mono(mix), sr, ANALYSIS_SR)
        mono_stems = {k: dsp.resample(dsp.as_mono(v), sr, ANALYSIS_SR) for k, v in stems.items()}
        futures = {}
        if self.transcriber is not None:
            if "bass" in mono_stems:
                futures["bass"] = pool.submit(self.transcriber.transcribe, mono_stems["bass"], ANALYSIS_SR)
            if melody and "vocals" in mono_stems:
                futures["vocals"] = pool.submit(self.transcriber.transcribe, mono_stems["vocals"], ANALYSIS_SR)
        duration = len(mono) / ANALYSIS_SR
        warnings: list[str] = []
        used = sorted(mono_stems)

        # ---- beats
        mix_onset = dsp.onset_envelope(mono, ANALYSIS_SR, HOP)
        drums = mono_stems.get("drums")
        onset = mix_onset
        if drums is not None:
            d_on = _fit(dsp.onset_envelope(drums, ANALYSIS_SR, HOP), len(mix_onset))
            onset = d_on / (d_on.std() + 1e-9) + 0.5 * mix_onset / (mix_onset.std() + 1e-9)
        beat_name = NumpyBeatTracker.name
        beats = downbeats = None
        if self.beat_backend is not None:
            try:
                beats, downbeats = self.beat_backend.track(mono, ANALYSIS_SR)
                beat_name = self.beat_backend.name
            except Exception as exc:  # noqa: BLE001 - fall back to the numpy tracker
                warnings.append(f"beat backend failed ({type(exc).__name__}); used numpy tracker")
                beats = None
        if beats is None or len(beats) < 4:
            beats, downbeats = NumpyBeatTracker().track(mono, ANALYSIS_SR, onset)
            beat_name = NumpyBeatTracker.name
        beats = np.asarray(beats, dtype=np.float64)
        if len(beats) < 4:  # silence / no pulse: a nominal 120 BPM grid keeps the composer working
            beats = np.arange(0, max(duration, 2.0), 0.5)
            warnings.append("no reliable beat found; using a nominal 120 BPM grid")
        ibi = np.diff(beats)
        med = np.median(ibi)
        tempo = float(60 / np.mean(ibi[np.abs(ibi - med) < 0.2 * med]))  # mean of inliers: frame quantisation averages out
        fps = ANALYSIS_SR / HOP
        beat_frames = np.clip(np.round(beats * fps).astype(int), 0, len(mix_onset) - 1)

        # ---- harmony evidence
        harmony_src = [mono_stems[k] for k in mono_stems if k not in ("vocals", "drums", "bass")]
        if harmony_src:
            n = max(len(h) for h in harmony_src)
            harm = sum(np.pad(h, (0, n - len(h))) for h in harmony_src)
        elif mono_stems:
            harm = mono - sum(_fit(mono_stems[k], len(mono)) for k in ("vocals", "drums") if k in mono_stems)
        else:
            harm = mono
        chroma_h = _fit_rows(chroma(harm, ANALYSIS_SR, 80, 2000), len(mix_onset))
        bass_src = mono_stems.get("bass", None)
        bass_notes: list[Note] = []
        if bass_src is not None and self.transcriber is not None:
            try:
                bass_notes = [n_ for n_ in futures["bass"].result() if n_.pitch < 64]
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"bass transcription failed ({type(exc).__name__}); used bass chroma")
        if bass_notes:
            chroma_b = np.zeros_like(chroma_h)
            for nt in bass_notes:
                a, b = int(nt.start * fps), max(int(nt.start * fps) + 1, int(nt.end * fps))
                chroma_b[a:b, nt.pitch % 12] += nt.velocity / 127
            bass_method = "transcribed-bass"
        else:
            chroma_b = _fit_rows(chroma(bass_src if bass_src is not None else mono, ANALYSIS_SR, 30, 260, n_fft=8192), len(mix_onset))
            bass_method = "bass-stem-chroma" if bass_src is not None else "mix-low-band-chroma"
        bc_h = beat_sync(chroma_h, beat_frames)
        bc_b = beat_sync(chroma_b, beat_frames)

        key = estimate_key(bc_h.sum(axis=0) + 0.5 * bc_b.sum(axis=0))
        chord_idx, chord_conf = decode_chords(bc_h, bc_b, key)
        beat_ends = np.concatenate([beats[1:], [duration]])

        # ---- downbeats (if the backend gave none): bar phase with the most chord changes + low onsets
        if downbeats is None or len(downbeats) == 0:
            low_on = _fit(dsp.onset_envelope(drums if drums is not None else mono, ANALYSIS_SR, HOP, fmin=30, fmax=150), len(mix_onset))
            change = np.r_[1.0, (chord_idx[1:] != chord_idx[:-1]).astype(float)]
            strength = low_on[beat_frames] / (low_on[beat_frames].mean() + 1e-9)
            phase_score = [change[p::4].mean() + 0.3 * strength[p::4].mean() for p in range(4)]
            phase = int(np.argmax(phase_score))
            downbeats = beats[phase::4]
        downbeats = np.asarray(downbeats, dtype=np.float64)

        chords = _chord_segments(chord_idx, chord_conf, beats, beat_ends)
        sections = estimate_sections(mono, bc_h, beats, downbeats, duration)
        # Local key per section: songs modulate (and a global profile then fits neither part).
        for sec in sections:
            sel = (beats >= sec["start"]) & (beats < sec["end"])
            if sel.sum() >= 8:
                local = estimate_key(bc_h[sel].sum(axis=0) + 0.5 * bc_b[sel].sum(axis=0))
                sec["key"] = local if local["correlation"] >= 0.5 else key
            else:
                sec["key"] = key

        # ---- groove
        groove = {"swing": estimate_swing(onset, beats, fps), "kick_times": [], "density": []}
        if drums is not None:
            kick_env = dsp.onset_envelope(drums, ANALYSIS_SR, HOP, fmin=30, fmax=150)
            kicks = _local_peaks(kick_env, threshold=kick_env.mean() + 2 * kick_env.std(), min_distance=int(0.12 * fps))
            groove["kick_times"] = [round(float(k / fps), 3) for k in kicks]
            drum_env = dsp.onset_envelope(drums, ANALYSIS_SR, HOP)
            hits = _local_peaks(drum_env, threshold=drum_env.mean() + 1.0 * drum_env.std(), min_distance=int(0.06 * fps)) / fps
        else:
            hits = _local_peaks(mix_onset, threshold=mix_onset.mean() + 1.0 * mix_onset.std(), min_distance=int(0.06 * fps)) / fps
        for s in sections:
            dur = max(1e-3, s["end"] - s["start"])
            groove["density"].append(round(float(np.sum((hits >= s["start"]) & (hits < s["end"])) / dur), 3))
        groove["onsets_per_beat"] = round(float(len(hits) / max(1, len(beats))), 3)

        # ---- vocals: activity + melody
        vocals = mono_stems.get("vocals")
        vocal_activity = None
        melody_notes: list[Note] | None = None
        if vocals is not None:
            vocal_activity = activity_intervals(vocals, ANALYSIS_SR)
            if melody and self.transcriber is not None:
                try:
                    melody_notes = monophonic(futures["vocals"].result() if "vocals" in futures else self.transcriber.transcribe(vocals, ANALYSIS_SR), lo=40, hi=90)
                    melody_notes = [n_ for n_ in melody_notes if _active(n_.start, n_.end, vocal_activity)]
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"melody transcription failed ({type(exc).__name__})")
            elif melody:
                warnings.append("no transcriber loaded: melody_midi omitted")
        elif melody:
            warnings.append("no vocal stem: melody_midi and vocal activity omitted")

        pool.shutdown(wait=False)
        register = register_occupancy(mono_stems, melody_notes, bass_notes)
        space = estimate_space(vocals if vocals is not None else mono, ANALYSIS_SR)

        return {
            "duration": round(duration, 4),
            "tempo": round(tempo, 2),
            "meter": 4,
            "beats": [round(float(b), 4) for b in beats],
            "downbeats": [round(float(b), 4) for b in downbeats],
            "key": key,
            "chords": chords,
            "sections": sections,
            "melody_midi": [n_.to_dict() for n_ in melody_notes] if melody_notes is not None else None,
            "vocal_activity": vocal_activity,
            "groove": groove,
            "register": register,
            "space": space,
            # Lets a later generate call (cached analysis, no stems uploaded) still sit behind the vocal.
            "levels": {"vocals_lufs": round(dsp.loudness_lufs(vocals, ANALYSIS_SR), 2) if vocals is not None else None},
            "context": {"stems_used": used, "beat_backend": beat_name, "transcriber": getattr(self.transcriber, "name", None),
                        "chord_method": f"chroma-templates+hmm ({bass_method})", "key_method": "krumhansl-kessler"},
            "warnings": warnings,
        }


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _fit_rows(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))


def _active(start: float, end: float, intervals: list[list[float]] | None) -> bool:
    if not intervals:
        return True
    return any(a < end and start < b for a, b in intervals)


# --------------------------------------------------------------------------- key & chords

def estimate_key(profile: np.ndarray) -> dict[str, Any]:
    p = np.asarray(profile, dtype=np.float64)
    if not np.any(p > 0):
        return {"tonic": "C", "pc": 0, "mode": "major", "label": "C major", "confidence": 0.0}
    scores = []
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for pc in range(12):
            scores.append((float(np.corrcoef(p, np.roll(prof, pc))[0, 1]), pc, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {"tonic": PITCH_NAMES[best[1]], "pc": best[1], "mode": best[2], "label": f"{PITCH_NAMES[best[1]]} {best[2]}",
            "confidence": round(best[0] - second[0], 4), "correlation": round(best[0], 4)}


def key_at(analysis: dict[str, Any], t: float) -> dict[str, Any]:
    for s in analysis.get("sections") or []:
        if s["start"] <= t < s["end"] and s.get("key"):
            return s["key"]
    return analysis.get("key") or {"pc": 0, "mode": "major"}


def scale_pcs(key: dict[str, Any]) -> set[int]:
    steps = MAJOR_SCALE if key.get("mode") == "major" else MINOR_SCALE
    return {(key.get("pc", 0) + s) % 12 for s in steps}


def chord_vocab() -> list[tuple[int, str]]:
    return [(r, q) for q in QUALITIES for r in range(12)]


def decode_chords(bc_h: np.ndarray, bc_b: np.ndarray, key: dict[str, Any], self_prob: float = 0.82) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi over per-beat chord scores. Index 24 is 'N' (no chord)."""
    vocab = chord_vocab()
    templates = np.zeros((len(vocab), 12), np.float32)
    for i, (r, q) in enumerate(vocab):
        for iv, w in zip(QUALITIES[q], (1.0, 0.8, 0.9)):
            # Harmonic-aware: each chord note also contributes its 3rd and 5th harmonics.
            for h_iv, h_w in ((0, 1.0), (7, 0.28), (4, 0.12)):
                templates[i, (r + iv + h_iv) % 12] += w * h_w
    templates = _norm_rows(templates)
    energy = bc_h.sum(axis=1)
    h = _norm_rows(bc_h)
    b = bc_b / (bc_b.sum(axis=1, keepdims=True) + 1e-9)
    in_key = scale_pcs(key)
    tonic = key.get("pc", 0)
    harmonic_minor_v = (tonic + 7) % 12 if key.get("mode") == "minor" else None
    diatonic = np.array([all((r + iv) % 12 in in_key for iv in QUALITIES[q]) or (r == harmonic_minor_v and q == "maj") for r, q in vocab], dtype=np.float32)
    scores = h @ templates.T + 0.45 * b[:, [r for r, _ in vocab]] + 0.06 * diatonic[None, :]
    silent = energy < 0.08 * (np.median(energy) + 1e-9)
    n_score = np.where(silent, 1.0, 0.25)[:, None]
    scores = np.concatenate([scores, n_score], axis=1)
    logp = scores / 0.05
    logp = logp - logp.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    k = scores.shape[1]
    trans = np.full((k, k), np.log((1 - self_prob) / (k - 1)))
    np.fill_diagonal(trans, np.log(self_prob))
    T = len(scores)
    delta = np.zeros((T, k))
    psi = np.zeros((T, k), dtype=int)
    delta[0] = logp[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + trans
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(k)] + logp[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    conf = np.exp(logp[np.arange(T), path])
    return path, conf


def chord_label(idx: int) -> tuple[str, int | None, str | None]:
    vocab = chord_vocab()
    if idx >= len(vocab):
        return "N", None, None
    r, q = vocab[idx]
    return f"{PITCH_NAMES[r]}:{q}", r, q


def _chord_segments(path: np.ndarray, conf: np.ndarray, beats: np.ndarray, beat_ends: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, idx in enumerate(path):
        label, root, q = chord_label(int(idx))
        if out and out[-1]["chord"] == label:
            out[-1]["end"] = round(float(beat_ends[i]), 4)
            out[-1]["_c"].append(float(conf[i]))
            continue
        out.append({"start": round(float(beats[i]) if i else 0.0, 4), "end": round(float(beat_ends[i]), 4), "chord": label, "root_pc": root, "quality": q, "_c": [float(conf[i])]})
    for seg in out:
        seg["confidence"] = round(float(np.mean(seg.pop("_c"))), 3)
    return out


# --------------------------------------------------------------------------- sections, groove, vocals

def estimate_sections(mono: np.ndarray, bc_h: np.ndarray, beats: np.ndarray, downbeats: np.ndarray, duration: float) -> list[dict[str, Any]]:
    bars = [float(d) for d in downbeats if d < duration - 0.5]
    if len(bars) < 4:
        return [{"start": 0.0, "end": round(duration, 4), "label": "A", "name": "section", "energy": 1.0, "index": 0}]
    if bars[0] > 0.3:
        bars = [0.0] + bars
    edges = bars + [duration]
    rms = dsp.rms_envelope(mono, ANALYSIS_SR, hop_s=0.05, win_s=0.1)
    rms_t = np.arange(len(rms)) * 0.05
    feats = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (beats >= a) & (beats < b)
        c = bc_h[sel].mean(axis=0) if sel.any() else np.zeros(12)
        e = rms[(rms_t >= a) & (rms_t < b)]
        feats.append(np.r_[c / (np.linalg.norm(c) + 1e-9), 2.0 * np.log10(e.mean() + 1e-6) if e.size else -6])
    F = np.array(feats)
    loud = F[:, -1]
    Fz = np.c_[F[:, :12], (loud - loud.mean()) / (loud.std() + 1e-6) * 0.35]
    S = _norm_rows(Fz) @ _norm_rows(Fz).T
    nb = len(F)
    w = 4 if nb >= 16 else 2
    kern = np.kron(np.array([[1, -1], [-1, 1]]), np.ones((w, w)))
    nov = np.zeros(nb)
    for i in range(nb):
        acc = 0.0
        for a in range(-w, w):
            for b in range(-w, w):
                ia, ib = i + a, i + b
                if 0 <= ia < nb and 0 <= ib < nb:
                    acc += kern[a + w, b + w] * S[ia, ib]
        nov[i] = acc
    max_sections = max(2, min(10, int(duration // 12)))
    bounds = [0]
    for i in pick_peaks(nov, min_distance=4, threshold=np.percentile(nov, 70)):
        if 2 <= i <= nb - 2:
            bounds.append(int(i))
        if len(bounds) >= max_sections:
            break
    bounds = sorted(set(bounds)) + [nb]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        segs.append({"start": round(edges[a], 4), "end": round(edges[b], 4), "feat": Fz[a:b].mean(axis=0), "energy": float(loud[a:b].mean())})
    # Label by similarity of mean features: A, B, C ...
    labels: list[str] = []
    protos: list[np.ndarray] = []
    for s in segs:
        f = s["feat"] / (np.linalg.norm(s["feat"]) + 1e-9)
        sims = [float(f @ p) for p in protos]
        if sims and max(sims) > 0.93:
            labels.append(chr(ord("A") + int(np.argmax(sims))))
        else:
            protos.append(f)
            labels.append(chr(ord("A") + len(protos) - 1))
    energies = np.array([s["energy"] for s in segs])
    e_rel = 10 ** ((energies - energies.max()) / 2)  # back to linear rms relative to the loudest
    counts = {l: labels.count(l) for l in labels}
    by_energy = {l: np.mean([e for e, ll in zip(energies, labels) if ll == l]) for l in counts}
    repeated = [l for l in counts if counts[l] >= 2]
    chorus = max(repeated, key=lambda l: by_energy[l]) if repeated else max(by_energy, key=by_energy.get)
    out = []
    for i, (s, lab) in enumerate(zip(segs, labels)):
        if lab == chorus:
            name = "chorus"
        elif i == 0 and e_rel[i] < 0.6 and s["end"] - s["start"] < 25:
            name = "intro"
        elif i == len(segs) - 1 and e_rel[i] < 0.6 and len(segs) > 2:
            name = "outro"
        else:
            name = "verse" if counts[lab] >= 2 or lab == "A" or lab == labels[0] else "bridge"
        out.append({"index": i, "start": s["start"], "end": s["end"], "label": lab, "name": name, "energy": round(float(e_rel[i]), 3)})
    return out


def estimate_swing(onset: np.ndarray, beats: np.ndarray, fps: float) -> float:
    """Offbeat position as a fraction of the beat (0.5 straight ... 0.67 triplet swing)."""
    cands = np.array([0.5, 0.55, 0.6, 0.64, 0.67])
    acc = np.zeros(len(cands))
    for a, b in zip(beats[:-1], beats[1:]):
        for i, c in enumerate(cands):
            f = int(round((a + c * (b - a)) * fps))
            if 0 <= f < len(onset):
                acc[i] += onset[max(0, f - 1):f + 2].max()
    if acc.max() <= 0:
        return 0.5
    best = int(np.argmax(acc))
    # Only call it swung when the late offbeat is clearly stronger than the straight one.
    return float(cands[best]) if best > 0 and acc[best] > 1.3 * acc[0] else 0.5


def activity_intervals(x: np.ndarray, sr: int, hop_s: float = 0.01) -> list[list[float]]:
    env = dsp.rms_envelope(x, sr, hop_s=hop_s, win_s=0.04)
    edb = 20 * np.log10(env + 1e-9)
    thr = max(-50.0, float(np.percentile(edb, 99)) - 30.0)
    on = edb > thr
    # Hangover 150 ms, then drop runs shorter than 200 ms.
    hang = int(0.15 / hop_s)
    on = np.convolve(on.astype(float), np.ones(hang), mode="full")[:len(on)] > 0
    out: list[list[float]] = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i) * hop_s >= 0.2:
                if out and i * hop_s - out[-1][1] < 0.25:
                    out[-1][1] = round(j * hop_s, 3)
                else:
                    out.append([round(i * hop_s, 3), round(j * hop_s, 3)])
            i = j
        else:
            i += 1
    return out


def monophonic(notes: list[Note], lo: int = 0, hi: int = 127, min_dur: float = 0.08) -> list[Note]:
    """Keep the strongest note at any time (skyline by velocity, then pitch), trimmed to not overlap."""
    notes = sorted((n for n in notes if lo <= n.pitch <= hi and n.end - n.start >= min_dur), key=lambda n: n.start)
    out: list[Note] = []
    for n in notes:
        if out and n.start < out[-1].end:
            prev = out[-1]
            if (n.velocity, n.pitch) > (prev.velocity, prev.pitch):
                prev.end = n.start
                if prev.end - prev.start < min_dur:
                    out.pop()
                out.append(Note(n.start, n.end, n.pitch, n.velocity))
            continue
        out.append(Note(n.start, n.end, n.pitch, n.velocity))
    return out


def register_occupancy(stems: dict[str, np.ndarray], melody: list[Note] | None, bass_notes: list[Note]) -> dict[str, Any]:
    """Pitch-range occupancy per stem (MIDI), so a new part can avoid crowded registers."""
    out: dict[str, Any] = {}
    if melody:
        p = np.array([n.pitch for n in melody])
        out["vocals"] = {"low": int(np.percentile(p, 10)), "high": int(np.percentile(p, 90)), "median": int(np.median(p)), "source": "melody"}
    if bass_notes:
        p = np.array([n.pitch for n in bass_notes])
        out["bass"] = {"low": int(np.percentile(p, 10)), "high": int(np.percentile(p, 90)), "median": int(np.median(p)), "source": "transcription"}
    n_fft = 8192
    freqs = np.fft.rfftfreq(n_fft, 1 / ANALYSIS_SR)
    ok = (freqs > 30) & (freqs < 4200)
    midi = np.round(dsp.hz_to_midi(freqs[ok])).astype(int)
    for name, x in stems.items():
        if name == "drums" or name in out:
            continue
        mag = dsp.stft_mag(x[: ANALYSIS_SR * 240], n_fft, 4096)
        if not len(mag):
            continue
        spec = (mag[:, ok] ** 2).mean(axis=0)
        # Spectral peak picking per frame is costly; a long-term spectrum weighted towards
        # fundamentals (energy / harmonic number proxy: 1/f tilt) is enough for range estimates.
        hist = np.bincount(midi - midi.min(), weights=spec * (1 / np.maximum(freqs[ok], 30)), minlength=midi.max() - midi.min() + 1)
        if hist.sum() <= 0:
            continue
        cdf = np.cumsum(hist) / hist.sum()
        lo_m = int(np.searchsorted(cdf, 0.15) + midi.min())
        hi_m = int(np.searchsorted(cdf, 0.85) + midi.min())
        out[name] = {"low": lo_m, "high": hi_m, "median": int(np.searchsorted(cdf, 0.5) + midi.min()), "source": "spectrum"}
    return out


def estimate_space(x: np.ndarray, sr: int) -> dict[str, Any]:
    """RT60 guess from decay slopes after note offsets (dB/s of the RMS envelope)."""
    hop_s = 0.01
    env = dsp.rms_envelope(x, sr, hop_s=hop_s, win_s=0.03)
    edb = 20 * np.log10(env + 1e-9)
    peak_floor = np.percentile(edb, 95) - 25
    slopes = []
    i = 5
    n = len(edb)
    while i < n - 60:
        if edb[i] > peak_floor and edb[i] >= edb[i - 5:i + 5].max():
            seg = edb[i:i + 60]
            if seg[0] - seg.min() > 15:
                j = int(np.argmax(seg < seg[0] - 15))
                if j >= 4:
                    t = np.arange(j) * hop_s
                    slope = np.polyfit(t, seg[:j], 1)[0]
                    if slope < -5:
                        slopes.append(slope)
                i += 30
                continue
        i += 1
    if len(slopes) < 3:
        rt60 = 0.8
        method = "default"
    else:
        rt60 = float(np.clip(60 / abs(np.median(slopes)), 0.25, 3.0))
        method = f"decay-slopes(n={len(slopes)})"
    wet = float(np.clip(0.12 + 0.12 * rt60, 0.12, 0.45))
    return {"rt60": round(rt60, 3), "wet": round(wet, 3), "predelay_ms": 18.0, "method": method}


def analysis_dataclass_note(d: dict[str, Any]) -> Note:
    return Note(float(d["start"]), float(d["end"]), int(d["pitch"]), int(d.get("velocity", 80)))


__all__ = ["MusicAnalyzer", "Note", "NumpyBeatTracker", "estimate_key", "decode_chords", "scale_pcs", "monophonic", "asdict"]
