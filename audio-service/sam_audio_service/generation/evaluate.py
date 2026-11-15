"""Objective checks for a generated part. There is no ground truth for "sounds good", so these are
sanity metrics that catch a part that ignores the song:

note level (symbolic engines): chord-tone ratio, in-key ratio, onset alignment to the beat grid,
vocal clashes (minor 2nd / major 7th / tritone against the sung note) and vocal masking (within a
minor third of the sung note), register overlap with the vocal range.

audio level (every engine, incl. MusicGen): chroma energy on the detected chord tones (chance for a
triad = 0.25) and on the key's scale (chance = 0.58), onset alignment of the audio's onsets to the
grid, loudness relative to the song.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import music_analysis as ma
from . import dsp
from .composer import PNote


def _grid(analysis: dict[str, Any], k: int = 4) -> np.ndarray:
    beats = np.asarray(analysis["beats"], dtype=np.float64)
    if len(beats) < 2:
        return beats
    period = np.median(np.diff(beats))
    b = np.r_[beats, beats[-1] + period]
    pts = [b[i] + j * (b[i + 1] - b[i]) / k for i in range(len(b) - 1) for j in range(k)]
    swing = float((analysis.get("groove") or {}).get("swing", 0.5))
    if swing > 0.52:
        pts += [b[i] + swing * (b[i + 1] - b[i]) for i in range(len(b) - 1)]
    return np.sort(np.array(pts))


def _chord_pcs(analysis: dict[str, Any], t: float) -> set[int] | None:
    for c in analysis.get("chords", []):
        if c["start"] <= t < c["end"]:
            if c.get("root_pc") is None:
                return None
            return {(c["root_pc"] + i) % 12 for i in ma.QUALITIES.get(c.get("quality") or "maj", (0, 4, 7))}
    return None


def note_checks(notes: list[PNote], analysis: dict[str, Any]) -> dict[str, Any]:
    if not notes:
        return {"notes": 0}
    ct_w = ct_tot = key_w = 0.0
    for n in notes:
        d = n.end - n.start
        pcs = _chord_pcs(analysis, n.start + min(0.08, d / 2))  # past humanize jitter at chord changes
        if pcs is not None:
            ct_tot += d
            ct_w += d * (n.pitch % 12 in pcs)
        key_w += d * (n.pitch % 12 in ma.scale_pcs(ma.key_at(analysis, n.start)))
    total = sum(n.end - n.start for n in notes)
    grid = _grid(analysis)
    devs = np.array([np.min(np.abs(grid - n.start)) for n in notes]) * 1000 if grid.size else np.zeros(0)
    mel = analysis.get("melody_midi") or []
    clash = mask = overlap_t = 0.0
    for n in notes:
        for m in mel:
            o = min(n.end, m["end"]) - max(n.start, m["start"])
            if o <= 0:
                continue
            overlap_t += o
            iv = abs(n.pitch - m["pitch"])
            if iv % 12 in (1, 11, 6) and iv < 25:
                clash += o
            if iv < 3:
                mask += o
    reg = (analysis.get("register") or {}).get("vocals")
    in_vocal_range = float(np.mean([reg["low"] <= n.pitch <= reg["high"] for n in notes])) if reg else None
    return {
        "notes": len(notes),
        "chord_tone_ratio": round(ct_w / ct_tot, 3) if ct_tot else None,
        "in_key_ratio": round(key_w / total, 3) if total else None,
        "onset_dev_ms_median": round(float(np.median(devs)), 1) if devs.size else None,
        "onsets_within_30ms": round(float(np.mean(devs <= 30)), 3) if devs.size else None,
        "vocal_overlap_s": round(overlap_t, 2),
        "vocal_clash_ratio": round(clash / overlap_t, 3) if overlap_t else 0.0,
        "vocal_masking_ratio": round(mask / overlap_t, 3) if overlap_t else 0.0,
        "in_vocal_range_ratio": round(in_vocal_range, 3) if in_vocal_range is not None else None,
        "pitch_range": [int(min(n.pitch for n in notes)), int(max(n.pitch for n in notes))],
    }


def audio_checks(part: np.ndarray, sr: int, analysis: dict[str, Any], song: np.ndarray | None = None) -> dict[str, Any]:
    x = dsp.resample(dsp.as_mono(part), sr, ma.ANALYSIS_SR)
    fps = ma.ANALYSIS_SR / ma.HOP
    c = ma.chroma(x, ma.ANALYSIS_SR, 60, 2500)
    # Keep the harmonic peaks per frame (suppress the broadband floor) so chance stays ~3/12.
    c = np.maximum(c - np.median(c, axis=1, keepdims=True), 0)
    energy = c.sum(axis=1)
    active = energy > 0.05 * (np.percentile(energy, 95) + 1e-9)
    ct = tot = key = 0.0
    for i in np.where(active)[0]:
        t = i / fps
        pcs = _chord_pcs(analysis, t)
        if pcs is None:
            continue
        ct += c[i, list(pcs)].sum()
        key += c[i, list(ma.scale_pcs(ma.key_at(analysis, t)))].sum()
        tot += c[i].sum()
    onset = dsp.onset_envelope(x, ma.ANALYSIS_SR, ma.HOP)
    peaks = ma._local_peaks(onset, threshold=onset.mean() + 1.5 * onset.std(), min_distance=int(0.08 * fps)) / fps
    grid = _grid(analysis)
    devs = np.array([np.min(np.abs(grid - p)) for p in peaks]) * 1000 if peaks.size and grid.size else np.zeros(0)
    out = {
        "active_seconds": round(float(active.sum() / fps), 2),
        "chroma_chord_tone_ratio": round(ct / tot, 3) if tot else None,
        "chroma_in_key_ratio": round(key / tot, 3) if tot else None,
        "audio_onsets": int(peaks.size),
        "audio_onset_dev_ms_median": round(float(np.median(devs)), 1) if devs.size else None,
        "audio_onsets_within_30ms": round(float(np.mean(devs <= 30)), 3) if devs.size else None,
    }
    if song is not None:
        out["relative_lu"] = round(dsp.loudness_lufs(part, sr) - dsp.loudness_lufs(song, sr), 2)
    return out
