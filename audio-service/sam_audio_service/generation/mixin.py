"""Mix-in: make a rendered part sit in the song.

1. Role-dependent high-pass (keeps pads and leads out of the bass/kick).
2. Dynamic EQ carve: up to ``carve_db`` cut in the 1.5-5 kHz presence band while the vocal sings.
3. Reverb matched to the song's estimated space (synthetic stereo IR: early reflections + decaying
   noise, high frequencies decay faster; RT60/wet/pre-delay from the analysis).
4. Loudness relative to the song (BS.1770-style gated LUFS), per-role offset + user offset.
5. Light ducking under the vocal (attack 30 ms / release 300 ms follower).
6. Window fades, peak guard (part <= -1 dBFS; warns if song + part would clip).

The output is a new layer the same length as the song.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import dsp

ROLE_LEVEL_LU = {"pad": -9.0, "counter_melody": -8.0, "lead": -6.5, "ostinato": -9.5, "bassline": -6.0, "riff": -8.0, "stabs": -8.0, "chords": -7.5}
ROLE_HPF = {"bassline": 28.0, "pad": 110.0, "chords": 70.0, "stabs": 120.0, "ostinato": 140.0, "counter_melody": 150.0, "lead": 160.0, "riff": 90.0}


def synth_ir(sr: int, rt60: float, predelay_ms: float = 18.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * min(4.0, rt60 * 1.2 + 0.1))
    t = np.arange(n) / sr
    ir = np.zeros((2, n), np.float32)
    for c in range(2):
        noise = rng.standard_normal(n).astype(np.float32)
        low = dsp.fft_eq(noise, sr, lambda f: dsp.lowpass_gain(f, 3500, 1))
        high = noise - low
        decay_lo = np.exp(-6.91 * t / rt60)
        decay_hi = np.exp(-6.91 * t / max(0.15, rt60 * 0.45))
        tail = low * decay_lo + 0.6 * high * decay_hi
        fade_in = np.clip(t / 0.02, 0, 1)  # the diffuse tail builds up after the early reflections
        ir[c] = tail * fade_in * 0.6
        for ms, g in ((7, 0.5), (13, 0.38), (21, 0.3), (29, 0.22), (37, 0.17)):
            k = int((ms + 3 * c) * sr / 1000)
            if k < n:
                ir[c, k] += g * (1 if (ms + c) % 2 else -1)
    pre = int(predelay_ms * sr / 1000)
    ir = np.pad(ir, ((0, 0), (pre, 0)))
    return ir / (np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True)) + 1e-9)


def _vocal_presence(vocals: np.ndarray | None, sr: int, n: int, hop: int) -> np.ndarray | None:
    if vocals is None:
        return None
    env = dsp.rms_envelope(vocals, sr, hop_s=hop / sr, win_s=0.04)
    edb = 20 * np.log10(env + 1e-9)
    top = np.percentile(edb, 98)
    pres = np.clip((edb - (top - 30)) / 24, 0, 1)
    pres = dsp.attack_release(pres.astype(np.float32), sr / hop, 0.03, 0.3)
    frames = n // hop + 2
    return pres[:frames] if len(pres) >= frames else np.pad(pres, (0, frames - len(pres)))


def mix_in(part: np.ndarray, sr: int, song: np.ndarray, vocals: np.ndarray | None, analysis: dict[str, Any], role: str,
           windows: list[tuple[float, float]], *, level_db: float = 0.0, duck_db: float = 3.0, carve_db: float = 4.0,
           reverb: bool = True, has_bass_stem: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    n = song.shape[-1]
    part = part[:, :n] if part.shape[1] >= n else np.pad(part, ((0, 0), (0, n - part.shape[1])))
    report: dict[str, Any] = {}
    # 1. high-pass + gentle air roll-off
    hpf = ROLE_HPF.get(role, 100.0)
    part = dsp.fft_eq(part, sr, lambda f: dsp.highpass_gain(f, hpf, 2) * dsp.lowpass_gain(f, 15000, 1))
    # 2. dynamic EQ carve around the vocal
    hop, n_fft = 512, 2048
    pres = _vocal_presence(vocals, sr, n, hop)
    if pres is not None and carve_db > 0 and role != "bassline":
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)
        band = np.exp(-0.5 * (np.log2(np.maximum(freqs, 1) / 2800) / 0.7) ** 2)  # centred 2.8 kHz, ~1.6 oct wide
        out = np.empty_like(part)
        for c in range(part.shape[0]):
            spec = dsp.stft(part[c], n_fft, hop)
            p = pres[: len(spec)]
            if len(p) < len(spec):
                p = np.pad(p, (0, len(spec) - len(p)))
            gain = 10 ** (-carve_db * p[:, None] * band[None, :] / 20)
            out[c] = dsp.istft(spec * gain, n, n_fft, hop)
        part = out
        report["eq_carve"] = {"band_hz": [1500, 5000], "max_cut_db": carve_db, "vocal_active_fraction": round(float((pres > 0.5).mean()), 3)}
    # 3. reverb matched to the song
    space = analysis.get("space") or {"rt60": 0.8, "wet": 0.2, "predelay_ms": 18.0}
    if reverb:
        rt60 = float(space.get("rt60", 0.8))
        wet = float(space.get("wet", 0.2)) * (1.25 if role == "pad" else 0.4 if role == "bassline" else 1.0)
        ir = synth_ir(sr, rt60, float(space.get("predelay_ms", 18.0)))
        wet_sig = np.stack([dsp.fft_convolve(part[c], ir[c])[:n] for c in range(2)])
        dry_rms, wet_rms = np.sqrt(np.mean(part ** 2) + 1e-12), np.sqrt(np.mean(wet_sig ** 2) + 1e-12)
        part = np.sqrt(1 - wet) * part + np.sqrt(wet) * wet_sig * (dry_rms / wet_rms)
        report["reverb"] = {"rt60": round(rt60, 3), "wet": round(wet, 3), "method": space.get("method", "default")}
    # 5. ducking under the vocal (before loudness, so the level target includes it)
    if pres is not None and duck_db > 0:
        depth = duck_db * (0.35 if role == "bassline" else 1.0)
        g = 10 ** (-depth * pres / 20)
        g_s = np.interp(np.arange(n), np.arange(len(g)) * hop, g).astype(np.float32)
        part = part * g_s
        report["ducking"] = {"max_db": round(depth, 2), "attack_ms": 30, "release_ms": 300}
    # 6. window fades (keep a short tail for releases / reverb)
    mask = np.zeros(n, np.float32)
    for a, b in windows:
        s, e = int(a * sr), min(n, int((b + 1.2) * sr))
        mask[s:e] = 1
    fade = int(0.05 * sr)
    mask = np.convolve(mask, np.ones(fade) / fade, mode="same").astype(np.float32)
    part = part * mask
    # 4. loudness relative to the song
    song_lufs = dsp.loudness_lufs(song, sr)
    part_lufs = dsp.loudness_lufs(part, sr)
    offset = ROLE_LEVEL_LU.get(role, -8.0) - (3.0 if role == "bassline" and has_bass_stem else 0.0) + level_db
    target = song_lufs + offset
    gain_db = float(np.clip(target - part_lufs, -40, 30)) if part_lufs > -69 else 0.0
    part = part * 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(part))) if part.size else 0.0
    if peak > 10 ** (-1 / 20):
        part *= 10 ** (-1 / 20) / peak
        gain_db += dsp.db(10 ** (-1 / 20) / peak)
    final_lufs = dsp.loudness_lufs(part, sr)
    song2 = song if song.shape[0] == 2 else np.repeat(song[:1], 2, axis=0)
    mix_peak = float(np.max(np.abs(song2[:, :n] + part)))
    report["loudness"] = {"song_lufs": round(song_lufs, 2), "part_lufs": round(final_lufs, 2), "relative_lu": round(final_lufs - song_lufs, 2),
                          "target_relative_lu": round(offset, 2), "gain_db": round(gain_db, 2)}
    report["peak"] = {"part_dbfs": round(dsp.db(np.max(np.abs(part)) if part.size else 0), 2), "mix_dbfs": round(dsp.db(mix_peak), 2)}
    warnings = []
    if mix_peak > 1.0:
        warnings.append(f"song + part peaks at {dsp.db(mix_peak):.1f} dBFS; lower the song or apply a limiter on export")
    report["warnings"] = warnings
    return part.astype(np.float32), report
