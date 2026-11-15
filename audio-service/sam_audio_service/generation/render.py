"""Renderers: notes -> stereo audio at the song's sample rate.

* ``SampleRenderer`` - VSCO-2 Community Edition (CC0) multisamples played by a small numpy sampler
  (nearest-zone pick by measured pitch + velocity layer, round robin, resample-to-pitch, crossfade
  looping for sustains, attack/release envelopes). Reads ``manifest.json`` written by
  ``scripts/build_vsco_manifest.py`` (pitch of every sample is *measured*, not parsed).
* ``FluidSynthRenderer`` - any General MIDI program through the FluidSynth CLI (LGPL, run as a
  separate process) and FluidR3_GM.sf2 (MIT). Wide coverage: guitars, brass, choir, organ ...
* ``SynthRenderer`` - a licence-free numpy subtractive synth (polyBLEP oscillators, per-note filter
  envelopes, unison detune, chorus, tempo-synced delay): pad, arp80s, lead, bass, pluck.

``RendererSet.render(spec, notes, ...)`` dispatches on ``spec.renderer`` (``vsco:<preset>``,
``gm:<program>``, ``synth:<preset>``) and falls back (``spec.fallback_renderer``, then the synth)
when a backend is not installed, recording what was actually used.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import dsp
from .composer import PNote
from .midi import write_midi

log = logging.getLogger("sam_audio_service")


class RenderUnavailable(RuntimeError):
    pass


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """PCM 16/24/32-bit or float32 WAV -> float32 (channels, n). No external dependencies."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"not a WAV file: {path}")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError(f"WAV without fmt/data: {path}")
    tag, ch, sr, _, _, bits = fmt
    if tag == 0xFFFE and len(body) >= 0:  # WAVE_FORMAT_EXTENSIBLE: sub-format GUID decides int vs float
        tag = 3 if fmt_ext_float(data) else 1
    if bits == 16:
        x = np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768
    elif bits == 24:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8).astype(np.float32) / 8388608
    elif bits == 32 and tag == 3:
        x = np.frombuffer(raw[: len(raw) // 4 * 4], "<f4").astype(np.float32)
    elif bits == 32:
        x = np.frombuffer(raw[: len(raw) // 4 * 4], "<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported WAV bit depth {bits}")
    x = x[: len(x) // ch * ch].reshape(-1, ch).T
    return np.ascontiguousarray(x), sr


def fmt_ext_float(data: bytes) -> bool:
    i = data.find(b"fmt ")
    return i >= 0 and data[i + 8 + 24:i + 8 + 26] == b"\x03\x00"


def _env(n: int, sr: int, attack: float, release: float, total: int) -> np.ndarray:
    """Linear-in-dB-ish envelope: attack ramp, hold, cosine release starting at sample n."""
    e = np.ones(total, np.float32)
    a = min(total, max(1, int(attack * sr)))
    e[:a] = np.linspace(0, 1, a) ** 1.5
    r = max(1, total - n)
    if total > n:
        e[n:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, r)))).astype(np.float32)
    return e


def _vel_gain(v: int) -> float:
    return float((max(1, v) / 127) ** 1.6)


# --------------------------------------------------------------------------- VSCO sampler

COMPOSITE = {
    # strings ensemble: voices split across sections by pitch
    "strings": [(0, 52, "cellos"), (53, 61, "violas"), (62, 127, "violins")],
    "strings_spic": [(0, 54, "cellos_spic"), (55, 127, "violins_spic")],
}
# Section presets rendered as an ensemble: 3 layers, detuned, time-offset and spread across the stereo field.
ENSEMBLE = {"violins", "violas", "cellos", "violins_spic", "cellos_spic", "violins_pizz", "horn"}
ARTICULATION_PRESET = {("violins", "spiccato"): "violins_spic", ("violins", "pizzicato"): "violins_pizz", ("cellos", "spiccato"): "cellos_spic",
                       ("strings", "spiccato"): "strings_spic", ("strings", "pizzicato"): "violins_pizz", ("solo_violin", "pizzicato"): "violins_pizz"}


class SampleRenderer:
    name = "vsco2-ce"
    license = "CC0-1.0 (VSCO-2 Community Edition samples)"

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        self._manifest: dict[str, Any] | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._rr: dict[str, int] = {}

    @property
    def available(self) -> bool:
        return bool(self.root and (self.root / "manifest.json").exists())

    def presets(self) -> list[str]:
        return sorted(self.manifest().get("presets", {})) + sorted(COMPOSITE) if self.available else []

    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            if not self.available:
                raise RenderUnavailable("VSCO-2 sample manifest not found (AUDIO_GEN_SAMPLES_DIR)")
            self._manifest = json.loads((self.root / "manifest.json").read_text())
        return self._manifest

    def has(self, preset: str) -> bool:
        if not self.available:
            return False
        if preset in COMPOSITE:
            return all(p in self.manifest()["presets"] for _, _, p in COMPOSITE[preset])
        return preset in self.manifest()["presets"]

    def _sample(self, rel: str, sr: int) -> np.ndarray:
        key = f"{rel}@{sr}"
        with self._lock:
            if key not in self._cache:
                x, src_sr = read_wav(self.root / rel)
                if x.shape[0] == 1:
                    x = np.repeat(x, 2, axis=0)
                x = x[:2]
                self._cache[key] = dsp.resample(x, src_sr, sr)
            return self._cache[key]

    def render(self, preset: str, notes: list[PNote], n_out: int, sr: int, articulation: str = "normal") -> np.ndarray:
        preset = ARTICULATION_PRESET.get((preset, articulation), preset)
        out = np.zeros((2, n_out), np.float32)
        if preset in COMPOSITE:
            for lo, hi, sub in COMPOSITE[preset]:
                sel = [n for n in notes if lo <= n.pitch <= hi]
                if sel:
                    out += self.render(sub, sel, n_out, sr, "normal") * (0.85 if sub == "cellos" else 1.0)
            return out
        info = self.manifest()["presets"][preset]
        zones = info["zones"]
        layers = max(z["layer"] for z in zones) + 1
        sustain = info.get("sustain", True)
        attack = float(info.get("attack", 0.0))
        release = float(info.get("release", 0.3))
        gain = float(info.get("gain", 1.0))
        ensemble = preset in ENSEMBLE
        n_layers = 3 if ensemble else 1
        rng = np.random.default_rng(len(notes) * 7919 + n_out % 9973)
        last_file: dict[int, str] = {}
        for n in notes:
            want_layer = int(round((n.velocity / 127) * (layers - 1)))
            def cost(z):
                return abs(z["pitch"] - n.pitch) + 1.5 * abs(z["layer"] - want_layer)
            ranked = sorted(zones, key=cost)
            dur = max(0.03, n.end - n.start)
            s0 = int(n.start * sr)
            if s0 >= n_out:
                continue
            long_note = sustain and dur > 0.6
            # Bow attack: softer notes speak slower (40-150 ms); short notes keep a crisp start.
            att = (0.04 + 0.11 * (1 - n.velocity / 127)) if (sustain and n.articulation == "sustain") else min(attack, 0.012) if not sustain else 0.035
            rel_s = max(release, 0.35) if sustain else release
            for k in range(n_layers):
                # Each layer uses a different sample (round robin / neighbouring zone), never the previous one.
                pool = ranked[: max(1, min(len(ranked), 3 if ensemble else 1))]
                rr_key = f"{preset}:{n.pitch}:{want_layer}"
                idx = (self._rr.get(rr_key, 0) + k) % len(pool)
                zone = pool[idx]
                if last_file.get(k) == zone["file"] and len(pool) > 1:
                    zone = pool[(idx + 1) % len(pool)]
                last_file[k] = zone["file"]
                src = self._sample(zone["file"], sr)
                cents = 0.0 if k == 0 else float(rng.choice([-1, 1]) * rng.uniform(4, 12))
                offset = 0 if k == 0 else int(rng.uniform(0.008, 0.025) * sr)
                hold = int(dur * sr)
                rel = int(rel_s * sr)
                base_ratio = 2 ** ((n.pitch - zone["pitch"] + cents / 100) / 12)
                total = hold + rel if sustain else min(int(len(src[0]) / base_ratio), hold + rel)
                t = np.arange(total) / sr
                ratio = np.full(total, base_ratio)
                if long_note:  # delayed vibrato on top of the sample's own: ~5.5 Hz, 8-18 cents, from ~200 ms
                    depth = rng.uniform(8, 18) * np.clip((t - 0.2) / 0.35, 0, 1)
                    ratio = ratio * 2 ** (depth * np.sin(2 * np.pi * rng.uniform(5.1, 5.9) * t + rng.uniform(0, 6.28)) / 1200)
                pos = np.concatenate([[0.0], np.cumsum(ratio[:-1])])
                need = int(pos[-1]) + 4
                if need > src.shape[1]:
                    src = _extend_loop(src, need, sr, zone.get("loop"))
                seg = np.stack([np.interp(pos, np.arange(src.shape[1]), c) for c in src])
                env = _env(hold if sustain else min(hold, total), sr, att, rel_s, total)
                if long_note:  # gentle swell into the middle of the note, easing off before the release
                    env *= (0.82 + 0.18 * np.sin(np.pi * np.clip(t / max(dur, 1e-3), 0, 1)) ** 0.8).astype(np.float32)
                g = _vel_gain(n.velocity) * gain / np.sqrt(n_layers)
                if ensemble:
                    pan = (-0.65, 0.65, 0.0, -0.3)[k % 4] if n_layers > 1 else 0.0
                    seg = np.stack([seg[0] * np.sqrt(0.5 * (1 - pan)) * 1.41, seg[1] * np.sqrt(0.5 * (1 + pan)) * 1.41])
                s = s0 + offset
                if s >= n_out:
                    continue
                e = min(n_out, s + total)
                out[:, s:e] += seg[:, : e - s] * env[: e - s] * g
            self._rr[f"{preset}:{n.pitch}:{want_layer}"] = self._rr.get(f"{preset}:{n.pitch}:{want_layer}", 0) + 1
        return out


def _extend_loop(src: np.ndarray, need: int, sr: int, loop: list[float] | None) -> np.ndarray:
    """Crossfade-loop the steady part of a sustain sample until it is ``need`` samples long."""
    n = src.shape[1]
    a, b = (int(loop[0] * n), int(loop[1] * n)) if loop else (int(0.35 * n), int(0.88 * n))
    xf = min(int(0.12 * sr), (b - a) // 3)
    if b - a < 3 * xf or xf < 16:
        return np.pad(src, ((0, 0), (0, need - n)))
    body = src[:, a:b]
    fade_in = np.linspace(0, 1, xf, dtype=np.float32) ** 0.5
    fade_out = np.linspace(1, 0, xf, dtype=np.float32) ** 0.5
    out = src[:, :b].copy()
    while out.shape[1] < need:
        tail = out[:, -xf:] * fade_out + body[:, :xf] * fade_in
        out = np.concatenate([out[:, :-xf], tail, body[:, xf:]], axis=1)
    return out


# --------------------------------------------------------------------------- FluidSynth (GM)

class FluidSynthRenderer:
    name = "fluidsynth-gm"

    def __init__(self, soundfont: str | None = None, binary: str | None = None):
        self.binary = binary or shutil.which("fluidsynth")
        candidates = [soundfont] if soundfont else ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2"]
        self.soundfont = next((c for c in candidates if c and os.path.exists(c)), None)
        self.license = "FluidR3_GM.sf2 (MIT); FluidSynth (LGPL-2.1, separate process)"

    @property
    def available(self) -> bool:
        return bool(self.binary and self.soundfont)

    def render(self, program: int, notes: list[PNote], n_out: int, sr: int, tempo: float = 120.0, channel: int = 0, timeout: int = 300) -> np.ndarray:
        if not self.available:
            raise RenderUnavailable("fluidsynth binary or GM soundfont not found")
        with tempfile.TemporaryDirectory(prefix="gen-fs-") as tmp:
            mid = Path(tmp) / "part.mid"
            wav = Path(tmp) / "part.wav"
            mid.write_bytes(write_midi(notes, tempo=tempo, program=program, channel=channel))
            cmd = [self.binary, "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.5", "-r", str(sr), "-F", str(wav), "-T", "wav", "-O", "float", self.soundfont, str(mid)]
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            x, got_sr = read_wav(wav)
        if got_sr != sr:
            x = dsp.resample(x, got_sr, sr)
        if x.shape[0] == 1:
            x = np.repeat(x, 2, axis=0)
        x = x[:2]
        return x[:, :n_out] if x.shape[1] >= n_out else np.pad(x, ((0, 0), (0, n_out - x.shape[1])))


# --------------------------------------------------------------------------- built-in synth

SYNTH_PRESETS: dict[str, dict[str, Any]] = {
    "pad": {"wave": "saw", "detune": [-14, -6, 0, 7, 13], "cutoff": 1400, "env_amt": 900, "f_attack": 1.2, "f_decay": 2.0, "res_db": 2,
            "a": 0.7, "d": 0.8, "s": 0.85, "r": 1.4, "chorus": 0.6, "delay": 0.0, "vib": 0.0, "gain": 0.5},
    "arp80s": {"wave": "saw+square", "detune": [-7, 7], "cutoff": 700, "env_amt": 4200, "f_attack": 0.002, "f_decay": 0.16, "res_db": 5,
               "a": 0.002, "d": 0.22, "s": 0.35, "r": 0.12, "chorus": 0.3, "delay": 0.32, "vib": 0.0, "gain": 0.6},
    "lead": {"wave": "saw", "detune": [-5, 5], "cutoff": 2600, "env_amt": 1800, "f_attack": 0.01, "f_decay": 0.4, "res_db": 3,
             "a": 0.015, "d": 0.3, "s": 0.8, "r": 0.18, "chorus": 0.2, "delay": 0.18, "vib": 14.0, "gain": 0.55},
    "bass": {"wave": "saw+sub", "detune": [0], "cutoff": 380, "env_amt": 1100, "f_attack": 0.002, "f_decay": 0.14, "res_db": 4,
             "a": 0.003, "d": 0.3, "s": 0.7, "r": 0.06, "chorus": 0.0, "delay": 0.0, "vib": 0.0, "gain": 0.8},
    "pluck": {"wave": "saw", "detune": [-4, 4], "cutoff": 900, "env_amt": 3800, "f_attack": 0.001, "f_decay": 0.11, "res_db": 3,
              "a": 0.001, "d": 0.35, "s": 0.0, "r": 0.2, "chorus": 0.35, "delay": 0.25, "vib": 0.0, "gain": 0.6},
}


def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1
    m2 = t > 1 - dt
    x = (t[m2] - 1) / dt[m2]
    out[m2] = x * x + x + x + 1
    return out


def _osc(freq: np.ndarray, sr: int, wave: str, phase0: float) -> np.ndarray:
    dt = freq / sr
    phase = (phase0 + np.cumsum(dt)) % 1.0
    saw = 2 * phase - 1 - _polyblep(phase, dt)
    if wave == "saw":
        return saw
    sq = np.where(phase < 0.5, 1.0, -1.0) + _polyblep(phase, dt) - _polyblep((phase + 0.5) % 1.0, dt)
    if wave == "square":
        return sq
    if wave == "saw+square":
        return 0.6 * saw + 0.4 * sq
    if wave == "saw+sub":
        # Sine reinforcement of the fundamental (bass notes are already low; an octave-down sub is rumble).
        return 0.65 * saw + 0.55 * np.sin(2 * np.pi * phase)
    return saw


def _adsr(n_hold: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    na, nd, nr = max(1, int(a * sr)), max(1, int(d * sr)), max(1, int(r * sr))
    env = np.concatenate([np.linspace(0, 1, na), 1 - (1 - s) * (1 - np.exp(-5 * np.arange(nd) / nd)) / (1 - np.exp(-5)), ])
    if len(env) >= n_hold:
        env = env[:n_hold]
    else:
        env = np.r_[env, np.full(n_hold - len(env), s)]
    level = env[-1] if len(env) else 0.0
    return np.r_[env, level * np.exp(-5 * np.arange(nr) / nr)].astype(np.float32)


def _filter_sweep(x: np.ndarray, sr: int, cutoff: np.ndarray, res_db: float, n_fft: int = 1024) -> np.ndarray:
    """Time-varying 24 dB/oct lowpass with a resonant bump, applied in the STFT domain."""
    hop = n_fft // 4
    spec = dsp.stft(x, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    idx = np.clip(np.arange(len(spec)) * hop, 0, len(cutoff) - 1)
    fc = np.maximum(cutoff[idx], 40.0)[:, None]
    g = 1 / np.sqrt(1 + (freqs[None, :] / fc) ** 8)
    g *= 1 + (10 ** (res_db / 20) - 1) * np.exp(-0.5 * (np.log2(np.maximum(freqs[None, :], 1) / fc) / 0.25) ** 2)
    return dsp.istft(spec * g, len(x), n_fft, hop)


class SynthRenderer:
    name = "numpy-synth"
    license = "built-in (no third-party assets)"

    def render(self, preset: str, notes: list[PNote], n_out: int, sr: int, tempo: float = 120.0, seed: int = 0) -> np.ndarray:
        p = SYNTH_PRESETS.get(preset, SYNTH_PRESETS["pad"])
        rng = np.random.default_rng(seed)
        out = np.zeros((2, n_out), np.float32)
        nv = len(p["detune"])
        pans = np.linspace(-0.7, 0.7, nv) if nv > 1 else np.zeros(1)
        for n in notes:
            hold = max(1, int((n.end - n.start) * sr))
            env = _adsr(hold, sr, p["a"], p["d"], p["s"], p["r"])
            total = len(env)
            t = np.arange(total) / sr
            f0 = float(dsp.midi_to_hz(n.pitch))
            vib = 1.0
            if p["vib"]:
                depth = p["vib"] / 1200 * np.clip((t - 0.25) / 0.3, 0, 1)
                vib = 2 ** (depth * np.sin(2 * np.pi * 5.3 * t))
            vel = _vel_gain(n.velocity)
            f_env = np.where(t < p["f_attack"], t / max(p["f_attack"], 1e-4), np.exp(-(t - p["f_attack"]) / max(p["f_decay"], 1e-3)))
            cutoff = p["cutoff"] * (0.7 + 0.6 * vel) + p["env_amt"] * vel * f_env
            cutoff = np.minimum(cutoff, 0.45 * sr)
            voice = np.zeros((2, total), np.float32)
            for vi, (cents, pan) in enumerate(zip(p["detune"], pans)):
                freq = f0 * 2 ** (cents / 1200) * vib * np.ones(total)
                sig = _osc(freq, sr, p["wave"], float(rng.random()))
                left, right = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
                voice[0] += left * sig
                voice[1] += right * sig
            voice /= np.sqrt(nv)
            for c in range(2):
                voice[c] = _filter_sweep(voice[c], sr, cutoff, p["res_db"])
            voice *= env * vel * p["gain"]
            s = int(n.start * sr)
            if s >= n_out:
                continue
            e = min(n_out, s + total)
            out[:, s:e] += voice[:, : e - s]
        if p["chorus"]:
            out = _chorus(out, sr, p["chorus"])
        if p["delay"]:
            out = _tempo_delay(out, sr, tempo, p["delay"])
        return out


def _chorus(x: np.ndarray, sr: int, depth: float) -> np.ndarray:
    n = x.shape[1]
    t = np.arange(n) / sr
    out = x.copy()
    for c, phase in ((0, 0.0), (1, np.pi / 2)):
        d = (0.012 + 0.004 * np.sin(2 * np.pi * 0.6 * t + phase)) * sr
        idx = np.arange(n) - d
        out[c] = (1 - 0.35 * depth) * x[c] + 0.5 * depth * np.interp(idx, np.arange(n), x[c], left=0)
    return out.astype(np.float32)


def _tempo_delay(x: np.ndarray, sr: int, tempo: float, mix: float, feedback: float = 0.38) -> np.ndarray:
    """Dotted-eighth ping-pong delay, lowpassed repeats (a classic 80s arp move)."""
    d = int(sr * 60 / max(40.0, tempo) * 0.75)
    out = x.copy()
    tap = x.mean(axis=0) * mix
    for k in range(1, 6):
        g = feedback ** (k - 1)
        shift = d * k
        if shift >= x.shape[1]:
            break
        ch = k % 2  # alternate sides
        out[ch, shift:] += g * tap[: x.shape[1] - shift]
    return out


# --------------------------------------------------------------------------- dispatch

class RendererSet:
    def __init__(self, samples_dir: str | Path | None = None, soundfont: str | None = None):
        self.sampler = SampleRenderer(samples_dir)
        self.gm = FluidSynthRenderer(soundfont)
        self.synth = SynthRenderer()

    def status(self) -> dict[str, Any]:
        return {"vsco": {"available": self.sampler.available, "presets": self.sampler.presets() if self.sampler.available else [], "license": self.sampler.license},
                "gm": {"available": self.gm.available, "soundfont": self.gm.soundfont, "license": self.gm.license},
                "synth": {"available": True, "presets": sorted(SYNTH_PRESETS), "license": self.synth.license}}

    def resolve(self, renderer: str, fallback: str | None) -> str:
        for r in (renderer, fallback):
            if not r:
                continue
            kind, _, arg = r.partition(":")
            if kind == "vsco" and self.sampler.has(arg):
                return r
            if kind == "gm" and self.gm.available:
                return r
            if kind == "synth":
                return r
        return "synth:" + _synth_for(renderer)

    def render(self, renderer: str, fallback: str | None, notes: list[PNote], n_out: int, sr: int, *, tempo: float, articulation: str, seed: int = 0) -> tuple[np.ndarray, str]:
        used = self.resolve(renderer, fallback)
        kind, _, arg = used.partition(":")
        if kind == "vsco":
            audio = self.sampler.render(arg, notes, n_out, sr, articulation)
        elif kind == "gm":
            audio = self.gm.render(int(arg), notes, n_out, sr, tempo=tempo)
        else:
            audio = self.synth.render(arg, notes, n_out, sr, tempo=tempo, seed=seed)
        return audio.astype(np.float32), used


def _synth_for(renderer: str) -> str:
    kind, _, arg = renderer.partition(":")
    if kind == "synth":
        return arg
    if "bass" in arg or "contrabass" in arg or arg in {"32", "33", "34", "58"}:
        return "bass"
    if arg in {"violins_pizz", "piano", "0", "46", "25", "27", "8", "11"}:
        return "pluck"
    return "pad"
