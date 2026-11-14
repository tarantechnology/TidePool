"""Explode: split a clip into named layers with the recipe tuned on real AudioSep.

Order of work (the editor sends the sounds in this order; this module also copes
with any order):

1. Voices: a dialogue specialist, Demucs (``htdemucs_ft``, two stems, ``vocals``),
   on the full mix. The rest is mix minus vocals, so nothing is lost. Measured on the
   Batman v Superman clip: speech 0.65 / music 0.06 against 0.14 / 0.37 for AudioSep's
   "people talking". Only one voice layer is possible this way; more voices merge
   into the first. Without Demucs installed, voices fall back to AudioSep, last.
2. Weather: AudioSep cannot tell rain from thunder when they overlap; whichever goes
   first takes both. So one "storm" pass, then a crossover: rain is the storm
   high-passed at 200 Hz (two 2-pole stages), thunder is the storm minus rain, the
   low rumble. Rain + thunder equals the storm exactly.
3. Every other sound: AudioSep, one at a time, each from what is left.
   With split_music, a music layer is then split into instruments by the music stem
   splitter (Demucs htdemucs_6s on the box): one layer per audible instrument, named
   "<music prompt>: <Instrument>", summing exactly to the music layer.
4. Everything else: the mix minus every layer, so the layers always sum back to the
   original.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

log = logging.getLogger("sam_audio_service")

VOICE_WORDS = ("talking", "talk", "speech", "speaking", "speaker", "voice", "voices", "dialogue", "dialog",
               "conversation", "narrator", "narration", "voiceover", "whisper", "whispering", "shouting",
               "yelling", "singing", "vocals", "people", "person", "man", "woman", "men", "women")
RAIN_WORDS = ("rain", "raining", "drizzle", "downpour", "storm", "rainstorm", "weather")
MUSIC_WORDS = ("music", "score", "soundtrack", "orchestra", "orchestral", "song", "melody", "strings", "choir", "instrumental")
THUNDER_WORDS = ("thunder", "thunderclap", "thunderstorm", "lightning", "rumble", "rumbling")
# Words that make a prompt a voice even with "rain" in it ("man shouting over the rain").
VOICE_ACTIONS = ("talking", "speech", "speaking", "voice", "dialogue", "narrator", "singing", "vocals", "whispering", "shouting", "yelling")


def _words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def _has(text: str, vocab: tuple[str, ...]) -> bool:
    words = _words(text)
    return any(w in words or w + "s" in words or w + "es" in words for w in vocab)


def kind_of(prompt: str) -> str:
    """voice | thunder | rain | music | sound."""
    if _has(prompt, VOICE_ACTIONS) or (_has(prompt, VOICE_WORDS) and not _has(prompt, RAIN_WORDS + THUNDER_WORDS)):
        return "voice"
    if _has(prompt, THUNDER_WORDS):
        return "thunder"
    if _has(prompt, RAIN_WORDS):
        return "rain"
    if _has(prompt, MUSIC_WORDS):
        return "music"
    return "sound"


def highpass(samples: np.ndarray, sample_rate: int, cutoff: float = 200.0, stages: int = 2) -> np.ndarray:
    """``stages`` 2-pole (RBJ, Q = 1/sqrt 2) high-pass passes along the last axis, alternating
    forward and backward in time. Two stages are the recipe's 2x 2-pole slope with zero phase,
    so "storm minus rain" is a clean low band (a one-way filter shifts the highs' phase and
    leaves part of them in the subtraction)."""
    w0 = 2 * math.pi * cutoff / sample_rate
    alpha = math.sin(w0) / (2 * (1 / math.sqrt(2)))
    cos = math.cos(w0)
    a0 = 1 + alpha
    b = np.array([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]) / a0
    a = np.array([1.0, -2 * cos / a0, (1 - alpha) / a0])
    x = np.asarray(samples, dtype=np.float64)
    try:
        import torch
        import torchaudio

        y = torch.as_tensor(x)
        for _ in range(stages):
            y = torchaudio.functional.lfilter(y, torch.as_tensor(a), torch.as_tensor(b), clamp=False).flip(-1)
        if stages % 2:
            y = y.flip(-1)
        return y.numpy().astype(np.float32)
    except ImportError:
        pass
    y = x.reshape(-1, x.shape[-1]).copy()
    for _ in range(stages):
        out = np.zeros_like(y)
        for c in range(y.shape[0]):
            x1 = x2 = y1 = y2 = 0.0
            row, dst = y[c], out[c]
            for n in range(row.shape[0]):
                v = b[0] * row[n] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
                x2, x1, y2, y1 = x1, row[n], y1, v
                dst[n] = v
        y = out[:, ::-1].copy()
    if stages % 2:
        y = y[:, ::-1]
    return y.reshape(x.shape).astype(np.float32)


class DemucsVocals:
    """Dialogue extraction with Demucs. Loaded on first use; ``available`` is False without the package."""

    def __init__(self, model: str | None = None, shifts: int | None = None):
        self.model_name = model or os.getenv("AUDIO_DEMUCS_MODEL", "htdemucs_ft")
        self.shifts = int(shifts if shifts is not None else os.getenv("AUDIO_DEMUCS_SHIFTS", "2"))
        self.model = None
        self.device = "cpu"

    @property
    def available(self) -> bool:
        try:
            import demucs  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from demucs.pretrained import get_model

        self.device = os.getenv("AUDIO_DEMUCS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(self.model_name)
        model.eval()
        self.model = model.to(self.device)

    def vocals(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Vocals of ``samples`` (channels, frames), same shape and rate."""
        import torch
        import torchaudio
        from demucs.apply import apply_model

        self.load()
        model = self.model
        x = torch.as_tensor(np.asarray(samples, dtype=np.float32))
        channels, frames = x.shape
        if channels != model.audio_channels:
            x = x.mean(0, keepdim=True).repeat(model.audio_channels, 1)
        x = torchaudio.functional.resample(x, sample_rate, model.samplerate)
        # As the demucs CLI does: normalise by the mono reference, then undo it.
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.inference_mode():
            sources = apply_model(model, ((x - mean) / std)[None].to(self.device), shifts=self.shifts,
                                  split=True, overlap=0.25, progress=False, device=self.device)[0]
        vocals = (sources[model.sources.index("vocals")] * std + mean).float().cpu()
        vocals = torchaudio.functional.resample(vocals, model.samplerate, sample_rate)
        if channels != vocals.shape[0]:
            vocals = vocals.mean(0, keepdim=True).repeat(channels, 1)
        out = np.zeros((channels, frames), dtype=np.float32)
        n = min(frames, vocals.shape[1])
        out[:, :n] = vocals[:, :n].numpy()
        return out


@dataclass
class Layer:
    name: str
    samples: np.ndarray
    prompt: str
    how: str  # demucs | audiosep | crossover-high | crossover-low | storm


@dataclass
class ExplodeResult:
    layers: list[Layer]
    residual: np.ndarray
    warnings: list[str] = field(default_factory=list)


def explode(mix: np.ndarray, sample_rate: int, prompts: list[str],
            separate: Callable[[np.ndarray, str], np.ndarray],
            vocals: Callable[[np.ndarray], np.ndarray] | None,
            storm_prompt: str | None = None, crossover_hz: float | None = None,
            split_music: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
            quiet_db: float = -35.0) -> ExplodeResult:
    """Run the recipe. ``separate(audio, prompt)`` returns the AudioSep target for ``audio``;
    ``vocals(audio)`` returns Demucs vocals, or is None when Demucs is unavailable.
    ``split_music(audio)`` returns instrument stems that sum to ``audio``; music layers are
    replaced by their audible instruments (quieter than ``quiet_db`` below the loudest fold
    into the loudest, so the sum is unchanged)."""
    storm_prompt = storm_prompt or os.getenv("AUDIO_STORM_PROMPT", "rain")
    crossover_hz = crossover_hz or float(os.getenv("AUDIO_STORM_CROSSOVER_HZ", "200"))
    remaining = np.asarray(mix, dtype=np.float32).copy()
    layers: list[Layer] = []
    warnings: list[str] = []
    seen: set[str] = set()
    ordered = [p.strip() for p in prompts if p.strip() and not (p.strip().lower() in seen or seen.add(p.strip().lower()))]
    kinds = {p: kind_of(p) for p in ordered}
    voices = [p for p in ordered if kinds[p] == "voice"]
    rain = next((p for p in ordered if kinds[p] == "rain"), None)
    thunder = next((p for p in ordered if kinds[p] == "thunder"), None)
    late_voices: list[str] = []
    weather_done = False

    for prompt in ordered:
        kind = kinds[prompt]
        if kind == "voice":
            if prompt != voices[0]:
                warnings.append(f'"{prompt}" is in the "{voices[0]}" layer: the dialogue model makes one voice layer.')
                continue
            if vocals is None:
                late_voices.append(prompt)
                warnings.append("Dialogue model (Demucs) is not installed; voices were separated by text last.")
                continue
            v = vocals(remaining)
            # Every voice lands in this one layer, so it is named for what it is, not for the first prompt.
            layers.append(Layer("dialogue", v, prompt, "demucs"))
            remaining = remaining - v
        elif kind in ("rain", "thunder"):
            if weather_done:
                continue
            weather_done = True
            storm = separate(remaining, storm_prompt)
            remaining = remaining - storm
            if rain and thunder:
                high = highpass(storm, sample_rate, crossover_hz)
                # Same position in the list as the first weather sound, rain first.
                layers.append(Layer(rain, high, storm_prompt, "crossover-high"))
                layers.append(Layer(thunder, (storm - high).astype(np.float32), storm_prompt, "crossover-low"))
            else:
                only = rain or thunder
                layers.append(Layer(only, storm, storm_prompt, "storm"))
        else:
            target = separate(remaining, prompt)
            remaining = remaining - target
            if kind == "music" and split_music is not None:
                layers.extend(_instruments(prompt, target, split_music, quiet_db, warnings))
            else:
                layers.append(Layer(prompt, target, prompt, "audiosep"))
    for prompt in late_voices:
        target = separate(remaining, prompt)
        layers.append(Layer(prompt, target, prompt, "audiosep"))
        remaining = remaining - target
    residual = (np.asarray(mix, dtype=np.float32) - sum((l.samples for l in layers), np.zeros_like(remaining))).astype(np.float32)
    return ExplodeResult(layers, residual, warnings)


def _instruments(prompt: str, music: np.ndarray, split_music: Callable[[np.ndarray], dict[str, np.ndarray]],
                 quiet_db: float, warnings: list[str]) -> list[Layer]:
    try:
        parts = split_music(music)
    except Exception as exc:  # the music layer stays whole rather than failing the Explode
        log.exception("music split failed")
        warnings.append(f'"{prompt}" was kept whole: splitting it into instruments failed ({exc}).')
        return [Layer(prompt, music, prompt, "audiosep")]
    if not parts:
        return [Layer(prompt, music, prompt, "audiosep")]
    level = {name: 10 * math.log10(float(np.mean(np.square(x))) + 1e-12) for name, x in parts.items()}
    loudest = max(level, key=level.get)
    keep = {n: np.asarray(x, dtype=np.float32).copy() for n, x in parts.items() if level[n] >= level[loudest] + quiet_db}
    for name, x in parts.items():
        if name not in keep:
            keep[loudest] = keep[loudest] + np.asarray(x, dtype=np.float32)
    # Exact: the instruments sum to the music layer.
    keep[loudest] = keep[loudest] + (np.asarray(music, dtype=np.float32) - sum(keep.values()))
    if len(keep) == 1:
        return [Layer(prompt, music, prompt, "audiosep")]
    order = sorted(keep, key=lambda n: -level[n])
    return [Layer(f"{prompt}: {n.title()}", keep[n], prompt, "music-stems") for n in order]


def recipe_info() -> dict[str, Any]:
    return {"storm_prompt": os.getenv("AUDIO_STORM_PROMPT", "rain"), "crossover_hz": float(os.getenv("AUDIO_STORM_CROSSOVER_HZ", "200")),
            "dialogue_model": os.getenv("AUDIO_DEMUCS_MODEL", "htdemucs_ft")}
