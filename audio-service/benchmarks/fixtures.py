"""Deterministic synthetic benchmark fixtures with ground-truth sources.

Nothing here is checked in as audio: every fixture is generated at runtime from a seed,
so CI, a laptop and a GPU box all measure the same signals. Written files go to
``audio-service/fixtures/generated/`` (gitignored) together with a ``manifest.json`` so
Person 3 (editor) and Person 4 (backend) can drive their own flows from the same mixtures
and compare against the same ground truth.

Fixture set (the minimum the cinematic demo and the separation experiments need):

``speech_plus_siren``
    One person arguing over a background siren. Prompt ``"siren"``. This is the demo
    case: the siren has to come out as its own stem so the editor can automate it.
``two_speakers``
    Two people arguing, different pitch ranges and speaking rhythms, panned apart.
    Prompt ``"man speaking"`` -- the speaker-separation experiment.
``music_mix``
    Percussive kit (kick/click/noise) plus a sustained plucked harmonic tone.
    Prompt ``"guitar"`` -- the music-separation experiment.
``long_mixture(seconds)``
    Mono speech + siren of arbitrary length, for latency / real-time-factor measurement.

Speech is real speech when macOS ``say`` is available (``speech_mode="auto"``) and a
synthetic formant/harmonic voice otherwise, so the suite behaves identically on Linux CI.
The fixture records which was used in ``speech_source``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python benchmarks/fixtures.py` as well as -m
    sys.path.insert(0, str(ROOT))

from sam_audio_service.audio_io import MODEL_SAMPLE_RATE, decode, encode_wav_pcm16  # noqa: E402

SAMPLE_RATE = MODEL_SAMPLE_RATE
DEFAULT_OUT_DIR = ROOT / "fixtures" / "generated"
MANIFEST_SCHEMA = "soundcanvas/audio-fixtures-1"
PEAK_CEILING = 0.89  # leaves headroom so 16-bit WAV round-trips never clip

SpeechMode = Literal["auto", "say", "synthetic"]

# macOS voices in preference order, split by register so `two_speakers` really is two
# different voices rather than the same one twice.
SAY_VOICES_LOW = ("Alex", "Daniel", "Fred", "Ralph", "Tom", "Albert")
SAY_VOICES_HIGH = ("Samantha", "Karen", "Victoria", "Moira", "Tessa", "Kathy")
DEMO_LINES = (
    "I told you to turn it off, not unplug the entire thing.",
    "You never listen to a single word I say, not once.",
)

__all__ = [
    "Fixture",
    "SAMPLE_RATE",
    "PEAK_CEILING",
    "MANIFEST_SCHEMA",
    "QUALITY_FIXTURES",
    "DEFAULT_OUT_DIR",
    "speech_plus_siren",
    "two_speakers",
    "music_mix",
    "long_mixture",
    "build_all",
    "write_fixture_set",
    "say_available",
]


# --------------------------------------------------------------------------- container
@dataclass(frozen=True)
class Fixture:
    """A mixture plus the ground-truth sources that were summed to make it.

    ``sources`` maps source name -> float32 ``(channels, frames)``. ``mixture`` is exactly
    their sum, so ``reconstruction_error`` on target+residual is meaningful.
    """

    name: str
    prompt: str
    target: str
    sources: dict[str, np.ndarray]
    sample_rate: int = SAMPLE_RATE
    notes: str = ""
    speech_source: str = "synthetic"
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def mixture(self) -> np.ndarray:
        total = np.zeros_like(next(iter(self.sources.values())))
        for source in self.sources.values():
            total = total + source
        return total.astype(np.float32)

    @property
    def target_signal(self) -> np.ndarray:
        return self.sources[self.target]

    @property
    def interference(self) -> np.ndarray:
        """Everything that is not the target, summed: the reference for leakage."""
        others = [s for name, s in self.sources.items() if name != self.target]
        if not others:
            return np.zeros_like(self.target_signal)
        total = np.zeros_like(others[0])
        for source in others:
            total = total + source
        return total.astype(np.float32)

    @property
    def frames(self) -> int:
        return int(self.target_signal.shape[-1])

    @property
    def channels(self) -> int:
        return int(self.target_signal.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


# ------------------------------------------------------------------------ dsp helpers
def _frames(seconds: float) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return int(round(seconds * SAMPLE_RATE))


def _time(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / SAMPLE_RATE


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _at_rms(x: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale ``x`` to a target RMS level in dBFS (silence is returned untouched)."""
    current = _rms(x)
    if current <= 1e-12:
        return x
    return x * (10.0 ** (dbfs / 20.0) / current)


def _pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan a mono signal to ``(2, frames)``. ``pan`` is -1 left, +1 right."""
    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    return np.stack([mono * np.cos(angle), mono * np.sin(angle)])


def _fade(x: np.ndarray, seconds: float = 0.02) -> np.ndarray:
    """Short raised-cosine fade in/out so sources never start or stop with a click."""
    n = min(int(seconds * SAMPLE_RATE), x.shape[-1] // 2)
    if n < 2:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))
    out = x.copy()
    out[..., :n] *= ramp
    out[..., -n:] *= ramp[::-1]
    return out


def _normalise(sources: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scale all sources by one common gain so their sum peaks at ``PEAK_CEILING``."""
    total = np.zeros_like(next(iter(sources.values())))
    for source in sources.values():
        total = total + source
    peak = float(np.max(np.abs(total))) if total.size else 0.0
    gain = PEAK_CEILING / peak if peak > PEAK_CEILING else 1.0
    return {name: (source * gain).astype(np.float32) for name, source in sources.items()}


def _syllable_envelope(n: int, rng: np.random.Generator, rate: float, pause_prob: float = 0.22) -> np.ndarray:
    """Amplitude envelope of syllables at ``rate`` per second with occasional pauses."""
    env = np.zeros(n, dtype=np.float64)
    period = SAMPLE_RATE / rate
    position = 0.0
    while position < n:
        length = int(period * rng.uniform(0.45, 0.85))
        start, end = int(position), min(int(position) + length, n)
        if end - start >= 2 and rng.random() > pause_prob:
            env[start:end] = np.hanning(end - start) ** 0.65 * rng.uniform(0.6, 1.0)
        position += period * rng.uniform(0.85, 1.3)
    return env


def _formant_gain(freqs: np.ndarray, formants: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    gain = np.zeros_like(freqs)
    for centre, bandwidth, amplitude in formants:
        gain += amplitude * np.exp(-0.5 * ((freqs - centre) / bandwidth) ** 2)
    return gain + 0.01


def _speech_like(
    seconds: float,
    rng: np.random.Generator,
    *,
    f0: float = 112.0,
    syllable_rate: float = 3.8,
    formants: tuple[tuple[float, float, float], ...] = ((700.0, 130.0, 1.0), (1220.0, 180.0, 0.55), (2600.0, 260.0, 0.22)),
) -> np.ndarray:
    """Synthetic voice: jittered harmonic source shaped by three formants, plus breath.

    Not intelligible, but it has the spectral shape, harmonic structure and syllable-rate
    modulation that separation models and band-based mocks react to.
    """
    n = _frames(seconds)
    t = _time(n)
    contour = f0 * (1.0 + 0.07 * np.sin(2 * np.pi * 0.63 * t + rng.uniform(0, 6.28)) + 0.03 * np.sin(2 * np.pi * 2.7 * t))
    phase = 2 * np.pi * np.cumsum(contour) / SAMPLE_RATE
    harmonics = np.arange(1, int(8000.0 / f0) + 1, dtype=np.float64)
    weights = _formant_gain(harmonics * f0, formants) / harmonics
    signal = np.zeros(n, dtype=np.float64)
    for order, weight in zip(harmonics, weights, strict=True):
        signal += weight * np.sin(order * phase + rng.uniform(0, 2 * np.pi))
    envelope = _syllable_envelope(n, rng, syllable_rate)
    breath = 0.04 * rng.standard_normal(n)
    return _fade((signal / max(float(np.max(np.abs(signal))), 1e-9) + breath) * envelope)


def _siren(seconds: float, *, low: float = 600.0, high: float = 1500.0, sweep_hz: float = 0.55) -> np.ndarray:
    """Swept-tone siren between ``low`` and ``high`` Hz with a quiet second harmonic."""
    n = _frames(seconds)
    t = _time(n)
    triangle = 2.0 * np.abs(((t * sweep_hz) % 1.0) - 0.5)  # 0..1..0
    frequency = low + (high - low) * triangle
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    wave = np.sin(phase) + 0.25 * np.sin(2 * phase)
    return _fade(wave * (0.92 + 0.08 * np.sin(2 * np.pi * 5.0 * t)), 0.08)


def _percussion(seconds: float, rng: np.random.Generator, *, beat_hz: float = 2.0) -> np.ndarray:
    """Kick + click/hat pattern: low chirped thumps on the beat, bright bursts between."""
    n = _frames(seconds)
    out = np.zeros(n, dtype=np.float64)
    period = SAMPLE_RATE / beat_hz
    for index in range(int(n / period) + 1):
        start = int(index * period)
        if start >= n:
            break
        # Kick: downward chirp 95 -> 45 Hz, fast exponential decay.
        length = min(int(0.28 * SAMPLE_RATE), n - start)
        if length > 16:
            kt = _time(length)
            sweep = 95.0 * np.exp(-kt / 0.045) + 45.0
            out[start:start + length] += np.sin(2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE) * np.exp(-kt / 0.075)
        # Off-beat bright burst: noise differenced once (a cheap high-pass) and decayed.
        offset = start + int(period * 0.5)
        length = min(int(0.09 * SAMPLE_RATE), n - offset)
        if offset < n and length > 16:
            noise = np.diff(rng.standard_normal(length + 1))
            out[offset:offset + length] += 0.45 * noise * np.exp(-_time(length) / 0.016)
    return _fade(out / max(float(np.max(np.abs(out))), 1e-9))


def _plucked_tone(seconds: float, rng: np.random.Generator, *, f0: float = 330.0, pluck_hz: float = 1.1) -> np.ndarray:
    """Sustained plucked harmonic tone ("guitar"): 12 harmonics, vibrato, decaying notes."""
    n = _frames(seconds)
    t = _time(n)
    semitones = np.array([0, 3, 7, 5], dtype=np.float64)
    envelope = np.zeros(n, dtype=np.float64)
    pitch = np.zeros(n, dtype=np.float64)
    period = SAMPLE_RATE / pluck_hz
    for index in range(int(n / period) + 1):
        start = int(index * period)
        length = min(int(period * 1.05), n - start)
        if length <= 16:
            break
        note = f0 * 2.0 ** (semitones[index % len(semitones)] / 12.0)
        decay = np.exp(-_time(length) / 0.55)
        attack = np.minimum(_time(length) / 0.006, 1.0)
        envelope[start:start + length] += decay * attack * rng.uniform(0.8, 1.0)
        pitch[start:start + length] = note
    pitch[pitch == 0.0] = f0
    vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.1 * t)
    phase = 2 * np.pi * np.cumsum(pitch * vibrato) / SAMPLE_RATE
    orders = np.arange(1, 13, dtype=np.float64)
    signal = np.zeros(n, dtype=np.float64)
    for order in orders:
        signal += np.sin(order * phase + rng.uniform(0, 2 * np.pi)) / order ** 1.35
    return _fade(signal * envelope / max(float(np.max(np.abs(signal * envelope))), 1e-9))


# ------------------------------------------------------------------------ real speech
def say_available() -> bool:
    """True when macOS ``say`` can be used (never true on Linux CI)."""
    return shutil.which("say") is not None


def _say_voices(pool: tuple[str, ...]) -> list[str]:
    """Installed voices from ``pool``, in preference order (empty when ``say`` is absent)."""
    if not say_available():
        return []
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, timeout=20, check=False).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return []
    installed = {line.split()[0] for line in listing.splitlines() if line.strip()}
    return [voice for voice in pool if voice in installed]


def _say_speech(text: str, voice: str, seconds: float, rng: np.random.Generator) -> np.ndarray | None:
    """Render ``text`` with ``say``, decode to mono 48 kHz and fit it to ``seconds``.

    Returns ``None`` whenever anything is missing or fails, so callers fall back to the
    synthetic voice. Decoding reuses :func:`sam_audio_service.audio_io.decode`.
    """
    n = _frames(seconds)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "speech.aiff"
        try:
            proc = subprocess.run(["say", "-v", voice, "-r", "180", "-o", str(path), text], capture_output=True, timeout=60, check=False)
            if proc.returncode != 0 or not path.exists():
                return None
            audio = decode(path=path, target_rate=SAMPLE_RATE, max_channels=1)
        except Exception:  # noqa: BLE001 - any failure falls back to the synthetic voice
            return None
    mono = audio.samples.mean(axis=0).astype(np.float64)
    if mono.size >= n:
        return _fade(mono[:n])
    out = np.zeros(n, dtype=np.float64)
    tile = mono
    position = 0
    while position < n:  # repeat the line with a short gap so long fixtures stay busy
        end = min(position + tile.size, n)
        out[position:end] += tile[: end - position]
        position = end + int(rng.uniform(0.25, 0.6) * SAMPLE_RATE)
    return _fade(out)


def _speech(seconds: float, rng: np.random.Generator, mode: SpeechMode, *, line: str, voices: tuple[str, ...], f0: float, syllable_rate: float) -> tuple[np.ndarray, str]:
    """Speech for one talker: real ``say`` audio when allowed and available, else synthetic.

    Returns the mono signal and a tag naming where it came from (``say:<voice>`` or
    ``synthetic``) so the benchmark report can state what was actually measured.
    """
    if mode in ("auto", "say"):
        for voice in _say_voices(voices):
            rendered = _say_speech(line, voice, seconds, rng)
            if rendered is not None:
                return rendered, f"say:{voice}"
        if mode == "say":
            raise RuntimeError("speech_mode='say' requested but macOS `say` is unavailable or failed")
    formants = ((700.0 * (f0 / 112.0) ** 0.25, 130.0, 1.0), (1220.0 * (f0 / 112.0) ** 0.2, 180.0, 0.55), (2600.0, 260.0, 0.22))
    return _speech_like(seconds, rng, f0=f0, syllable_rate=syllable_rate, formants=formants), "synthetic"


# --------------------------------------------------------------------------- fixtures
def speech_plus_siren(seconds: float = 6.0, *, seed: int = 11, speech_mode: SpeechMode = "auto") -> Fixture:
    """Demo case: one talker over a background siren. Prompt ``"siren"``."""
    rng = np.random.default_rng(seed)
    speech, source = _speech(seconds, rng, speech_mode, line=DEMO_LINES[0], voices=SAY_VOICES_LOW, f0=112.0, syllable_rate=3.8)
    siren = _siren(seconds)
    sources = {
        "speech": _pan(_at_rms(speech, -18.0), -0.15),
        "siren": _pan(_at_rms(siren, -24.0), 0.45),
    }
    return Fixture(
        name="speech_plus_siren",
        prompt="siren",
        target="siren",
        sources=_normalise(sources),
        notes="Cinematic demo case: siren must come out as its own stem, 6 dB under the voice, panned right.",
        speech_source=source,
    )


def two_speakers(seconds: float = 6.0, *, seed: int = 23, speech_mode: SpeechMode = "auto") -> Fixture:
    """Speaker-separation experiment: two talkers, prompt the lower-pitched one."""
    rng = np.random.default_rng(seed)
    low, low_source = _speech(seconds, rng, speech_mode, line=DEMO_LINES[0], voices=SAY_VOICES_LOW, f0=104.0, syllable_rate=3.4)
    high, high_source = _speech(seconds, rng, speech_mode, line=DEMO_LINES[1], voices=SAY_VOICES_HIGH, f0=205.0, syllable_rate=4.9)
    sources = {
        "speaker_low": _pan(_at_rms(low, -18.0), -0.5),
        "speaker_high": _pan(_at_rms(high, -18.5), 0.5),
    }
    return Fixture(
        name="two_speakers",
        prompt="man speaking",
        target="speaker_low",
        sources=_normalise(sources),
        notes="Two talkers at equal level, ~100 Hz vs ~205 Hz f0, panned apart. Hard case: prompts cannot name a person.",
        speech_source=f"low={low_source} high={high_source}",
        extra={"other_prompt": "woman speaking", "other_target": "speaker_high"},
    )


def music_mix(seconds: float = 6.0, *, seed: int = 37, speech_mode: SpeechMode = "auto") -> Fixture:
    """Music-separation experiment: percussion + sustained plucked tone. Prompt ``"guitar"``.

    ``speech_mode`` is accepted (and unused) so every fixture builder has one signature.
    """
    rng = np.random.default_rng(seed)
    sources = {
        "drums": _pan(_at_rms(_percussion(seconds, rng), -20.0), -0.25),
        "guitar": _pan(_at_rms(_plucked_tone(seconds, rng), -19.0), 0.25),
    }
    return Fixture(
        name="music_mix",
        prompt="guitar",
        target="guitar",
        sources=_normalise(sources),
        notes="Percussive kit vs sustained harmonic tone; overlapping low band makes band-pass mocks fail here.",
        speech_source="n/a",
        extra={"other_prompt": "drums", "other_target": "drums"},
    )


def long_mixture(seconds: float = 30.0, *, seed: int = 59, speech_mode: SpeechMode = "auto") -> Fixture:
    """Mono speech + siren of arbitrary length, used only for latency / RTF."""
    rng = np.random.default_rng(seed)
    speech, source = _speech(seconds, rng, speech_mode, line=DEMO_LINES[1], voices=SAY_VOICES_LOW, f0=118.0, syllable_rate=4.1)
    sources = {
        "speech": _at_rms(speech, -18.0)[None, :].astype(np.float32),
        "siren": _at_rms(_siren(seconds, sweep_hz=0.4), -23.0)[None, :].astype(np.float32),
    }
    return Fixture(
        name=f"long_mixture_{seconds:g}s",
        prompt="siren",
        target="siren",
        sources=_normalise(sources),
        notes=f"Mono {seconds:g}s mixture for latency and real-time factor.",
        speech_source=source,
    )


QUALITY_FIXTURES = ("speech_plus_siren", "two_speakers", "music_mix")
_BUILDERS = {"speech_plus_siren": speech_plus_siren, "two_speakers": two_speakers, "music_mix": music_mix}


def build_all(seconds: float = 6.0, *, speech_mode: SpeechMode = "auto", names: tuple[str, ...] = QUALITY_FIXTURES) -> list[Fixture]:
    """Build the quality fixtures (not the latency one, whose length is separate)."""
    unknown = [name for name in names if name not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown fixture(s) {unknown}; available: {sorted(_BUILDERS)}")
    return [_BUILDERS[name](seconds, speech_mode=speech_mode) for name in names]


# ---------------------------------------------------------------------------- writing
def write_fixture_set(
    out_dir: Path | str = DEFAULT_OUT_DIR,
    *,
    seconds: float = 6.0,
    speech_mode: SpeechMode = "auto",
    latency_seconds: float = 30.0,
) -> Path:
    """Write every fixture as 16-bit WAVs plus ``manifest.json``; returns the manifest path.

    Layout (paths in the manifest are relative to ``out_dir``)::

        <out_dir>/manifest.json
        <out_dir>/speech_plus_siren/mixture.wav
        <out_dir>/speech_plus_siren/speech.wav
        <out_dir>/speech_plus_siren/siren.wav
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = build_all(seconds, speech_mode=speech_mode)
    if latency_seconds > 0:
        fixtures.append(long_mixture(latency_seconds, speech_mode=speech_mode))
    entries = []
    for fixture in fixtures:
        folder = directory / fixture.name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "mixture.wav").write_bytes(encode_wav_pcm16(fixture.mixture, fixture.sample_rate))
        sources = []
        for name, samples in fixture.sources.items():
            (folder / f"{name}.wav").write_bytes(encode_wav_pcm16(samples, fixture.sample_rate))
            sources.append({"name": name, "file": f"{fixture.name}/{name}.wav", "role": "target" if name == fixture.target else "interference"})
        entries.append({
            "name": fixture.name,
            "prompt": fixture.prompt,
            "target_source": fixture.target,
            "mixture": f"{fixture.name}/mixture.wav",
            "sources": sources,
            "duration": round(fixture.duration, 6),
            "channels": fixture.channels,
            "sample_rate": fixture.sample_rate,
            "speech_source": fixture.speech_source,
            "notes": fixture.notes,
            **({"extra": fixture.extra} if fixture.extra else {}),
        })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "audio-service/benchmarks/fixtures.py (Person 1)",
        "sample_rate": SAMPLE_RATE,
        "format": "wav/pcm_s16le",
        "speech_mode": speech_mode,
        "seconds": seconds,
        "latency_seconds": latency_seconds,
        "note": "Synthetic, deterministic from fixed seeds. Not checked in; regenerate with "
                "`python -m benchmarks.fixtures --out <dir>`. Ground-truth sources sum exactly to mixture.wav "
                "before 16-bit quantisation.",
        "fixtures": entries,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write the SoundCanvas audio benchmark fixture set.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help=f"output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--seconds", type=float, default=6.0, help="length of each quality fixture (default: 6)")
    parser.add_argument("--latency-seconds", type=float, default=30.0, help="length of the latency fixture, 0 to skip (default: 30)")
    parser.add_argument("--speech-mode", choices=("auto", "say", "synthetic"), default="auto", help="speech source (default: auto)")
    args = parser.parse_args(argv)
    path = write_fixture_set(args.out, seconds=args.seconds, speech_mode=args.speech_mode, latency_seconds=args.latency_seconds)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
