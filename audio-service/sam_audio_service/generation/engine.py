"""Generator: analyse the song (from its stems) and add a new part that fits it.

Pipeline: analyse (music_analysis) -> resolve the ask (vocabulary) -> compose (rule-based, or AMT)
-> render (VSCO-2 / FluidSynth GM / built-in synth) -> mix in (level, EQ carve, matched reverb,
ducking) -> checks. The audio-generative path (MusicGen-melody) replaces compose+render with a
text + chroma-conditioned model and is flagged with its licence.

Locks: ``_model_lock`` serialises the GPU models used here (transcriber, AMT, MusicGen); it is
separate from the separation engine's lock, so a generate request never blocks /separate on a
long split (and vice versa). Rendering/mixing run outside the lock.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import InvalidRequest, MissingModel, UnsupportedOperation
from ..music_analysis import MusicAnalyzer
from . import dsp
from .backends import LICENSES, AmtComposer, BasicPitchTranscriber, BeatThisBackend, MusicGenGenerator
from .composer import PNote, compose
from .evaluate import audio_checks, note_checks
from .midi import write_midi
from .mixin import mix_in
from .render import RendererSet
from .settings import GenSettings, resolve_device
from .vocabulary import PartSpec, capabilities as vocab_capabilities, resolve

log = logging.getLogger("sam_audio_service")

ENGINES = ("auto", "symbolic", "amt", "musicgen")
SECTION_WORDS = ("intro", "verse", "chorus", "bridge", "outro")


@dataclass
class PartRequest:
    instrument: str | None = None
    prompt: str | None = None
    role: str | None = None
    style: str | None = None
    sections: list[str | int] | None = None
    interval: tuple[float, float] | None = None
    density: float | None = None
    engine: str = "auto"
    seed: int = 0
    level_db: float = 0.0
    duck_db: float = 3.0
    humanize: bool = True
    analysis: dict[str, Any] | None = None


@dataclass
class PartResult:
    audio: np.ndarray
    sample_rate: int
    midi: bytes | None
    notes: list[PNote]
    spec: PartSpec
    analysis: dict[str, Any]
    windows: list[tuple[float, float]]
    engine: dict[str, Any]
    renderer: dict[str, Any] | None
    mix: dict[str, Any]
    checks: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    elapsed: dict[str, float] = field(default_factory=dict)


def _windows(analysis: dict[str, Any], sections, interval, prompt: str | None) -> tuple[list[tuple[float, float]], list[str]]:
    dur = float(analysis["duration"])
    secs = analysis.get("sections") or []
    warnings: list[str] = []
    if interval is not None:
        a, b = float(interval[0]), float(interval[1])
        if not (0 <= a < b):
            raise InvalidRequest("interval must satisfy 0 <= start < end")
        return [(a, min(b, dur))], warnings
    wanted = list(sections or [])
    if not wanted and prompt:
        wanted = [w for w in SECTION_WORDS if re.search(r"\b" + w + r"(e?s)?\b", prompt.lower()) and not re.search(r"chorus (voice|effect)", prompt.lower())]
    if not wanted:
        return [(0.0, dur)], warnings
    out = []
    for w in wanted:
        if isinstance(w, int) or (isinstance(w, str) and w.isdigit()):
            out += [(s["start"], s["end"]) for s in secs if s["index"] == int(w)]
        else:
            key = str(w).lower().strip()
            out += [(s["start"], s["end"]) for s in secs if s.get("name") == key or s.get("label", "").lower() == key]
    if not out:
        warnings.append(f"no section matched {wanted}; generating over the whole song")
        return [(0.0, dur)], warnings
    out.sort()
    merged = [out[0]]
    for a, b in out[1:]:
        if a <= merged[-1][1] + 1e-3:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged, warnings


def _gm_program(spec: PartSpec) -> int:
    for r in (spec.renderer, spec.fallback_renderer):
        if r and r.startswith("gm:"):
            return int(r[3:])
    return {"strings": 48, "keys": 0, "brass": 61, "winds": 73, "synth": 81, "bass": 33, "guitar": 27, "voice": 52, "mallets": 11, "plucked": 46}.get(spec.family, 48)


def _audiogen_prompt(spec: PartSpec, analysis: dict[str, Any], prompt: str | None) -> str:
    key = (analysis.get("key") or {}).get("label", "")
    tempo = analysis.get("tempo")
    what = prompt or spec.display_name
    role = {"pad": "sustained pad", "counter_melody": "counter-melody", "lead": "lead melody", "ostinato": "arpeggio", "bassline": "bassline",
            "riff": "riff", "stabs": "stabs", "chords": "chords"}.get(spec.role, spec.role)
    return f"solo {what}, {role}, {key}, {tempo:.0f} bpm, studio recording, no drums, no vocals"


class Generator:
    def __init__(self, settings: GenSettings | None = None, *, analyzer: MusicAnalyzer | None = None, renderers: RendererSet | None = None,
                 amt: AmtComposer | None = None, audiogen: MusicGenGenerator | None = None, load_models: bool = True):
        self.settings = settings or GenSettings()
        device = resolve_device(self.settings.device)
        beat = trans = None
        if load_models:
            if self.settings.beats in ("auto", "beat_this") and _importable("beat_this"):
                beat = BeatThisBackend(device=device)
            if self.settings.transcriber in ("auto", "basic_pitch") and _importable("basic_pitch"):
                trans = BasicPitchTranscriber()
            if amt is None and self.settings.amt_model != "off" and _importable("anticipation") and _importable("transformers"):
                amt = AmtComposer(self.settings.amt_model, device=device)
            if audiogen is None and self.settings.audiogen == "musicgen" and _importable("transformers"):
                audiogen = MusicGenGenerator(self.settings.musicgen_model, device=device)
        self.analyzer = analyzer or MusicAnalyzer(beat_backend=beat, transcriber=trans)
        self.renderers = renderers or RendererSet(self.settings.samples_dir, self.settings.soundfont)
        self.amt = amt
        self.audiogen = audiogen
        self._model_lock = threading.Lock()

    # -------------------------------------------------------------- capabilities
    def capabilities(self) -> dict[str, Any]:
        vc = vocab_capabilities()
        r = self.renderers.status()
        return {
            "available": self.settings.enabled,
            "instruments": [i["key"] for i in vc["instruments"]],
            "instrument_table": vc["instruments"],
            "roles": vc["roles"],
            "free_text": True,
            "engines": {
                "symbolic": {"available": True, "default": True, "composer": "rule-based voice-leading arranger", "commercial_use": True},
                # Experimental, disabled by default; enable explicitly per deployment.
                "amt": {"available": self.amt is not None, "experimental": True, "model": getattr(self.amt, "model_id", None), "license": LICENSES["amt"], "commercial_use": True},
                "musicgen": {"available": self.audiogen is not None, "model": getattr(self.audiogen, "model_id", None), "license": LICENSES["musicgen"],
                             "commercial_use": False, "enabled_for_noncommercial": self.settings.allow_noncommercial, "experimental": True},
                "stable_audio_open": {"available": False, "license": LICENSES["stable_audio_open"], "reason": "gated weights (licence agreement required); not installed"},
            },
            "renderers": r,
            "analysis": {"beats": getattr(self.analyzer.beat_backend, "name", "numpy-dp"), "transcriber": getattr(self.analyzer.transcriber, "name", None),
                         "chords": "chroma-templates+hmm", "accepts_stems": True},
            "max_duration_seconds": self.settings.max_seconds,
        }

    # -------------------------------------------------------------- analysis
    def analyze(self, mix: np.ndarray | None, sr: int, stems: dict[str, np.ndarray] | None = None, melody: bool = True) -> dict[str, Any]:
        n = mix.shape[-1] if mix is not None else max(v.shape[-1] for v in (stems or {}).values())
        if n / sr > self.settings.max_seconds:
            raise UnsupportedOperation("audio too long for music analysis", details={"max_seconds": self.settings.max_seconds})
        with self._model_lock:  # transcriber / beat model are GPU models
            return self.analyzer.analyze(mix, sr, stems, melody=melody)

    # -------------------------------------------------------------- generation
    def generate(self, mix: np.ndarray | None, sr: int, stems: dict[str, np.ndarray] | None, req: PartRequest) -> PartResult:
        if not self.settings.enabled:
            raise UnsupportedOperation("part generation is disabled (AUDIO_GEN_ENABLED=false)")
        if req.engine not in ENGINES:
            raise InvalidRequest(f"engine must be one of {ENGINES}")
        if not (req.instrument or req.prompt):
            raise InvalidRequest("instrument or prompt is required")
        t_start = time.perf_counter()
        elapsed: dict[str, float] = {}
        stems = stems or {}
        if mix is None:
            if not stems:
                raise InvalidRequest("generate-part needs the song mix or its stems")
            n = max(v.shape[-1] for v in stems.values())
            mix = sum(np.pad(np.atleast_2d(v), ((0, 0), (0, n - v.shape[-1]))) for v in stems.values())
        mix = np.atleast_2d(mix).astype(np.float32)
        n_out = mix.shape[-1]
        analysis = req.analysis or self.analyze(mix, sr, stems)
        elapsed["analysis"] = round(time.perf_counter() - t_start, 3)
        spec = resolve(req.instrument, req.role, req.prompt, req.density, req.style)
        windows, warnings = _windows(analysis, req.sections, req.interval, req.prompt)
        warnings = list(analysis.get("warnings") or []) + warnings

        engine = req.engine
        if engine == "auto":
            # The rule-based arranger is the default and only path unless an experimental engine is
            # explicitly requested and enabled (listening tests: it beat AMT and MusicGen).
            engine = "symbolic"
            if spec.audio_generative_suggested:
                warnings.append("this sounds like a texture rather than an instrument part; rendered with the built-in synth")
        if engine == "amt" and self.amt is None:
            raise UnsupportedOperation("the AMT engine is disabled (experimental; AUDIO_GEN_AMT_MODEL)")
        if engine == "musicgen" and self.audiogen is None:
            raise UnsupportedOperation("the MusicGen engine is disabled (experimental, CC-BY-NC weights; AUDIO_GEN_AUDIOGEN=musicgen)")
        vocals = next((v for k, v in stems.items() if "vocal" in k.lower()), None)
        has_bass = any("bass" in k.lower() for k in stems)
        notes: list[PNote] = []
        renderer_info: dict[str, Any] | None = None
        t0 = time.perf_counter()
        if engine == "musicgen":
            part, engine_info = self._musicgen(mix, sr, n_out, spec, analysis, windows, req)
        else:
            if engine == "amt":
                notes, engine_info = self._amt(spec, analysis, windows, req)
            else:
                notes = compose(spec, analysis, windows, seed=req.seed, humanize=req.humanize)
                engine_info = {"name": "symbolic", "composer": "rule-based", "commercial_use": True}
            elapsed["compose"] = round(time.perf_counter() - t0, 3)
            t1 = time.perf_counter()
            part, used = self.renderers.render(spec.renderer, spec.fallback_renderer, notes, n_out, sr, tempo=float(analysis["tempo"]), articulation=spec.articulation, seed=req.seed)
            elapsed["render"] = round(time.perf_counter() - t1, 3)
            kind = used.split(":")[0]
            renderer_info = {"requested": spec.renderer, "used": used,
                             "license": {"vsco": self.renderers.sampler.license, "gm": self.renderers.gm.license, "synth": self.renderers.synth.license}[kind]}
            if used != spec.renderer:
                warnings.append(f"renderer {spec.renderer} unavailable; used {used}")
        if not np.any(part):
            warnings.append("the part is silent (no notes in the requested sections)")
        t2 = time.perf_counter()
        mixed, mix_report = mix_in(part, sr, mix, vocals, analysis, spec.role, windows, level_db=req.level_db, duck_db=req.duck_db, has_bass_stem=has_bass)
        elapsed["mix"] = round(time.perf_counter() - t2, 3)
        warnings += mix_report.pop("warnings", [])
        checks = {"audio": audio_checks(mixed, sr, analysis, mix)}
        if notes:
            checks["notes"] = note_checks(notes, analysis)
        midi = write_midi(notes, tempo=float(analysis["tempo"]), program=_gm_program(spec), name=spec.display_name) if notes else None
        elapsed["total"] = round(time.perf_counter() - t_start, 3)
        return PartResult(mixed, sr, midi, notes, spec, analysis, windows, engine_info, renderer_info, mix_report, checks, warnings, elapsed)

    def _amt(self, spec: PartSpec, analysis: dict[str, Any], windows, req: PartRequest) -> tuple[list[PNote], dict[str, Any]]:
        if self.amt is None:
            raise MissingModel("AMT composer is not installed/enabled (AUDIO_GEN_AMT_MODEL)")
        seed_part = compose(spec, analysis, windows, seed=req.seed, humanize=False)
        with self._model_lock:
            notes = self.amt.compose(analysis, _gm_program(spec), spec.low, spec.high, windows, seed=req.seed, prompt_notes=seed_part)
        # AMT works on a 10 ms clock, not the song's beat grid: snap onsets that are close to a
        # 16th-note grid point (<= 70 ms), keep the rest (deliberate pushes/anticipations).
        beats = np.asarray(analysis["beats"], dtype=np.float64)
        if len(beats) > 1:
            b = np.r_[beats, beats[-1] + np.median(np.diff(beats))]
            grid = np.array([b[i] + j * (b[i + 1] - b[i]) / 4 for i in range(len(b) - 1) for j in range(4)])
            for n in notes:
                j = int(np.argmin(np.abs(grid - n.start)))
                if abs(grid[j] - n.start) <= 0.07:
                    shift = grid[j] - n.start
                    n.start += shift
                    n.end = max(n.start + 0.05, n.end + shift)
        for n in notes:
            n.velocity = 72
        return notes, {"name": "amt", "model": self.amt.model_id, "license": LICENSES["amt"], "commercial_use": True}

    def _musicgen(self, mix, sr, n_out, spec, analysis, windows, req) -> tuple[np.ndarray, dict[str, Any]]:
        if self.audiogen is None:
            raise MissingModel("audio-generative engine not loaded (AUDIO_GEN_AUDIOGEN=musicgen)")
        if not self.settings.allow_noncommercial:
            raise UnsupportedOperation("MusicGen weights are CC-BY-NC-4.0; set AUDIO_GEN_ALLOW_NONCOMMERCIAL=true to use them", details={"license": LICENSES["musicgen"]})
        prompt = _audiogen_prompt(spec, analysis, req.prompt or req.instrument)
        part = np.zeros((2, n_out), np.float32)
        for a, b in windows:
            b = min(b, a + self.settings.audiogen_max_seconds)
            cond = mix[:, int(a * sr):int(b * sr)]
            with self._model_lock:
                y, ysr = self.audiogen.generate(prompt, cond, sr, b - a, seed=req.seed)
            y = dsp.resample(y, ysr, sr)
            s = int(a * sr)
            e = min(n_out, s + len(y))
            part[:, s:e] += y[: e - s]
        return part, {"name": "musicgen", "model": self.audiogen.model_id, "prompt": prompt, "license": LICENSES["musicgen"], "commercial_use": False, "experimental": True}


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
