"""Audio separation benchmark: quality + latency, in-process or over HTTP.

Runs every fixture in :mod:`benchmarks.fixtures` for its documented prompt, measures the
target stem against ground truth and writes ``benchmarks/out/report.json`` and
``benchmarks/out/REPORT.md``.

Two execution modes, deliberately kept identical in everything else so a mock run and a
real run are comparable:

``--backend mock|sam``
    In-process through :class:`sam_audio_service.engine.Engine`. Latency is the engine
    call alone (no HTTP, no WAV encode).
``--url http://host:8001/separate``
    Over HTTP, posting the same multipart body Person 4's backend posts
    (``files={"audio": ...}``, ``data={"metadata": json}``). Latency therefore includes
    transfer, decode and WAV encode: it is the number the application actually sees.

Examples::

    python -m benchmarks.run_benchmark                                   # mock, defaults
    python -m benchmarks.run_benchmark --seconds 10 --repeat 3
    python -m benchmarks.run_benchmark --url http://127.0.0.1:8001/separate
    python -m benchmarks.run_benchmark --backend sam --latency-seconds 60
    python -m benchmarks.run_benchmark --backend sam --passes 2           # multi-pass SI-SDR
    python -m benchmarks.run_benchmark --backend sam --refine mask --refine-p 2
    python -m benchmarks.run_benchmark --backend sam --sweep quality-v1   # one table, many configs

Separation knobs (``--seed``, ``--candidates``, ``--window-candidates``, ``--select``,
``--refine*``, ``--chunk-seconds``) travel as ``metadata.options`` and are only sent when
given, so a plain run posts exactly what Person 4's backend posts. ``--chunk-overlap``,
``--ode-step`` and ``--model-id`` change the in-process engine and are ignored with ``--url``.
``--sweep`` loads the engine once and runs every fixture under each config of a named sweep
(:mod:`benchmarks.sweeps`) or a JSON file, writing one report per config plus ``SWEEP.md``.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import platform
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python benchmarks/run_benchmark.py`
    sys.path.insert(0, str(ROOT))

from benchmarks.fixtures import _BUILDERS, LONG_FIXTURES, QUALITY_FIXTURES, Fixture, build_all, long_mixture, write_fixture_set  # noqa: E402
from sam_audio_service.audio_io import DecodedAudio, decode, encode_wav_pcm16  # noqa: E402
from sam_audio_service.config import load_settings  # noqa: E402
from sam_audio_service.contracts import SeparationMetadata  # noqa: E402
from sam_audio_service.engine import Engine  # noqa: E402
from sam_audio_service.metrics import reconstruction_error, sdr, si_sdr, si_sdr_improvement, spectral_leakage_db  # noqa: E402

DEFAULT_OUT = ROOT / "benchmarks" / "out"
DEFAULT_SEED = 1234  # sent as options.seed whenever the backend is the real model (in-process sam)

# Config key (= CLI dest) -> metadata.options key. Sent only when not None.
REQUEST_OPTIONS = {
    "seed": "seed",
    "candidates": "reranking_candidates",
    "window_candidates": "candidates",
    "select": "candidate_select",
    "refine": "refine",
    "refine_p": "refine_p",
    "refine_nfft": "refine_n_fft",
    "refine_floor": "refine_floor",
    "refine_blend": "refine_blend",
    "refine_threshold": "refine_threshold",
    "refine_gate_db": "refine_gate_db",
    "chunk_seconds": "chunk_seconds",
}
# Config keys that change the in-process engine rather than the request.
IN_PROCESS_ONLY = ("chunk_overlap", "ode_step", "model_id")
CONFIG_KEYS = ("passes", *REQUEST_OPTIONS, *IN_PROCESS_ONLY)
CHOICES = {"select": ("first", "overlap", "mean"), "refine": ("off", "mask")}
MOCK_DISCLAIMER = (
    "These numbers come from the MOCK backend, which is a prompt-selected band-pass filter, "
    "not source separation. They verify that the pipeline, contract, stem roles, residual and "
    "timing all work end to end. They are NOT quality claims: do not quote them as SAM Audio "
    "separation quality."
)


# ---------------------------------------------------------------------------- results
@dataclass
class Outcome:
    """Normalised result of one separation, whichever mode produced it."""

    target: np.ndarray  # (channels, frames) float32
    residual: np.ndarray | None
    model: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    stem_names: list[str] = field(default_factory=list)

    @property
    def latency(self) -> float:
        """Median wall-clock latency across repeats (median resists one cold first run)."""
        return float(np.median(self.latencies)) if self.latencies else math.nan


# ------------------------------------------------------------------------------ modes
def _metadata(prompt: str, passes: int = 1, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """The request body. Shaped exactly like Person 4's backend sends it (prompts only);
    ``--passes`` > 1 adds ``options.passes`` (multi-pass separation) and ``options`` is
    merged into ``metadata.options``. No ``options`` key at all when there is nothing to send."""
    merged: dict[str, Any] = {"passes": passes} if passes > 1 else {}
    merged.update(options or {})
    if merged:
        return {"prompts": [prompt], "options": merged}
    return {"prompts": [prompt]}


def request_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """``metadata.options`` for a config: only the knobs that were actually set."""
    return {REQUEST_OPTIONS[key]: value for key, value in cfg.items() if key in REQUEST_OPTIONS and value is not None}


def _split_stems(names: list[str], roles: list[str]) -> tuple[int, int | None]:
    """Index of the target stem and of the residual stem (``None`` if there is none)."""
    target = next((i for i, role in enumerate(roles) if role == "target"), 0)
    residual = next((i for i, role in enumerate(roles) if role == "residual"), None)
    if residual is None:
        residual = next((i for i, name in enumerate(names) if name.lower() == "residual" and i != target), None)
    return target, residual


def run_engine(engine: Engine, fixture: Fixture, *, repeat: int, passes: int = 1, options: dict[str, Any] | None = None) -> Outcome:
    mixture = fixture.mixture
    audio = DecodedAudio(samples=mixture, sample_rate=fixture.sample_rate, source_sample_rate=fixture.sample_rate, source_channels=fixture.channels)
    metadata = SeparationMetadata.model_validate(_metadata(fixture.prompt, passes, options))
    latencies: list[float] = []
    result = None
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        result = engine.separate(audio, metadata)
        latencies.append(time.perf_counter() - started)
    assert result is not None
    names = [stem.name for stem in result.stems]
    target, residual = _split_stems(names, [stem.role for stem in result.stems])
    return Outcome(
        target=result.stems[target].samples,
        residual=result.stems[residual].samples if residual is not None else None,
        model=dict(result.model),
        warnings=list(result.warnings),
        latencies=latencies,
        stem_names=names,
    )


def run_http(url: str, fixture: Fixture, *, repeat: int, timeout: float, passes: int = 1, options: dict[str, Any] | None = None) -> Outcome:
    import httpx

    wav = encode_wav_pcm16(fixture.mixture, fixture.sample_rate)
    payload = json.dumps(_metadata(fixture.prompt, passes, options))
    latencies: list[float] = []
    body: dict[str, Any] = {}
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        response = httpx.post(url, files={"audio": ("audio.wav", wav, "audio/wav")}, data={"metadata": payload}, timeout=timeout)
        latencies.append(time.perf_counter() - started)
        response.raise_for_status()
        body = response.json()
    stems = body.get("stems") or []
    if not stems:
        raise RuntimeError(f"{url} returned no stems: {json.dumps(body)[:400]}")
    names = [stem.get("name", "") for stem in stems]
    target, residual = _split_stems(names, [stem.get("role", "") for stem in stems])

    def samples(stem: dict[str, Any]) -> np.ndarray:
        encoded = stem.get("audio_base64")
        if not encoded:
            raise RuntimeError(f"stem {stem.get('name')!r} has no audio_base64; run the service with return_mode=base64")
        return decode(data=base64.b64decode(encoded, validate=True), target_rate=fixture.sample_rate).samples

    return Outcome(
        target=samples(stems[target]),
        residual=samples(stems[residual]) if residual is not None else None,
        model=dict(body.get("model") or {}),
        warnings=list(body.get("warnings") or []),
        latencies=latencies,
        stem_names=names,
    )


# ---------------------------------------------------------------------------- metrics
def measure(fixture: Fixture, outcome: Outcome) -> dict[str, Any]:
    """All quality metrics for one fixture/outcome pair, plus latency and RTF."""
    reference, interference, mixture = fixture.target_signal, fixture.interference, fixture.mixture
    stems = [outcome.target] + ([outcome.residual] if outcome.residual is not None else [])
    duration = fixture.duration
    latency = outcome.latency
    return {
        "si_sdr_db": si_sdr(outcome.target, reference),
        "si_sdr_input_db": si_sdr(mixture, reference),
        "si_sdri_db": si_sdr_improvement(outcome.target, reference, mixture),
        "sdr_db": sdr(outcome.target, reference),
        "leakage_db": spectral_leakage_db(outcome.target, reference, interference),
        "reconstruction_error": reconstruction_error(stems, mixture),
        "has_residual": outcome.residual is not None,
        "latency_seconds": latency,
        "latency_seconds_all": [round(value, 4) for value in outcome.latencies],
        "real_time_factor": latency / duration if duration > 0 and not math.isnan(latency) else math.nan,
        "repeats": len(outcome.latencies),
    }


# ----------------------------------------------------------------------------- report
def _json_safe(value: Any) -> Any:
    """``inf``/``nan`` are not valid JSON; they become ``null`` and keep their meaning in MD."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    return value


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        if abs(value) < 1e-3 and value != 0.0:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return "n/a"
    return f"{value / 2**20:.0f} MB"


def render_markdown(report: dict[str, Any]) -> str:
    backend = report["backend"]
    lines = ["# SoundCanvas audio separation benchmark", ""]
    lines.append(f"- **Backend**: `{backend.get('name')}` ({backend['mode']} mode)"
                 + (f" at `{backend['url']}`" if backend.get("url") else ""))
    lines.append(f"- **Mock backend**: **{'YES' if backend.get('is_mock') else 'no'}**")
    lines.append(f"- **Model**: `{backend.get('model_id')}` on `{backend.get('device')}` ({backend.get('dtype')})")
    lines.append(f"- **Generated**: {report['generated_at']} on {report['environment']['platform']}, "
                 f"python {report['environment']['python']}, numpy {report['environment']['numpy']}")
    lines.append(f"- **Fixtures**: synthetic, deterministic, {report['settings']['seconds']:g}s quality clips "
                 f"+ {report['settings'].get('long_seconds', 0):g}s long clips "
                 f"+ {report['settings']['latency_seconds']:g}s latency clip, speech source `{report['settings']['speech_mode']}`")
    knobs = {key: report["settings"].get(key) for key in CONFIG_KEYS if report["settings"].get(key) is not None}
    lines.append(f"- **Config**: `{report['settings'].get('config_name') or 'cli'}` -- "
                 + (", ".join(f"{key}={value}" for key, value in knobs.items()) or "defaults")
                 + f"; request options `{json.dumps(report['settings'].get('request_options') or {})}`")
    if report["settings"].get("effective_model_id") is not None:
        lines.append(f"- **Engine**: model `{report['settings']['effective_model_id']}`, ODE step `{report['settings']['effective_ode_step']}`, "
                     f"chunk overlap `{report['settings']['effective_chunk_overlap']}`s; "
                     f"peak RSS {_fmt_bytes(report['settings'].get('peak_rss_bytes'))}")
    lines.append("")
    if backend.get("is_mock"):
        lines += [f"> **MOCK RUN -- NOT A QUALITY RESULT.** {MOCK_DISCLAIMER}", ""]
    lines += [
        "| Fixture | Prompt | Target | Dur (s) | SI-SDR (dB) | input SI-SDR (dB) | SI-SDRi (dB) | SDR (dB) | Leakage (dB) | Recon err | Latency (s) | RTF (proc/audio) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["fixtures"]:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['fixture']}` | `{row['prompt']}` | `{row['target_source']}` | {row['duration']:.2f} "
            f"| {_fmt(metrics['si_sdr_db'])} | {_fmt(metrics['si_sdr_input_db'])} | {_fmt(metrics['si_sdri_db'])} "
            f"| {_fmt(metrics['sdr_db'])} | {_fmt(metrics['leakage_db'])} | {_fmt(metrics['reconstruction_error'])} "
            f"| {_fmt(metrics['latency_seconds'], 3)} | {_fmt(metrics['real_time_factor'], 3)} |"
        )
    latency_row = next((row for row in report["fixtures"] if row["fixture"].startswith("long_mixture")), None)
    lines.append("")
    if latency_row:
        rtf = latency_row["metrics"]["real_time_factor"]
        lines.append(f"**Latency reference** (`{latency_row['fixture']}`, {latency_row['duration']:.1f}s mono, "
                     f"{latency_row['metrics']['repeats']} run(s)): {_fmt(latency_row['metrics']['latency_seconds'], 3)} s "
                     f"= **{_fmt(rtf, 3)} s of processing per second of audio**"
                     + (f" ({_fmt(1.0 / rtf, 1)}x faster than real time)" if isinstance(rtf, float) and math.isfinite(rtf) and rtf > 0 else "") + ".")
        lines.append("")
    lines += ["## What the fixtures are", ""]
    for row in report["fixtures"]:
        lines.append(f"- **`{row['fixture']}`** -- prompt `{row['prompt']}`, ground-truth target `{row['target_source']}`, "
                     f"{row['channels']}ch, speech source `{row['speech_source']}`. {row['notes']}")
    lines += ["", "## Metrics", "",
              "- **SI-SDR** -- scale-invariant signal-to-distortion ratio of the target stem against ground truth. Higher is better; above ~10 dB the wanted sound clearly dominates.",
              "- **input SI-SDR** -- the same measure on the untouched mixture: the score for doing nothing.",
              "- **SI-SDRi** -- improvement over that baseline. **This is the headline number.** <= 0 dB means the stem is no better than the mixture.",
              "- **SDR** -- scale-dependent variant; punishes level errors, which matter when the stem is dropped into the editor's mix.",
              "- **Leakage** -- energy of non-target sound in the target stem relative to the wanted sound, in dB. Lower is better; 0 dB means half the stem is the wrong sound.",
              "- **Recon err** -- max absolute sample error of `target + residual - mixture`. ~1e-7 is float round-off, ~1e-4 is 16-bit WAV quantisation over HTTP; larger means a pipeline bug.",
              "- **RTF** -- seconds of processing per second of audio (lower is faster; < 1 is faster than real time).",
              ""]
    warnings = report.get("warnings") or []
    if warnings:
        lines += ["## Service warnings", ""] + [f"- {item}" for item in warnings] + [""]
    lines += ["## Reproduce", "", "```sh", f"cd audio-service && {report['command']}", "```", ""]
    lines.append("See `benchmarks/README.md` for how to run this against a real SAM Audio service.")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarks.run_benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("mock", "sam"), default="mock", help="in-process backend (default: mock)")
    parser.add_argument("--url", default=None, help="instead of running in-process, POST to this /separate endpoint")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"report directory (default: {DEFAULT_OUT})")
    parser.add_argument("--seconds", type=float, default=6.0, help="length of each quality fixture (default: 6)")
    parser.add_argument("--long-seconds", type=float, default=24.0, help=f"length of the long fixtures ({', '.join(LONG_FIXTURES)}); 0 skips them (default: 24)")
    parser.add_argument("--latency-seconds", type=float, default=30.0, help="length of the latency fixture; 0 skips it (default: 30)")
    parser.add_argument("--repeat", type=int, default=1, help="separations per fixture; latency is the median (default: 1)")
    parser.add_argument("--fixtures", default=",".join(QUALITY_FIXTURES + LONG_FIXTURES), help="comma-separated quality and long fixtures to run")
    parser.add_argument("--speech-mode", choices=("auto", "say", "synthetic"), default="auto", help="'auto' uses macOS `say` when present (default)")
    parser.add_argument("--passes", type=int, default=1, help="options.passes: rerun each prompt on its residual this many times (default: 1); "
                        "compare SI-SDR across runs with --passes 1/2/3")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds (default: 600)")
    parser.add_argument("--write-fixtures", nargs="?", const="", default=None, metavar="DIR",
                        help="also write the fixture WAVs + manifest.json for other teams (default dir: fixtures/generated)")

    knobs = parser.add_argument_group("separation settings (sent as metadata.options only when given)")
    knobs.add_argument("--seed", type=int, default=None, help=f"options.seed (default: {DEFAULT_SEED} for in-process --backend sam, unset otherwise)")
    knobs.add_argument("--candidates", type=int, default=None, help="options.reranking_candidates (model-level re-ranking)")
    knobs.add_argument("--window-candidates", type=int, default=None, help="options.candidates: candidate separations per chunk window")
    knobs.add_argument("--select", choices=CHOICES["select"], default=None, help="options.candidate_select: how window candidates are chosen")
    knobs.add_argument("--refine", choices=CHOICES["refine"], default=None, help="options.refine: post-separation refinement")
    knobs.add_argument("--refine-p", type=float, default=None, help="options.refine_p: mask exponent")
    knobs.add_argument("--refine-nfft", type=int, default=None, help="options.refine_n_fft: STFT size of the refinement mask")
    knobs.add_argument("--refine-floor", type=float, default=None, help="options.refine_floor: minimum mask value")
    knobs.add_argument("--refine-blend", type=float, default=None, help="options.refine_blend: 1 = fully refined, 0 = raw model output")
    knobs.add_argument("--refine-threshold", type=float, default=None, help="options.refine_threshold (0..1): mask values below this are zeroed")
    knobs.add_argument("--refine-gate-db", type=float, default=None, help="options.refine_gate_db (0..80): gate bins this far below the stem peak")
    knobs.add_argument("--chunk-seconds", type=float, default=None, help="options.chunk_seconds: chunk window length, 0 = whole file")

    engine_group = parser.add_argument_group("in-process engine settings (ignored with --url)")
    engine_group.add_argument("--chunk-overlap", type=float, default=None, help="settings.chunk_overlap_seconds")
    engine_group.add_argument("--ode-step", type=float, default=None, help="ODE step size (backend ode_opt, else settings.ode_step_size)")
    engine_group.add_argument("--model-id", default=None, help="settings.model_id before the engine is built")

    parser.add_argument("--sweep", default=None, metavar="NAME|FILE.json",
                        help="run every config of a named sweep (benchmarks/sweeps.py) or a JSON list of configs; "
                             "loads the engine once and writes <out>/<config>/ + <out>/SWEEP.md")
    return parser


def peak_rss_bytes() -> int:
    """Peak resident set size of this process so far, in bytes.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux; normalised here. It is a
    process-lifetime high-water mark, so in a sweep each config reports the peak *up to and
    including* that config (it can only grow).
    """
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = {key: getattr(args, key) for key in CONFIG_KEYS}
    if cfg["seed"] is None and not args.url and args.backend == "sam":
        cfg["seed"] = DEFAULT_SEED
    return cfg


def validate_config(cfg: dict[str, Any], *, url: str | None, label: str = "") -> dict[str, Any]:
    """Check a config's values; with ``--url``, warn about and drop in-process-only knobs."""
    prefix = f"{label}: " if label else ""
    unknown = sorted(set(cfg) - set(CONFIG_KEYS) - {"name"})
    if unknown:
        raise SystemExit(f"{prefix}unknown config key(s) {unknown}; allowed: {list(CONFIG_KEYS)}")
    passes = cfg.get("passes") or 1
    if not isinstance(passes, int) or not 1 <= passes <= 5:
        raise SystemExit(f"{prefix}--passes must be between 1 and 5")
    for key, choices in CHOICES.items():
        if cfg.get(key) is not None and cfg[key] not in choices:
            raise SystemExit(f"{prefix}{key} must be one of {choices}, got {cfg[key]!r}")
    if url:
        ignored = [key for key in IN_PROCESS_ONLY if cfg.get(key) is not None]
        if ignored:
            print(f"warning: {prefix}{', '.join('--' + key.replace('_', '-') for key in ignored)} only apply in-process; ignored with --url", file=sys.stderr)
            cfg = {**cfg, **{key: None for key in ignored}}
    return cfg


def build_fixtures(args: argparse.Namespace) -> list[Fixture]:
    names = tuple(name.strip() for name in args.fixtures.split(",") if name.strip())
    unknown = [name for name in names if name not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown fixture(s) {unknown}; available: {sorted(_BUILDERS)}")
    fixtures = build_all(args.seconds, speech_mode=args.speech_mode, names=tuple(name for name in names if name not in LONG_FIXTURES))
    if args.long_seconds > 0:
        fixtures += [_BUILDERS[name](args.long_seconds, speech_mode=args.speech_mode) for name in names if name in LONG_FIXTURES]
    if args.latency_seconds > 0:
        fixtures.append(long_mixture(args.latency_seconds, speech_mode=args.speech_mode))
    return fixtures


# ------------------------------------------------------------------------------ engine
def _effective(engine: Engine | None) -> dict[str, Any]:
    if engine is None:
        return {"effective_model_id": None, "effective_ode_step": None, "effective_chunk_overlap": None}
    ode_opt = getattr(engine.backend, "ode_opt", None)
    step = ode_opt["options"]["step_size"] if isinstance(ode_opt, dict) else engine.settings.ode_step_size
    return {"effective_model_id": engine.settings.model_id, "effective_ode_step": step,
            "effective_chunk_overlap": engine.settings.chunk_overlap_seconds}


def apply_engine_config(engine: Engine, cfg: dict[str, Any], defaults: dict[str, Any]) -> None:
    """Point a loaded engine at ``cfg``'s in-process knobs (unset knobs go back to ``defaults``)."""
    settings = engine.settings
    settings.chunk_overlap_seconds = cfg.get("chunk_overlap") or defaults["chunk_overlap"]
    step = cfg.get("ode_step") or defaults["ode_step"]
    settings.ode_step_size = step
    ode_opt = getattr(engine.backend, "ode_opt", None)
    if isinstance(ode_opt, dict):  # the real backend copied the step at construction; change it live
        ode_opt.setdefault("options", {})["step_size"] = step
    try:
        settings.validate()
    except ValueError as exc:
        raise SystemExit(f"invalid engine settings: {exc}") from exc


def load_engine(args: argparse.Namespace, cfg: dict[str, Any], *, max_passes: int) -> tuple[Engine, dict[str, Any]]:
    """Build and load the in-process engine for ``cfg``. Returns it and the env defaults."""
    settings = load_settings()
    settings.backend = args.backend
    defaults = {"chunk_overlap": settings.chunk_overlap_seconds, "ode_step": settings.ode_step_size}
    settings.max_passes = max(settings.max_passes, max_passes)  # in-process: lift AUDIO_MAX_PASSES for the run
    if cfg.get("model_id"):
        settings.model_id = cfg["model_id"]
    if cfg.get("ode_step"):
        settings.ode_step_size = cfg["ode_step"]  # the backend reads it at construction
    if cfg.get("chunk_overlap"):
        settings.chunk_overlap_seconds = cfg["chunk_overlap"]
    try:
        settings.validate()
    except ValueError as exc:
        raise SystemExit(f"invalid engine settings: {exc}") from exc
    engine = Engine(settings)
    engine.load()
    apply_engine_config(engine, cfg, defaults)
    return engine, defaults


def _backend_info(engine: Engine | None, url: str | None) -> dict[str, Any]:
    if engine is None:
        return {"mode": "http", "url": url, "name": None, "is_mock": None}
    status = engine.status()
    return {"mode": "engine", "name": status.get("backend"), "is_mock": status.get("is_mock"),
            "model_id": status.get("model_id"), "device": status.get("device"), "dtype": status.get("dtype")}


# ---------------------------------------------------------------------------- running
Runner = Any  # callable(fixture, *, passes, options) -> Outcome


def make_runner(args: argparse.Namespace, engine: Engine | None) -> Runner:
    if args.url:
        return lambda fixture, *, passes, options: run_http(args.url, fixture, repeat=args.repeat, timeout=args.timeout, passes=passes, options=options)
    return lambda fixture, *, passes, options: run_engine(engine, fixture, repeat=args.repeat, passes=passes, options=options)  # type: ignore[arg-type]


def run_config(runner: Runner, fixtures: list[Fixture], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every fixture under one config and return the report rows."""
    options = request_options(cfg)
    passes = cfg.get("passes") or 1
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        print(f"-> {fixture.name}: prompt {fixture.prompt!r}, {fixture.duration:.1f}s, {fixture.channels}ch", flush=True)
        outcome = runner(fixture, passes=passes, options=options)
        metrics = measure(fixture, outcome)
        rows.append({
            "fixture": fixture.name,
            "prompt": fixture.prompt,
            "target_source": fixture.target,
            "sources": list(fixture.sources),
            "duration": round(fixture.duration, 6),
            "channels": fixture.channels,
            "speech_source": fixture.speech_source,
            "notes": fixture.notes,
            "stems": outcome.stem_names,
            "metrics": metrics,
            "model": outcome.model,
            "warnings": list(outcome.warnings),
        })
        print(f"   SI-SDR {_fmt(metrics['si_sdr_db'])} dB, SI-SDRi {_fmt(metrics['si_sdri_db'])} dB, "
              f"leakage {_fmt(metrics['leakage_db'])} dB, RTF {_fmt(metrics['real_time_factor'], 3)}", flush=True)
    return rows


def build_report(args: argparse.Namespace, argv: list[str] | None, backend_info: dict[str, Any], rows: list[dict[str, Any]],
                 cfg: dict[str, Any], engine: Engine | None, *, name: str | None = None) -> dict[str, Any]:
    info = dict(backend_info)
    if info.get("name") is None and rows:  # HTTP mode: the service reports what it is
        model = rows[0]["model"]
        info.update({key: model.get(key) for key in ("model_id", "device", "dtype")})
        info.update({"name": model.get("backend"), "is_mock": model.get("is_mock")})
    warnings: list[str] = []
    for row in rows:
        for item in row.get("warnings", []):
            if item not in warnings:
                warnings.append(item)
    command = "python -m benchmarks.run_benchmark " + " ".join(argv if argv is not None else sys.argv[1:])
    settings = {"seconds": args.seconds, "long_seconds": args.long_seconds, "latency_seconds": args.latency_seconds,
                "repeat": args.repeat, "speech_mode": args.speech_mode, "config_name": name,
                **{key: cfg.get(key) for key in CONFIG_KEYS}, "passes": cfg.get("passes") or 1,
                "request_options": request_options(cfg), **_effective(engine), "peak_rss_bytes": peak_rss_bytes()}
    return {
        "schema": "soundcanvas/audio-benchmark-1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command.strip(),
        "backend": info,
        "is_mock": bool(info.get("is_mock")),
        "mock_disclaimer": MOCK_DISCLAIMER if info.get("is_mock") else None,
        "settings": settings,
        "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__},
        "fixtures": rows,
        "warnings": warnings,
    }


def write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    (out_dir / "REPORT.md").write_text(render_markdown(report))


# ------------------------------------------------------------------------------ sweep
def load_sweep(spec: str) -> tuple[str, list[dict[str, Any]]]:
    """A named sweep from :mod:`benchmarks.sweeps` or a JSON file (a list of configs, or
    ``{"name": ..., "configs": [...]}``). Each config is a dict of CLI dest names + ``name``."""
    from benchmarks.sweeps import SWEEPS

    if spec in SWEEPS:
        name, configs = spec, SWEEPS[spec]
    else:
        path = Path(spec)
        if not path.is_file():
            raise SystemExit(f"--sweep {spec!r} is neither a named sweep ({', '.join(sorted(SWEEPS))}) nor a JSON file")
        data = json.loads(path.read_text())
        name, configs = (data.get("name") or path.stem, data.get("configs")) if isinstance(data, dict) else (path.stem, data)
    if not isinstance(configs, list) or not configs or not all(isinstance(item, dict) for item in configs):
        raise SystemExit(f"--sweep {spec!r}: expected a non-empty list of config objects")
    return name, [dict(item) for item in configs]


def _is_quality_row(row: dict[str, Any]) -> bool:
    return not row["fixture"].startswith("long_mixture")


def _mean(values: list[Any]) -> float | None:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def _delta(value: Any, base: Any) -> float | None:
    if isinstance(value, (int, float)) and isinstance(base, (int, float)) and math.isfinite(value) and math.isfinite(base):
        return float(value) - float(base)
    return None


def summarise_config(name: str, directory: str, report: dict[str, Any]) -> dict[str, Any]:
    quality = [row for row in report["fixtures"] if _is_quality_row(row)]
    latency = next((row for row in report["fixtures"] if not _is_quality_row(row)), None)
    return {
        "name": name,
        "dir": directory,
        "settings": {key: report["settings"].get(key) for key in CONFIG_KEYS if report["settings"].get(key) is not None},
        "request_options": report["settings"]["request_options"],
        "fixtures": {row["fixture"]: {"si_sdri_db": row["metrics"]["si_sdri_db"], "si_sdr_db": row["metrics"]["si_sdr_db"],
                                      "leakage_db": row["metrics"]["leakage_db"]} for row in quality},
        "mean_si_sdri_db": _mean([row["metrics"]["si_sdri_db"] for row in quality]),
        "mean_leakage_db": _mean([row["metrics"]["leakage_db"] for row in quality]),
        "rtf": latency["metrics"]["real_time_factor"] if latency else None,
        "peak_rss_bytes": report["settings"]["peak_rss_bytes"],
    }


def add_deltas(configs: list[dict[str, Any]]) -> None:
    """Δ vs the first config (the baseline): SI-SDRi up is better, leakage down is better."""
    base = configs[0]
    for entry in configs:
        delta: dict[str, Any] = {"mean_si_sdri_db": _delta(entry["mean_si_sdri_db"], base["mean_si_sdri_db"]),
                                 "mean_leakage_db": _delta(entry["mean_leakage_db"], base["mean_leakage_db"])}
        for fixture, metrics in entry["fixtures"].items():
            reference = base["fixtures"].get(fixture, {})
            delta[fixture] = {"si_sdri_db": _delta(metrics["si_sdri_db"], reference.get("si_sdri_db")),
                              "leakage_db": _delta(metrics["leakage_db"], reference.get("leakage_db"))}
        entry["delta"] = delta


def _with_delta(value: Any, delta: Any, is_baseline: bool) -> str:
    if is_baseline or delta is None:
        return _fmt(value)
    return f"{_fmt(value)} ({delta:+.2f})"


def render_sweep_markdown(sweep: dict[str, Any]) -> str:
    configs = sweep["configs"]
    fixtures = list(dict.fromkeys(name for entry in configs for name in entry["fixtures"]))
    backend = sweep.get("backend") or {}
    lines = [f"# Separation settings sweep: `{sweep['name']}`", ""]
    lines.append(f"- **Backend**: `{backend.get('name')}` ({backend.get('mode')} mode), model `{backend.get('model_id')}` on `{backend.get('device')}`")
    lines.append(f"- **Generated**: {sweep['generated_at']}; baseline (Δ reference) is the first row, `{configs[0]['name']}`")
    lines.append("- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** "
                 "(dB of non-target sound inside the stem; the 'faint background still audible' complaint).")
    lines.append("- Peak RSS is the process high-water mark up to that config (it never goes down).")
    lines.append("")
    if sweep.get("is_mock"):
        lines += [f"> **MOCK RUN -- NOT A QUALITY RESULT.** {MOCK_DISCLAIMER}", ""]
    header = ["Config"]
    for name in fixtures:
        header += [f"`{name}` SI-SDRi", f"`{name}` leak"]
    header += ["Mean SI-SDRi", "Δ mean SI-SDRi", "Mean leak", "Δ mean leak", "RTF", "Peak RSS"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, entry in enumerate(configs):
        baseline = index == 0
        cells = [f"`{entry['name']}`"]
        for name in fixtures:
            metrics = entry["fixtures"].get(name, {})
            delta = entry["delta"].get(name, {})
            cells += [_with_delta(metrics.get("si_sdri_db"), delta.get("si_sdri_db"), baseline),
                      _with_delta(metrics.get("leakage_db"), delta.get("leakage_db"), baseline)]
        dmean, dleak = entry["delta"]["mean_si_sdri_db"], entry["delta"]["mean_leakage_db"]
        cells += [_fmt(entry["mean_si_sdri_db"]), "--" if baseline or dmean is None else f"{dmean:+.2f}",
                  _fmt(entry["mean_leakage_db"]), "--" if baseline or dleak is None else f"{dleak:+.2f}",
                  _fmt(entry["rtf"], 3), _fmt_bytes(entry["peak_rss_bytes"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Configs", ""]
    for entry in configs:
        knobs = ", ".join(f"{key}={value}" for key, value in entry["settings"].items()) or "defaults"
        lines.append(f"- **`{entry['name']}`** -- {knobs}; request options `{json.dumps(entry['request_options'])}`; "
                     f"full report `{entry['dir']}/REPORT.md`")
    lines += ["", "## Reproduce", "", "```sh", f"cd audio-service && {sweep['command']}", "```", ""]
    return "\n".join(lines) + "\n"


def _safe_dirname(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "config"
    candidate, counter = base, 2
    while candidate in used:
        candidate, counter = f"{base}_{counter}", counter + 1
    used.add(candidate)
    return candidate


def run_sweep(args: argparse.Namespace, argv: list[str] | None, fixtures: list[Fixture], base_cfg: dict[str, Any]) -> int:
    sweep_name, raw_configs = load_sweep(args.sweep)
    configs = []
    for index, raw in enumerate(raw_configs):
        name = str(raw.get("name") or f"config{index}")
        cfg = validate_config({**base_cfg, **{k: v for k, v in raw.items() if k != "name"}}, url=args.url, label=name)
        configs.append((name, cfg))

    engine: Engine | None = None
    defaults: dict[str, Any] = {}
    if not args.url:
        max_passes = max(cfg.get("passes") or 1 for _, cfg in configs)
        try:
            engine, defaults = load_engine(args, configs[0][1], max_passes=max_passes)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"could not load the {args.backend!r} backend: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    out_dir = Path(args.out)
    used: set[str] = set()
    summaries: list[dict[str, Any]] = []
    backend_info: dict[str, Any] = {}
    for name, cfg in configs:
        print(f"\n=== sweep {sweep_name}: config {name} ===", flush=True)
        if engine is not None:
            if cfg.get("model_id") and cfg["model_id"] != engine.settings.model_id:
                # A different checkpoint cannot be swapped in place: this is the one reload.
                engine, defaults = load_engine(args, cfg, max_passes=engine.settings.max_passes)
            apply_engine_config(engine, cfg, defaults)
        rows = run_config(make_runner(args, engine), fixtures, cfg)
        report = build_report(args, argv, _backend_info(engine, args.url), rows, cfg, engine, name=name)
        backend_info = report["backend"]
        directory = _safe_dirname(name, used)
        write_report(out_dir / directory, report)
        summaries.append(summarise_config(name, directory, report))
    add_deltas(summaries)
    command = "python -m benchmarks.run_benchmark " + " ".join(argv if argv is not None else sys.argv[1:])
    sweep = {
        "schema": "soundcanvas/audio-sweep-1",
        "name": sweep_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command.strip(),
        "backend": backend_info,
        "is_mock": bool(backend_info.get("is_mock")),
        "configs": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(_json_safe(sweep), indent=2) + "\n")
    (out_dir / "SWEEP.md").write_text(render_sweep_markdown(sweep))
    print(f"\nsweep {sweep_name}: {len(summaries)} configs")
    for entry in summaries:
        print(f"  {entry['name']:<28} mean SI-SDRi {_fmt(entry['mean_si_sdri_db'])} dB ({_fmt(entry['delta']['mean_si_sdri_db'])}), "
              f"mean leakage {_fmt(entry['mean_leakage_db'])} dB ({_fmt(entry['delta']['mean_leakage_db'])})")
    print(f"wrote {out_dir / 'sweep.json'}\nwrote {out_dir / 'SWEEP.md'}")
    if sweep["is_mock"]:
        print(f"\nNOTE: {MOCK_DISCLAIMER}")
    return 0


# ------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if not 1 <= args.passes <= 5:
        raise SystemExit("--passes must be between 1 and 5")
    base_cfg = config_from_args(args)
    fixtures = build_fixtures(args)
    if args.sweep:
        return run_sweep(args, argv, fixtures, base_cfg)
    cfg = validate_config(base_cfg, url=args.url)

    engine: Engine | None = None
    if not args.url:
        try:
            engine, _ = load_engine(args, cfg, max_passes=args.passes)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - a gated checkpoint must read as a message, not a traceback
            print(f"could not load the {args.backend!r} backend: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("SAM Audio checkpoints are gated; see benchmarks/README.md. Use --backend mock for a pipeline check.", file=sys.stderr)
            return 2

    rows = run_config(make_runner(args, engine), fixtures, cfg)
    report = build_report(args, argv, _backend_info(engine, args.url), rows, cfg, engine)
    out_dir = Path(args.out)
    write_report(out_dir, report)
    if args.write_fixtures is not None:
        from benchmarks.fixtures import DEFAULT_OUT_DIR

        manifest = write_fixture_set(args.write_fixtures or DEFAULT_OUT_DIR, seconds=args.seconds, speech_mode=args.speech_mode, latency_seconds=args.latency_seconds)
        print(f"fixtures written: {manifest}")
    print(f"\nwrote {out_dir / 'report.json'}\nwrote {out_dir / 'REPORT.md'}")
    if report["is_mock"]:
        print(f"\nNOTE: {MOCK_DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
