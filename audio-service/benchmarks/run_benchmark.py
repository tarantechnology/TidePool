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
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import platform
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

from benchmarks.fixtures import QUALITY_FIXTURES, Fixture, build_all, long_mixture, write_fixture_set  # noqa: E402
from sam_audio_service.audio_io import DecodedAudio, decode, encode_wav_pcm16  # noqa: E402
from sam_audio_service.config import load_settings  # noqa: E402
from sam_audio_service.contracts import SeparationMetadata  # noqa: E402
from sam_audio_service.engine import Engine  # noqa: E402
from sam_audio_service.metrics import reconstruction_error, sdr, si_sdr, si_sdr_improvement, spectral_leakage_db  # noqa: E402

DEFAULT_OUT = ROOT / "benchmarks" / "out"
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
def _metadata(prompt: str) -> dict[str, Any]:
    """The request body. Shaped exactly like Person 4's backend sends it (prompts only)."""
    return {"prompts": [prompt]}


def _split_stems(names: list[str], roles: list[str]) -> tuple[int, int | None]:
    """Index of the target stem and of the residual stem (``None`` if there is none)."""
    target = next((i for i, role in enumerate(roles) if role == "target"), 0)
    residual = next((i for i, role in enumerate(roles) if role == "residual"), None)
    if residual is None:
        residual = next((i for i, name in enumerate(names) if name.lower() == "residual" and i != target), None)
    return target, residual


def run_engine(engine: Engine, fixture: Fixture, *, repeat: int) -> Outcome:
    mixture = fixture.mixture
    audio = DecodedAudio(samples=mixture, sample_rate=fixture.sample_rate, source_sample_rate=fixture.sample_rate, source_channels=fixture.channels)
    metadata = SeparationMetadata.model_validate(_metadata(fixture.prompt))
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


def run_http(url: str, fixture: Fixture, *, repeat: int, timeout: float) -> Outcome:
    import httpx

    wav = encode_wav_pcm16(fixture.mixture, fixture.sample_rate)
    payload = json.dumps(_metadata(fixture.prompt))
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
                 f"+ {report['settings']['latency_seconds']:g}s latency clip, speech source `{report['settings']['speech_mode']}`")
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
    parser.add_argument("--latency-seconds", type=float, default=30.0, help="length of the latency fixture; 0 skips it (default: 30)")
    parser.add_argument("--repeat", type=int, default=1, help="separations per fixture; latency is the median (default: 1)")
    parser.add_argument("--fixtures", default=",".join(QUALITY_FIXTURES), help="comma-separated quality fixtures to run")
    parser.add_argument("--speech-mode", choices=("auto", "say", "synthetic"), default="auto", help="'auto' uses macOS `say` when present (default)")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds (default: 600)")
    parser.add_argument("--write-fixtures", nargs="?", const="", default=None, metavar="DIR",
                        help="also write the fixture WAVs + manifest.json for other teams (default dir: fixtures/generated)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    names = tuple(name.strip() for name in args.fixtures.split(",") if name.strip())
    fixtures = build_all(args.seconds, speech_mode=args.speech_mode, names=names)
    if args.latency_seconds > 0:
        fixtures.append(long_mixture(args.latency_seconds, speech_mode=args.speech_mode))

    engine: Engine | None = None
    if args.url:
        backend_info: dict[str, Any] = {"mode": "http", "url": args.url, "name": None, "is_mock": None}
    else:
        settings = load_settings()
        settings.backend = args.backend
        settings.validate()
        try:
            engine = Engine(settings)
            engine.load()
        except Exception as exc:  # noqa: BLE001 - a gated checkpoint must read as a message, not a traceback
            print(f"could not load the {args.backend!r} backend: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("SAM Audio checkpoints are gated; see benchmarks/README.md. Use --backend mock for a pipeline check.", file=sys.stderr)
            return 2
        status = engine.status()
        backend_info = {"mode": "engine", "name": status.get("backend"), "is_mock": status.get("is_mock"),
                        "model_id": status.get("model_id"), "device": status.get("device"), "dtype": status.get("dtype")}

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for fixture in fixtures:
        print(f"-> {fixture.name}: prompt {fixture.prompt!r}, {fixture.duration:.1f}s, {fixture.channels}ch", flush=True)
        outcome = run_http(args.url, fixture, repeat=args.repeat, timeout=args.timeout) if args.url else run_engine(engine, fixture, repeat=args.repeat)  # type: ignore[arg-type]
        metrics = measure(fixture, outcome)
        for item in outcome.warnings:
            if item not in warnings:
                warnings.append(item)
        if backend_info.get("name") is None:  # HTTP mode: the service reports what it is
            backend_info.update({
                "name": outcome.model.get("backend"),
                "is_mock": outcome.model.get("is_mock"),
                "model_id": outcome.model.get("model_id"),
                "device": outcome.model.get("device"),
                "dtype": outcome.model.get("dtype"),
            })
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
        })
        print(f"   SI-SDR {_fmt(metrics['si_sdr_db'])} dB, SI-SDRi {_fmt(metrics['si_sdri_db'])} dB, "
              f"leakage {_fmt(metrics['leakage_db'])} dB, RTF {_fmt(metrics['real_time_factor'], 3)}", flush=True)

    command = "python -m benchmarks.run_benchmark " + " ".join(argv if argv is not None else sys.argv[1:])
    report = {
        "schema": "soundcanvas/audio-benchmark-1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command.strip(),
        "backend": backend_info,
        "is_mock": bool(backend_info.get("is_mock")),
        "mock_disclaimer": MOCK_DISCLAIMER if backend_info.get("is_mock") else None,
        "settings": {"seconds": args.seconds, "latency_seconds": args.latency_seconds, "repeat": args.repeat, "speech_mode": args.speech_mode},
        "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__},
        "fixtures": rows,
        "warnings": warnings,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    markdown = render_markdown(report)
    (out_dir / "REPORT.md").write_text(markdown)
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
