"""Fixture and benchmark-runner tests.

Everything runs on <= 2 s of audio with ``speech_mode="synthetic"`` so the suite needs no
macOS ``say`` (absent in CI) and stays fast. The HTTP-mode test posts through FastAPI's
test client so the exact multipart shape Person 4's backend sends is exercised without
starting a server.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from benchmarks import fixtures as fx
from benchmarks import run_benchmark as rb

SHORT = 1.0


# -------------------------------------------------------------------------- fixtures
def test_quality_fixtures_are_well_formed():
    built = fx.build_all(SHORT, speech_mode="synthetic")
    assert [f.name for f in built] == list(fx.QUALITY_FIXTURES)
    for fixture in built:
        assert fixture.sample_rate == 48_000
        assert fixture.channels == 2, "quality fixtures are stereo so stereo handling is exercised"
        assert fixture.duration == pytest.approx(SHORT, abs=1e-6)
        assert fixture.target in fixture.sources
        assert len(fixture.sources) >= 2
        mixture = fixture.mixture
        assert mixture.dtype == np.float32 and np.isfinite(mixture).all()
        assert 0.05 < float(np.abs(mixture).max()) <= fx.PEAK_CEILING + 1e-6
        summed = sum(fixture.sources.values())
        assert np.array_equal(summed, mixture), "mixture must be exactly the sum of the ground-truth sources"
        assert np.abs(fixture.interference).max() > 0.0


def test_fixtures_are_deterministic():
    first, second = fx.speech_plus_siren(SHORT, speech_mode="synthetic"), fx.speech_plus_siren(SHORT, speech_mode="synthetic")
    assert np.array_equal(first.mixture, second.mixture)
    assert np.array_equal(first.target_signal, second.target_signal)


def test_fixture_prompts_cover_the_required_experiments():
    prompts = {f.name: f.prompt for f in fx.build_all(SHORT, speech_mode="synthetic")}
    assert prompts["speech_plus_siren"] == "siren"
    assert "speaking" in prompts["two_speakers"]
    assert prompts["music_mix"] == "guitar"


def test_long_mixture_is_mono_and_sized():
    fixture = fx.long_mixture(2.0, speech_mode="synthetic")
    assert fixture.channels == 1
    assert fixture.frames == 96_000
    assert fixture.prompt == "siren"


def test_target_is_recoverable_in_principle():
    """Ground truth must be a real separation problem: the mixture alone scores poorly."""
    from sam_audio_service.metrics import si_sdr

    for fixture in fx.build_all(SHORT, speech_mode="synthetic"):
        assert si_sdr(fixture.mixture, fixture.target_signal) < 10.0


def test_unknown_fixture_name_raises():
    with pytest.raises(ValueError, match="unknown fixture"):
        fx.build_all(SHORT, names=("nope",))


def test_write_fixture_set_manifest(tmp_path):
    manifest_path = fx.write_fixture_set(tmp_path, seconds=SHORT, speech_mode="synthetic", latency_seconds=1.0)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == fx.MANIFEST_SCHEMA
    assert manifest["sample_rate"] == 48_000
    assert len(manifest["fixtures"]) == len(fx.QUALITY_FIXTURES) + 1
    from sam_audio_service.audio_io import decode

    for entry in manifest["fixtures"]:
        mixture_path = tmp_path / entry["mixture"]
        assert mixture_path.exists()
        decoded = decode(path=mixture_path)
        assert decoded.sample_rate == 48_000
        assert decoded.duration == pytest.approx(entry["duration"], abs=0.01)
        assert decoded.channels == entry["channels"]
        roles = {source["role"] for source in entry["sources"]}
        assert "target" in roles and "interference" in roles
        for source in entry["sources"]:
            assert (tmp_path / source["file"]).exists()
        assert entry["target_source"] in [source["name"] for source in entry["sources"]]


# ---------------------------------------------------------------------------- runner
def _report(out_dir):
    return json.loads((out_dir / "report.json").read_text())


def test_benchmark_smoke_with_mock_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    out = tmp_path / "out"
    assert rb.main(["--backend", "mock", "--seconds", "1", "--latency-seconds", "2", "--long-seconds", "0", "--repeat", "2",
                    "--speech-mode", "synthetic", "--out", str(out)]) == 0
    report = _report(out)
    assert report["backend"]["name"] == "mock"
    assert report["backend"]["mode"] == "engine"
    assert report["is_mock"] is True
    assert report["mock_disclaimer"]
    assert len(report["fixtures"]) == len(fx.QUALITY_FIXTURES) + 1
    for row in report["fixtures"]:
        metrics = row["metrics"]
        assert metrics["repeats"] == 2 and len(metrics["latency_seconds_all"]) == 2
        assert metrics["has_residual"] is True
        assert metrics["reconstruction_error"] < 1e-4, "engine residual must reconstruct the mixture"
        assert metrics["latency_seconds"] > 0 and metrics["real_time_factor"] > 0
        assert metrics["si_sdr_db"] is not None and math.isfinite(metrics["si_sdr_db"])
        assert metrics["leakage_db"] is not None
        assert row["stems"][0] == row["prompt"] or "residual" in row["stems"]
    markdown = (out / "REPORT.md").read_text()
    assert "MOCK RUN" in markdown and "NOT A QUALITY RESULT" in markdown
    assert "| Fixture |" in markdown
    for name in fx.QUALITY_FIXTURES:
        assert name in markdown
    assert "Latency reference" in markdown


def test_benchmark_fixture_subset_and_no_latency(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    out = tmp_path / "subset"
    assert rb.main(["--seconds", "1", "--latency-seconds", "0", "--fixtures", "music_mix", "--speech-mode", "synthetic", "--out", str(out)]) == 0
    report = _report(out)
    assert [row["fixture"] for row in report["fixtures"]] == ["music_mix"]
    assert "Latency reference" not in (out / "REPORT.md").read_text()


def test_benchmark_can_write_the_fixture_set(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    out, generated = tmp_path / "out", tmp_path / "generated"
    assert rb.main(["--seconds", "1", "--latency-seconds", "0", "--fixtures", "music_mix", "--speech-mode", "synthetic",
                    "--out", str(out), "--write-fixtures", str(generated)]) == 0
    assert (generated / "manifest.json").exists()


def test_benchmark_http_mode_matches_person4_multipart(tmp_path, monkeypatch, client):
    """``--url`` mode must post what ``app/media.py`` posts and read ``stems[].audio_base64``."""
    import httpx

    seen: dict[str, object] = {}

    def post(url, *, files, data, timeout):  # noqa: ANN001 - mirrors httpx.post's shape
        seen.update({"url": url, "file_field": sorted(files), "data_field": sorted(data), "timeout": timeout})
        return client.post("/separate", files=files, data=data)

    monkeypatch.setattr(httpx, "post", post)
    out = tmp_path / "http"
    assert rb.main(["--url", "http://example.invalid/separate", "--seconds", "1", "--latency-seconds", "0", "--long-seconds", "2",
                    "--speech-mode", "synthetic", "--out", str(out)]) == 0
    assert seen["file_field"] == ["audio"] and seen["data_field"] == ["metadata"]
    report = _report(out)
    assert report["backend"]["mode"] == "http"
    assert report["backend"]["name"] == "mock" and report["is_mock"] is True
    assert report["warnings"], "the service must tell callers the mock is in use"
    for row in report["fixtures"]:
        # 16-bit WAV round-trip over HTTP: quantisation only, no lost or shifted samples.
        assert row["metrics"]["reconstruction_error"] < 1e-3
        assert row["metrics"]["has_residual"] is True


def test_benchmark_rejects_bad_repeat(tmp_path):
    with pytest.raises(SystemExit):
        rb.main(["--repeat", "0", "--out", str(tmp_path)])


def test_metadata_payload_is_person4_shaped():
    assert rb._metadata("siren") == {"prompts": ["siren"]}


def test_metadata_payload_with_passes():
    assert rb._metadata("siren", 1) == {"prompts": ["siren"]}
    assert rb._metadata("siren", 2) == {"prompts": ["siren"], "options": {"passes": 2}}


def test_benchmark_passes_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    out = tmp_path / "passes"
    assert rb.main(["--seconds", "1", "--latency-seconds", "0", "--fixtures", "music_mix", "--speech-mode", "synthetic",
                    "--passes", "2", "--out", str(out)]) == 0
    report = _report(out)
    assert report["settings"]["passes"] == 2
    with pytest.raises(SystemExit):
        rb.main(["--passes", "0", "--out", str(tmp_path)])


# ------------------------------------------------------------- long fixture + settings
def test_metadata_payload_with_options():
    assert rb._metadata("siren", 1, {"seed": 1}) == {"prompts": ["siren"], "options": {"seed": 1}}
    assert rb._metadata("siren", 2, {"seed": 1}) == {"prompts": ["siren"], "options": {"passes": 2, "seed": 1}}
    assert rb._metadata("siren", 1, {}) == {"prompts": ["siren"]}


def test_competing_voices_fixture():
    seconds = 2.0
    fixture = fx.competing_voices(seconds, speech_mode="synthetic")
    assert fixture.name == "competing_voices" and fixture.name in fx.LONG_FIXTURES
    assert fixture.name not in fx.QUALITY_FIXTURES
    assert fixture.channels == 2
    assert fixture.duration == pytest.approx(seconds, abs=1e-6)
    assert fixture.prompt == "man speaking" and fixture.target == "speaker_low"
    assert fixture.extra == {"kind": "long", "other_prompt": "woman speaking"}
    summed = sum(fixture.sources.values())
    assert np.max(np.abs(summed - fixture.mixture)) < 1e-6
    assert float(np.abs(fixture.mixture).max()) <= fx.PEAK_CEILING + 1e-6
    schedule = fx.competing_schedule(seconds)
    for name, turns in schedule.items():
        source = fixture.sources[name]
        active = np.zeros(fixture.frames, dtype=bool)
        for start, end in turns:
            assert 0.0 <= start < end <= seconds + 1e-9
            active[int(round(start * fx.SAMPLE_RATE)):int(round(end * fx.SAMPLE_RATE))] = True
        assert np.abs(source[:, active]).max() > 0.0, f"{name} must speak during its turns"
        assert np.abs(source[:, ~active]).max() == 0.0, f"{name} must be silent outside its turns"


def test_long_fixtures_default_on_and_skippable(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    on, off = tmp_path / "on", tmp_path / "off"
    common = ["--seconds", "1", "--latency-seconds", "0", "--speech-mode", "synthetic"]
    assert rb.main([*common, "--long-seconds", "2", "--out", str(on)]) == 0
    assert [row["fixture"] for row in _report(on)["fixtures"]] == [*fx.QUALITY_FIXTURES, *fx.LONG_FIXTURES]
    assert _report(on)["settings"]["long_seconds"] == 2.0
    assert rb.main([*common, "--long-seconds", "0", "--out", str(off)]) == 0
    assert [row["fixture"] for row in _report(off)["fixtures"]] == list(fx.QUALITY_FIXTURES)


def test_ode_step_and_options_recorded_in_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    out = tmp_path / "ode"
    assert rb.main(["--seconds", "1", "--latency-seconds", "0", "--long-seconds", "0", "--fixtures", "music_mix",
                    "--speech-mode", "synthetic", "--ode-step", "0.03125", "--refine", "mask", "--refine-threshold", "0.3",
                    "--out", str(out)]) == 0
    settings = _report(out)["settings"]
    assert settings["ode_step"] == 0.03125
    assert settings["effective_ode_step"] == 0.03125  # mock has no ode_opt: settings.ode_step_size path
    assert settings["request_options"] == {"refine": "mask", "refine_threshold": 0.3}
    assert settings["seed"] is None, "mock runs send no seed unless asked"
    assert settings["peak_rss_bytes"] > 1_000_000


def test_request_options_only_sent_when_given():
    assert rb.request_options(rb.config_from_args(rb.build_parser().parse_args([]))) == {}
    args = rb.build_parser().parse_args(["--backend", "sam", "--window-candidates", "2", "--select", "overlap", "--refine-gate-db", "30"])
    assert rb.request_options(rb.config_from_args(args)) == {"seed": rb.DEFAULT_SEED, "candidates": 2, "candidate_select": "overlap",
                                                              "refine_gate_db": 30.0}


def test_named_sweeps_are_valid():
    from benchmarks.sweeps import SWEEPS

    assert {"ode", "refine", "refine-floor", "candidates", "chunks", "quality-v1", "leakage"} <= set(SWEEPS)
    for name, configs in SWEEPS.items():
        names = [config["name"] for config in configs]
        assert len(names) == len(set(names)), name
        for config in configs:
            rb.validate_config(dict(config), url=None, label=name)
    assert len(SWEEPS["leakage"]) == 7


def test_sweep_writes_one_table_with_deltas(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    spec = tmp_path / "mini.json"
    spec.write_text(json.dumps([{"name": "base"}, {"name": "finer-ode", "ode_step": 0.03125, "chunk_seconds": 0}, {"name": "base-again"}]))
    out = tmp_path / "sweep"
    assert rb.main(["--seconds", "1", "--latency-seconds", "1", "--long-seconds", "0", "--fixtures", "music_mix",
                    "--speech-mode", "synthetic", "--sweep", str(spec), "--out", str(out)]) == 0
    sweep = json.loads((out / "sweep.json").read_text())
    assert [entry["name"] for entry in sweep["configs"]] == ["base", "finer-ode", "base-again"]
    assert sweep["configs"][1]["delta"]["music_mix"]["si_sdri_db"] is not None
    assert "leakage_db" in sweep["configs"][1]["delta"]["music_mix"]
    assert sweep["configs"][1]["settings"]["ode_step"] == 0.03125
    assert sweep["configs"][1]["request_options"] == {"chunk_seconds": 0}
    assert sweep["configs"][0]["rtf"] is not None and sweep["configs"][0]["peak_rss_bytes"] > 0
    for name in ("base", "finer-ode", "base-again"):
        assert (out / name / "report.json").exists() and (out / name / "REPORT.md").exists()
    assert _report(out / "finer-ode")["settings"]["effective_ode_step"] == 0.03125
    assert _report(out / "base-again")["settings"]["effective_ode_step"] != 0.03125, "a config's engine knobs must not leak into the next"
    markdown = (out / "SWEEP.md").read_text()
    assert "Δ" in markdown and "leak" in markdown and "MOCK RUN" in markdown
    assert "`base`" in markdown and "`finer-ode`" in markdown
