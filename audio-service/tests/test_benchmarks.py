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
    assert rb.main(["--backend", "mock", "--seconds", "1", "--latency-seconds", "2", "--repeat", "2",
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
    assert rb.main(["--url", "http://example.invalid/separate", "--seconds", "1", "--latency-seconds", "0",
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
