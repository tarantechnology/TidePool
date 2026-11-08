"""Cross-team Test 1 stand-in: the Person 3 (editor) stem fixture.

Person 3's editor is not in the repository, so the editor -> audio test cannot be executed.
These tests instead pin the artefact Person 3 integrates against: ``scripts/make_editor_fixture.py``
must produce a Contract C manifest that validates against this service's own pydantic
models, stem WAVs whose timing matches the source sample-for-sample, and stems that sum
back to the mixture (so "mute the siren" in the editor equals "original minus siren").
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import have_ffmpeg

from sam_audio_service.audio_io import decode
from sam_audio_service.contracts import SeparationResponse

SERVICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SERVICE_ROOT / "scripts" / "make_editor_fixture.py"
COMMITTED_EXAMPLE = SERVICE_ROOT / "fixtures" / "editor" / "example_response.json"

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg is required to render the fixture mixture")


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("editor_fixture") / "editor"
    env = {**os.environ, "AUDIO_BACKEND": "mock", "AUDIO_CHUNK_SECONDS": "0", "AUDIO_MEDIA_ROOT": ""}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(SERVICE_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


@pytest.fixture(scope="module")
def manifest(fixture_dir) -> dict:
    return json.loads((fixture_dir / "stems.json").read_text())


def _wav_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def test_manifest_validates_as_contract_c(manifest):
    response = SeparationResponse.model_validate(manifest)
    assert response.contract_version
    assert [s.name for s in response.stems] == ["speech", "siren", "residual"]
    assert [s.role for s in response.stems] == ["target", "target", "residual"]
    assert response.model.is_mock is True
    assert any("MOCK" in w for w in response.warnings)
    assert response.source.sample_rate == 48000 and response.source.channels == 2
    # Mock output must never be presented as separation.
    assert manifest["fixture"]["is_real_separation"] is False


def test_every_stem_shares_the_source_timeline(manifest):
    response = SeparationResponse.model_validate(manifest)
    source = response.source
    assert source.duration == pytest.approx(3.0, abs=1e-6)
    assert source.sample_count == 144000
    for stem in response.stems:
        assert stem.start_time == 0.0, stem.name
        assert stem.sample_rate == source.sample_rate, stem.name
        assert stem.channels == source.channels, stem.name
        assert stem.sample_count == source.sample_count, stem.name
        assert stem.duration == pytest.approx(source.duration, abs=1e-6), stem.name
        assert stem.format == "wav/pcm_s16le", stem.name
        # The fixture serves files; base64 is the HTTP contract and must not be inlined here.
        assert stem.audio_base64 is None, stem.name


def test_stem_files_match_the_source_wav(fixture_dir, manifest):
    source_rate, source_channels, source_frames = _wav_params(fixture_dir / "source.wav")
    assert (source_rate, source_channels) == (48000, 2)
    assert source_frames == manifest["source"]["sample_count"]
    assert manifest["fixture"]["source_audio_url"] == "source.wav"
    for item in manifest["stems"]:
        url = item["audio_url"]
        assert "/" not in url and url.endswith(".wav"), url  # relative, dev-server loadable
        path = fixture_dir / url
        assert path.is_file(), url
        assert _wav_params(path) == (source_rate, source_channels, source_frames), url


def test_stems_sum_back_to_the_source(fixture_dir, manifest):
    original = decode(path=fixture_dir / "source.wav").samples
    total = sum(decode(path=fixture_dir / item["audio_url"]).samples for item in manifest["stems"])
    assert total.shape == original.shape
    # Each stem is rounded to 16-bit independently, so three stems can drift by at most a
    # few LSBs (1 LSB = 1/32767 ~ 3.05e-5) from the mixture.
    assert np.max(np.abs(total - original)) < 4 * (1.0 / 32767.0)


def test_committed_example_response_is_valid_contract_c():
    """The hand-written example in fixtures/editor/ must stay in sync with the models."""
    payload = json.loads(COMMITTED_EXAMPLE.read_text())
    response = SeparationResponse.model_validate(payload)
    assert [s.role for s in response.stems] == ["target", "target", "residual"]
    assert {s.sample_count for s in response.stems} == {response.source.sample_count}
    assert all(s.start_time == 0.0 for s in response.stems)
    assert all(s.duration == response.source.duration for s in response.stems)
    assert "TRUNCATED_PLACEHOLDER" in payload["stems"][0]["audio_base64"]


def test_generator_accepts_custom_out_and_prompts(tmp_path):
    out = tmp_path / "custom"
    env = {**os.environ, "AUDIO_BACKEND": "mock", "AUDIO_MEDIA_ROOT": ""}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--seconds", "1", "--prompts", "dog"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(SERVICE_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((out / "stems.json").read_text())
    assert [s["name"] for s in payload["stems"]] == ["dog", "residual"]
    assert payload["source"]["sample_count"] == 48000
    with wave.open(str(out / "dog.wav")) as w:
        assert w.getnframes() == 48000


def test_fixture_readme_documents_the_mock(fixture_dir):
    readme = (SERVICE_ROOT / "fixtures" / "editor" / "README.md").read_text()
    assert "mock" in readme.lower() and "not real separation" in readme.lower()
    assert "stems.json" in readme
