"""Music stems end to end over HTTP with the mock splitter (no weights, CPU only)."""
import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sam_audio_service.api import create_app
from sam_audio_service.audio_io import decode, encode_wav_pcm16
from sam_audio_service.classifier import MockClassifier
from sam_audio_service.config import load_settings
from sam_audio_service.engine import Engine

SR = 48_000


def song_wav(seconds: float = 1.0) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    mono = 0.3 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 5000 * t) + 0.05 * np.sin(2 * np.pi * 10000 * t)
    return encode_wav_pcm16(np.stack([mono, 0.8 * mono]).astype(np.float32), SR)


def post(client, metadata, path="/separate", audio=None):
    return client.post(path, files={"audio": ("song.wav", audio or song_wav(), "audio/wav")}, data={"metadata": json.dumps(metadata)})


def samples(stem):
    return decode(data=base64.b64decode(stem["audio_base64"])).samples


@pytest.fixture
def music_settings(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    return load_settings()


@pytest.fixture
def music_client(music_settings):
    with TestClient(create_app(music_settings)) as c:
        yield c


def classified_client(settings, music: float):
    return TestClient(create_app(settings, Engine(settings, classifier=MockClassifier(music=music, speech=1 - music))))


def test_full_split_returns_every_stem_and_no_residual(music_client):
    r = post(music_client, {"prompts": []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["name"] for s in body["stems"]] == ["Vocals", "Drums", "Bass", "Other"]
    assert [s["label"] for s in body["stems"]] == ["vocals", "drums", "bass", "other"]
    assert {s["role"] for s in body["stems"]} == {"target"}
    assert body["routing"] == {"mode": "music_stems", "engine": "mock", "model_id": "mock-bands", "reason": body["routing"]["reason"], "scores": None, "stems": ["vocals", "drums", "bass", "other"]}
    assert body["model"]["backend"] == "mock" and body["model"]["model_id"] == "mock-bands" and body["model"]["is_mock"]
    assert any("MOCK stem splitter" in w for w in body["warnings"])
    mix = decode(data=song_wav()).samples
    total = sum(samples(s) for s in body["stems"])
    assert total.shape == mix.shape and np.max(np.abs(total - mix)) < 1e-4


def test_subset_keeps_caller_wording_and_adds_one_residual(music_client):
    r = post(music_client, {"prompts": ["Lead vocal", "the drums"], "options": {"mode": "music_stems"}, "asset_id": "song_1"})
    assert r.status_code == 200, r.text
    stems = r.json()["stems"]
    assert [(s["name"], s["label"], s["role"]) for s in stems] == [("Lead vocal", "vocals", "target"), ("the drums", "drums", "target"), ("residual", "everything else", "residual")]
    assert stems[0]["source_asset_id"] == "song_1"
    mix = decode(data=song_wav()).samples
    assert np.max(np.abs(sum(samples(s) for s in stems) - mix)) < 1e-4


def test_instrumental_and_residual_options(music_client):
    body = post(music_client, {"options": {"mode": "music_stems", "stems": ["karaoke"], "residual_name": "Vocals only"}}).json()
    assert [(s["name"], s["label"]) for s in body["stems"]] == [("Instrumental", "instrumental"), ("Vocals only", "everything else")]
    body = post(music_client, {"options": {"mode": "music_stems", "stems": ["vocals"], "residual": "none"}}).json()
    assert [s["name"] for s in body["stems"]] == ["Vocals"]


def test_output_format_and_file_return(music_client):
    body = post(music_client, {"options": {"mode": "music_stems", "output_format": "pcm_s16le", "return_mode": "file"}}).json()
    assert all(s["format"] == "wav/pcm_s16le" and s["audio_ref"].startswith("file://") and "audio_base64" not in s for s in body["stems"])


def test_errors(music_client, client, music_settings, monkeypatch):
    r = post(music_client, {"prompts": ["vocals", "dog barking"], "options": {"mode": "music_stems"}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"
    assert r.json()["error"]["details"]["valid_stems"] == ["vocals", "drums", "bass", "other", "instrumental"]
    r = post(music_client, {"options": {"mode": "music_stems", "stem_model": "htdemucs_6s"}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"
    r = post(client, {"options": {"mode": "music_stems"}})  # default settings: no splitter
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"
    monkeypatch.setenv("AUDIO_MAX_DURATION_SECONDS", "0.5")
    with TestClient(create_app(load_settings())) as short:
        r = post(short, {"options": {"mode": "music_stems"}})
        assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_auto_routing_with_and_without_a_classifier(music_client, music_settings):
    # No classifier (the default for the mock splitter): stem prompts stay prompted...
    body = post(music_client, {"prompts": ["vocals"]}).json()
    assert body["routing"]["mode"] == "prompted" and body["routing"]["engine"] == "mock" and body["routing"]["scores"] is None
    assert [s["name"] for s in body["stems"]] == ["vocals", "residual"] and body["routing"].get("stems") is None
    # ...unless the app says the upload is audio-only.
    body = post(music_client, {"prompts": ["vocals"], "hints": {"source_kind": "audio"}}).json()
    assert body["routing"]["mode"] == "music_stems"
    with classified_client(music_settings, 0.9) as c:
        body = post(c, {"prompts": ["vocals", "bass"]}).json()
        assert body["routing"]["mode"] == "music_stems" and body["routing"]["scores"] == {"music": 0.9, "speech": pytest.approx(0.1)}
        assert [s["name"] for s in body["stems"]] == ["vocals", "bass", "residual"]
        assert post(c, {"prompts": ["vocals", "dog"]}).json()["routing"]["reason"].startswith("auto: mixed prompts")
    with classified_client(music_settings, 0.1) as c:
        assert post(c, {"prompts": ["vocals"]}).json()["routing"]["mode"] == "prompted"


def test_prompted_responses_carry_routing(client, wav_path):
    r = post(client, {"prompts": ["dog"]}, audio=wav_path.read_bytes())
    assert r.status_code == 200 and r.json()["routing"]["mode"] == "prompted"
    assert r.json()["routing"]["reason"] == "auto: no stem splitter is configured"
    r = post(client, {"prompts": []}, audio=wav_path.read_bytes())
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"  # unchanged without a splitter


def test_route_dry_run(music_client, music_settings):
    r = post(music_client, {"prompts": []}, path="/v1/route")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"routing", "source", "warnings", "contract_version"}
    assert body["routing"]["mode"] == "music_stems" and body["routing"]["scores"] is None
    assert body["source"]["sample_rate"] == SR
    assert post(music_client, {"prompts": ["dog"]}, path="/v1/route").json()["routing"]["mode"] == "prompted"
    r = post(music_client, {"prompts": ["dog"], "options": {"mode": "music_stems"}}, path="/v1/route")
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"
    with classified_client(music_settings, 0.2) as c:
        body = post(c, {}, path="/v1/route").json()
        assert body["routing"]["scores"]["music"] == 0.2 and any("does not sound like music" in w for w in body["warnings"])


def test_ref_endpoint_routes_too(music_client, music_settings):
    path = music_settings.media_root / "song.wav"
    path.write_bytes(song_wav())
    r = music_client.post("/v1/separate", json={"audio_ref": path.as_uri(), "options": {"stems": ["bass"], "mode": "music_stems"}})
    assert r.status_code == 200, r.text
    assert [s["name"] for s in r.json()["stems"]] == ["Bass", "residual"]


def test_capabilities_and_health(music_client, client):
    caps = music_client.get("/capabilities").json()
    assert caps["music_stems"]["available"] and caps["music_stems"]["backend"] == "mock"
    assert caps["music_stems"]["stems"] == ["vocals", "drums", "bass", "other", "instrumental"]
    assert caps["music_stems"]["aliases"]["karaoke"] == "instrumental"
    assert caps["music_stems"]["license"] == "n/a (no weights)"
    assert caps["error_codes"]["CANCELLED"]["http_status"] == 409
    assert caps["routing"] == {"default_mode": "auto", "modes": ["auto", "prompted", "music_stems"], "classifier": "off", "music_threshold": 0.5}
    health = music_client.get("/health").json()
    assert health["splitter"]["backend"] == "mock" and set(health["splitter"]["stems"]) == {"vocals", "drums", "bass", "other"}
    assert health["classifier"]["classifier"] == "off"
    off = client.get("/capabilities").json()["music_stems"]
    assert off["available"] is False and off["stems"] == []
    assert client.get("/health").json()["splitter"] is None


def test_engine_logs_split_progress(music_client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="sam_audio_service"):
        assert post(music_client, {"options": {"mode": "music_stems"}}).status_code == 200
    lines = [r.getMessage() for r in caplog.records if "stem split" in r.getMessage()]
    assert lines and lines[-1] == "mock stem split: done 100%"
