import base64
import io
import json
import wave

import numpy as np
import pytest

from sam_audio_service.audio_io import decode, encode_wav_pcm16


def _post(client, wav_bytes, metadata):
    return client.post("/separate", files={"audio": ("audio.wav", wav_bytes, "audio/wav")}, data={"metadata": json.dumps(metadata)})


def test_health_and_capabilities(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model"]["is_mock"] is True
    caps = client.get("/capabilities").json()
    assert caps["prompting_modes"]["text"] is True
    assert "INVALID_MEDIA" in caps["error_codes"]


def test_person4_contract_shape(client, wav_path):
    """Exactly what app/media.py (Person 4) sends and reads."""
    r = _post(client, wav_path.read_bytes(), {"prompts": ["dog"], "visual_prompt": {"track_id": "42"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["stems"], list) and body["stems"]
    names = [s["name"] for s in body["stems"]]
    assert names == ["dog", "residual"]
    for stem in body["stems"]:
        raw = base64.b64decode(stem["audio_base64"], validate=True)
        with wave.open(io.BytesIO(raw)) as w:
            assert w.getframerate() == 48000 and w.getnchannels() == 2 and w.getnframes() == 48000
        assert stem["sample_rate"] == 48000 and stem["channels"] == 2 and stem["start_time"] == 0.0
        assert abs(stem["duration"] - 1.0) < 1e-6
    assert body["model"]["is_mock"] is True
    assert any("MOCK" in w for w in body["warnings"])
    assert any("track_id" in w for w in body["warnings"])  # unsupported visual field reported, not fatal


def test_stems_reconstruct_mixture_exactly(client, wav_path):
    r = _post(client, wav_path.read_bytes(), {"prompts": ["siren", "music"]})
    body = r.json()
    original = decode(data=wav_path.read_bytes()).samples
    total = sum(decode(data=base64.b64decode(s["audio_base64"])).samples for s in body["stems"])
    assert total.shape == original.shape
    assert np.max(np.abs(total - original)) < 2e-4  # 16-bit quantisation of three stems


def test_prompt_objects_anchors_and_options(client, wav_path):
    meta = {
        "prompts": [{"text": "Isolate the Siren!", "name": "Police siren", "anchors": [{"token": "+", "start": 0.2, "end": 0.6}]}],
        "options": {"residual": "none", "predict_spans": True},
        "asset_id": "video_001",
        "request_id": "req-1",
    }
    body = _post(client, wav_path.read_bytes(), meta).json()
    assert body["request_id"] == "req-1"
    assert [s["name"] for s in body["stems"]] == ["Police siren"]
    stem = body["stems"][0]
    assert stem["prompt"] == "siren" and stem["source_asset_id"] == "video_001"
    assert stem["spans"] and stem["spans"][0][0] >= 0.15 and stem["spans"][-1][1] <= 0.65
    samples = decode(data=base64.b64decode(stem["audio_base64"])).samples
    assert np.abs(samples[:, : int(0.1 * 48000)]).max() < 1e-3  # gated outside the + span


def test_time_range_visual_prompt_becomes_anchor(client, wav_path):
    meta = {"prompts": ["person"], "visual_prompt": {"asset_id": "video_001", "object_id": "person_2", "label": "person", "time_range": {"start": 0.5, "end": 0.9}, "visual_prompt_ref": "visual/person_2"}}
    body = _post(client, wav_path.read_bytes(), meta).json()
    assert body["stems"][0]["spans"][0] == [0.5, 0.9]
    assert any("span anchors" in w for w in body["warnings"])


def test_errors_follow_contract(client, wav_path):
    r = _post(client, b"not audio", {"prompts": ["dog"]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_MEDIA" and r.json()["detail"]
    r = _post(client, wav_path.read_bytes(), {"prompts": []})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"
    r = client.post("/separate", files={"audio": ("a.wav", wav_path.read_bytes())}, data={"metadata": "{not json"})
    assert r.json()["error"]["code"] == "INVALID_REQUEST"
    r = _post(client, wav_path.read_bytes(), {"prompts": ["x"], "options": {"model_id": "facebook/other"}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_ref_endpoint_and_file_return(client, wav_path, settings):
    r = client.post("/v1/separate", json={"audio_ref": wav_path.as_uri(), "prompts": ["dog"], "options": {"return_mode": "file"}})
    assert r.status_code == 200, r.text
    stem = r.json()["stems"][0]
    assert stem["audio_ref"].startswith("file://") and "audio_base64" not in stem
    r = client.post("/v1/separate", json={"audio_ref": "file:///etc/passwd", "prompts": ["dog"]})
    assert r.status_code in (400, 404) and r.json()["error"]["code"] in ("INVALID_REQUEST", "MISSING_ASSET")
    r = client.post("/v1/separate", json={"audio_ref": (wav_path.parent / "missing.wav").as_uri(), "prompts": ["dog"]})
    assert r.json()["error"]["code"] == "MISSING_ASSET"


def test_mono_mp3_and_resampling(client, tmp_path):
    import subprocess

    src = tmp_path / "m.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-ar", "22050", "-ac", "1", str(src)], check=True)
    body = _post(client, src.read_bytes(), {"prompts": ["tone"]}).json()
    assert body["source"]["sample_rate"] == 48000 and body["source"]["channels"] == 1
    assert body["stems"][0]["channels"] == 1


def test_chunking_preserves_length_and_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_CHUNK_SECONDS", "1.0")
    monkeypatch.setenv("AUDIO_CHUNK_OVERLAP_SECONDS", "0.25")
    from sam_audio_service.audio_io import DecodedAudio
    from sam_audio_service.config import load_settings
    from sam_audio_service.contracts import SeparationMetadata
    from sam_audio_service.engine import Engine

    engine = Engine(load_settings())
    rate = 48000
    t = np.arange(int(3.3 * rate)) / rate
    mix = np.stack([np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 1000 * t)]).astype(np.float32)
    result = engine.separate(DecodedAudio(mix, rate, rate, 2), SeparationMetadata(prompts=["siren"]))
    assert result.model["chunked"] is True
    for stem in result.stems:
        assert stem.samples.shape == mix.shape
    total = sum(s.samples for s in result.stems)
    assert np.max(np.abs(total - mix)) < 1e-5
    # the 1 kHz tone sits inside the mock "siren" band, so the target should carry most energy
    target = result.stems[0].samples
    assert np.mean(target**2) > 0.8 * np.mean(mix**2)
