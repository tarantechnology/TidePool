"""Regressions for the findings of the independent code review on PR #13."""
import base64
import json
import subprocess

import numpy as np
import pytest

from sam_audio_service.audio_io import decode, encode_wav_pcm16


def _post(client, data, metadata, name="audio.wav"):
    return client.post("/separate", files={"audio": (name, data, "application/octet-stream")}, data={"metadata": json.dumps(metadata)})


def test_request_id_cannot_escape_output_root(client, wav_path, settings):
    for bad in ("../../evil", "/tmp/evil", "a/b"):
        r = _post(client, wav_path.read_bytes(), {"prompts": ["x"], "request_id": bad, "options": {"return_mode": "file"}})
        assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST", bad
    assert not (settings.output_root.parent / "evil").exists()


def test_per_request_chunk_seconds_is_validated(client, wav_path):
    r = _post(client, wav_path.read_bytes(), {"prompts": ["x"], "options": {"chunk_seconds": 2.0}})
    assert r.status_code == 400 and "chunk_seconds" in r.json()["detail"]
    r = _post(client, wav_path.read_bytes(), {"prompts": ["x"], "options": {"chunk_seconds": 0}})
    assert r.status_code == 200


def test_non_faststart_mp4_upload_decodes(client, tmp_path):
    video = tmp_path / "trailing_moov.mp4"
    # Default mp4 muxing puts the moov atom last; this cannot be demuxed from a pipe.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=32x32:r=10:d=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video)], check=True)
    r = _post(client, video.read_bytes(), {"prompts": ["tone"]}, name="trailing_moov.mp4")
    assert r.status_code == 200, r.text
    assert r.json()["source"]["duration"] > 0.9


def test_multichannel_wav_is_downmixed_symmetrically(tmp_path):
    rate = 48000
    channels = np.stack([np.full(rate, (i + 1) / 10.0, dtype=np.float32) for i in range(6)])
    path = tmp_path / "six.wav"
    path.write_bytes(encode_wav_pcm16(channels, rate))
    decoded = decode(path=path)
    assert decoded.channels == 2 and decoded.source_channels == 6
    left, right = decoded.samples.mean(axis=1)
    # The old fast path returned left == channel 0 only (0.1). A matrix downmix folds the
    # centre/surround channels into both sides.
    assert left > 0.3 and right > 0.3


def test_textless_prompt_with_person2_masks_is_accepted(tmp_path, monkeypatch):
    from test_visual_prompts import detections_doc, make_video, selection_doc

    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(tmp_path))
    video = make_video(tmp_path / "demo.mp4")
    from fastapi.testclient import TestClient

    from sam_audio_service.api import create_app
    from sam_audio_service.config import load_settings

    with TestClient(create_app(load_settings())) as client:
        wav = encode_wav_pcm16(np.zeros((2, 48000), dtype=np.float32) + 0.1, 48000)
        meta = {"prompts": [{"text": "", "name": "selected person"}], "source": {"original_ref": video.as_uri()}, "visual_prompt": {"selection": selection_doc(), "detections": detections_doc()}}
        r = _post(client, wav, meta)
        assert r.status_code == 200, r.text
        assert r.json()["stems"][0]["name"] == "selected person" and r.json()["stems"][0]["spans"] == [[0.2, 0.8]]
        # text-less prompt whose visual guidance is unusable -> 400, not 500
        r = _post(client, wav, {"prompts": [{"text": ""}], "visual_prompt": {"track_id": "42"}})
        assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"


def test_malformed_visual_metadata_is_a_client_error(client, wav_path):
    bad = {"selection": {"kind": "visual_selection", "interval": [0.2, 0.8], "track_id": "track_7"}, "detections": {"frames": []}}
    r = _post(client, wav_path.read_bytes(), {"prompts": ["person"], "visual_prompt": bad})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"


def test_generic_time_range_sets_visual_interval(tmp_path, monkeypatch):
    from sam_audio_service.config import load_settings
    from sam_audio_service.visual_prompts import translate_visual_prompt
    from test_visual_prompts import make_video

    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(tmp_path))
    video = make_video(tmp_path / "v.mp4", seconds=2.0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=64x48:r=10:d=2", "-c:v", "ffv1", str(tmp_path / "mask.mkv")], check=True)
    raw = {"video_ref": video.as_uri(), "mask_video_ref": (tmp_path / "mask.mkv").as_uri(), "time_range": {"start": 0.5, "end": 1.0}}
    t = translate_visual_prompt(raw, source=None, source_duration=2.0, settings=load_settings(), backend_supports_visual=True)
    assert t.visual is not None and t.visual.interval == (0.5, 1.0)


def test_error_codes_have_descriptions(client):
    codes = client.get("/capabilities").json()["error_codes"]
    assert all(v["description"] for v in codes.values())
