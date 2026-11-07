"""Cross-team Test 3 (Backend -> Audio) and the audio half of Test 4 (cinematic demo).

Runs Person 4's real FastAPI backend from ../app with INFERENCE_ADAPTER=http pointing at a
live instance of this service, then walks the documented application flow:
create project -> upload -> process -> poll job -> fetch stem audio -> patch stem -> export.
Person 4's code is imported, never modified.
"""
from __future__ import annotations

import base64
import importlib
import io
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import uvicorn

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (REPO / "app" / "main.py").exists(), reason="Person 4 backend not present")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def audio_service(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    from sam_audio_service.api import create_app
    from sam_audio_service.config import load_settings

    port = _free_port()
    config = uvicorn.Config(create_app(load_settings()), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/separate"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def backend_client(tmp_path, monkeypatch, audio_service):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("INFERENCE_ADAPTER", "http")
    monkeypatch.setenv("SAM_AUDIO_SERVICE_URL", audio_service)
    monkeypatch.syspath_prepend(str(REPO))
    for name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        sys.modules.pop(name)
    core = importlib.import_module("app.core")
    core.ROOT = tmp_path / "storage"
    core.ROOT.mkdir(parents=True, exist_ok=True)
    core.DB = core.ROOT / "db.sqlite"
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        yield client, core


def _wait(client, url, headers):
    for _ in range(400):
        data = client.get(url, headers=headers).json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.05)
    pytest.fail("job timed out")


def _demo_video(path: Path) -> Path:
    """Two 'voices' (tones) plus a siren sweep over a colour video, like the cinematic demo."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=2",
        "-f", "lavfi", "-i", "aevalsrc=0.3*sin(2*PI*220*t)+0.3*sin(2*PI*330*t)+0.3*sin(2*PI*(900+300*sin(2*PI*1.5*t))*t):s=48000:c=stereo:d=2",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path


def test_backend_to_audio_and_export(backend_client, tmp_path):
    client, core = backend_client
    p = client.post("/api/projects").json()
    pid, headers = p["project_id"], {"Authorization": "Bearer " + p["access_token"]}
    video = _demo_video(tmp_path / "demo.mp4")
    with video.open("rb") as f:
        r = client.post(f"/api/projects/{pid}/upload", headers=headers, files={"file": ("demo.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    source = r.json()["metadata"]

    # Person 2 style visual metadata forwarded unchanged by Person 4
    assert client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json={"object_id": "person_2", "label": "person", "time_range": {"start": 0.5, "end": 1.5}}).status_code == 200

    r = client.post(f"/api/projects/{pid}/process", headers=headers, json={"prompts": ["siren"]})
    assert r.status_code == 202, r.text
    job = _wait(client, "/api/jobs/" + r.json()["job_id"], headers)
    assert job["status"] == "completed", job
    names = [s["name"] for s in job["stems"]]
    assert names == ["siren", "residual"], names  # target + "everything else", as documented

    # Stems are consumable through Person 4's API with the source's timing
    for stem in job["stems"]:
        assert abs(stem["duration"] - source["duration"]) < 0.05, (stem["duration"], source["duration"])
        audio = client.get(stem["audio_url"], headers=headers)
        assert audio.status_code == 200
        with wave.open(io.BytesIO(audio.content)) as w:
            assert w.getframerate() == 48000 and w.getnchannels() == 2
    siren = next(s for s in job["stems"] if s["name"] == "siren")

    # Editor-style edits (Person 3 operations through Person 4's API) and the cinematic export
    r = client.patch(f"/api/projects/{pid}/stems/{siren['id']}", headers=headers, json={"volume": 0.2, "automation": [{"time": 0, "value": 1}, {"time": 1.5, "value": 0.1}]})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/export", headers=headers, json={"format": "mp4", "include_video": True})
    assert r.status_code == 202, r.text
    result = _wait(client, "/api/exports/" + r.json()["export_id"], headers)
    assert result["status"] == "completed", result
    out = tmp_path / "export.mp4"
    out.write_bytes(client.get(result["download_url"], headers=headers).content)
    info = core.probe(out)
    assert info["has_video"] and abs(info["duration"] - source["duration"]) < 0.1


def test_backend_surfaces_audio_service_error(backend_client, tmp_path):
    """Invalid prompt payload -> this service returns 400 INVALID_REQUEST -> Person 4 marks the job failed."""
    client, _ = backend_client
    p = client.post("/api/projects").json()
    pid, headers = p["project_id"], {"Authorization": "Bearer " + p["access_token"]}
    tone = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-ac", "2", "-ar", "48000", str(tone)], check=True)
    with tone.open("rb") as f:
        assert client.post(f"/api/projects/{pid}/upload", headers=headers, files={"file": ("tone.wav", f, "audio/wav")}).status_code == 200
    r = client.post(f"/api/projects/{pid}/process", headers=headers, json={"prompts": ["a" * 600]})
    job = _wait(client, "/api/jobs/" + r.json()["job_id"], headers)
    assert job["status"] == "failed" and job["error"]
