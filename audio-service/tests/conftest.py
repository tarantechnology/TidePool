import importlib
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AUDIO_BACKEND", "mock")


def synth(path: Path, seconds: float = 1.0, rate: int = 48000, channels: int = 2, freq: float = 440.0) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = 0.05 * np.random.default_rng(0).standard_normal(len(t))
    sig = np.stack([tone + noise] * channels)
    from sam_audio_service.audio_io import encode_wav_pcm16

    path.write_bytes(encode_wav_pcm16(sig.astype(np.float32), rate))
    return path


@pytest.fixture
def wav_path(tmp_path):
    return synth(tmp_path / "mix.wav")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIO_OUTPUT_ROOT", str(tmp_path / "out"))
    from sam_audio_service.config import load_settings

    return load_settings()


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from sam_audio_service.api import create_app

    with TestClient(create_app(settings)) as c:
        yield c


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


# --------------------------------------------------------------------------------------
# Live-server scaffolding shared by the cross-team integration modules
# (``test_integration_person4_backend.py``, ``test_merged_visual_workflow.py``): this
# service running under uvicorn on a free port, with Person 4's real backend talking to
# it over HTTP. One copy lives here so the two modules cannot drift apart.
# --------------------------------------------------------------------------------------
REPO = ROOT.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _poll(predicate, message: str, timeout: float, delay: float = 0.01, cap: float = 0.25):
    """Wait on ``predicate`` against a monotonic deadline, backing off between polls."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            pytest.fail(f"{message} (waited {timeout:g}s)")
        time.sleep(delay)
        delay = min(delay * 1.5, cap)


def wait_for_job(client, url: str, headers: dict, timeout: float = 120.0) -> dict:
    """Poll one of Person 4's job/export resources until it reports a terminal status."""
    def finished():
        data = client.get(url, headers=headers).json()
        return data if data["status"] in ("completed", "failed") else None

    return _poll(finished, f"{url} did not finish", timeout)


@pytest.fixture(scope="module")
def module_monkeypatch():
    """``monkeypatch`` is function-scoped; the live-server fixtures below are not."""
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def storage_root(tmp_path_factory) -> Path:
    """The single shared volume of ``compose.audio.yaml``: ``./storage`` -> ``/data`` in both
    containers, i.e. Person 4's ``STORAGE_ROOT`` *is* our ``AUDIO_MEDIA_ROOT``."""
    root = tmp_path_factory.mktemp("shared-volume") / "storage"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="module")
def audio_service(storage_root, module_monkeypatch) -> str:
    """This service, live on a free port, with the mock backend and the shared volume."""
    import uvicorn

    module_monkeypatch.setenv("AUDIO_BACKEND", "mock")
    module_monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(storage_root))
    module_monkeypatch.delenv("AUDIO_ALLOWED_MEDIA_HOSTS", raising=False)
    from sam_audio_service.api import create_app
    from sam_audio_service.config import load_settings

    settings = load_settings()
    assert settings.media_root == storage_root.resolve()
    port = _free_port()
    config = uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _poll(lambda: server.started, "audio service did not start", timeout=30.0)
    yield f"http://127.0.0.1:{port}/separate"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def backend_client(storage_root, module_monkeypatch, audio_service):
    """Person 4's FastAPI app (imported from the repo root, never modified) over real HTTP.

    ``app/core.py`` resolves ``STORAGE_ROOT`` into ``ROOT``/``DB`` at import time, so evicting
    every ``app.*`` module before importing it is all the redirection needed.
    """
    module_monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    module_monkeypatch.setenv("INFERENCE_ADAPTER", "http")
    module_monkeypatch.setenv("SAM_AUDIO_SERVICE_URL", audio_service)
    module_monkeypatch.syspath_prepend(str(REPO))
    for name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        sys.modules.pop(name)
    core = importlib.import_module("app.core")
    assert core.ROOT == storage_root.resolve(), "app.core did not pick up STORAGE_ROOT"
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        yield client, core
