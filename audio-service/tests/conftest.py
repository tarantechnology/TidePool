import os
import subprocess
import sys
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
