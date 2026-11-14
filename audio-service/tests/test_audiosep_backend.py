import pytest
from fastapi.testclient import TestClient

from sam_audio_service.api import create_app
from sam_audio_service.backends import create_backend
from sam_audio_service.backends.audiosep import AudioSepBackend
from sam_audio_service.backends.base import BackendOptions, BackendPrompt
from sam_audio_service.config import load_settings
from sam_audio_service.engine import Engine
from sam_audio_service.errors import UnsupportedOperation


def test_audiosep_config_and_capabilities(monkeypatch):
    monkeypatch.setenv('AUDIO_BACKEND', 'audiosep')
    monkeypatch.delenv('AUDIO_MODEL_ID', raising=False)
    monkeypatch.delenv('AUDIO_CHUNK_SECONDS', raising=False)
    monkeypatch.delenv('AUDIO_CHUNK_BATCH', raising=False)
    monkeypatch.delenv('AUDIO_CHUNK_WORKERS', raising=False)
    monkeypatch.setenv('AUDIO_LOAD_ON_STARTUP', 'false')
    settings = load_settings()
    assert settings.model_id == 'audiosep-base-4M'
    assert settings.chunk_seconds == 10 and settings.chunk_batch == 1
    assert settings.refine == 'off'
    backend = create_backend(settings)
    assert isinstance(backend, AudioSepBackend) and not backend.loaded
    with TestClient(create_app(settings, Engine(settings, backend))) as client:
        caps = client.get('/capabilities').json()
        assert caps['backend'] == 'audiosep' and caps['is_mock'] is False
        assert caps['independent_candidates'] is False
        assert caps['prompting_modes'] == {'text':True,'span_anchors':False,'span_prediction':False,'visual':False,'tagging':True,'explode':True}


def test_unavailable_audiosep_checkout_has_actionable_error():
    backend = AudioSepBackend('', 'audiosep-base-4M')
    with pytest.raises(RuntimeError, match='AUDIO_AUDIOSEP_ROOT'):
        backend.load()


def test_audiosep_rejects_unsupported_features_without_loading_torch():
    import numpy as np
    backend = AudioSepBackend('', 'audiosep-base-4M')
    backend.model = object()
    signal = np.zeros(24, dtype=np.float32)
    with pytest.raises(UnsupportedOperation, match='text prompts only'):
        backend.separate(signal, BackendPrompt(text='dog', anchors=[('+', 0, 1)]), BackendOptions())
    with pytest.raises(UnsupportedOperation, match='ranked or independent'):
        backend.separate(signal, BackendPrompt(text='dog'), BackendOptions(candidates=2))
