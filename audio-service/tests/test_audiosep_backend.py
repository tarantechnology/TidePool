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
        assert caps['prompting_modes'] == {'text':True,'span_anchors':False,'span_prediction':False,'visual':False}


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


def test_audiosep_is_default_and_can_be_explicitly_overridden(monkeypatch):
    for name in ['AUDIO_BACKEND','AUDIO_MODEL_ID','AUDIO_SUPPRESSION','AUDIO_CHUNK_BATCH','AUDIO_CHUNK_WORKERS']:
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.backend == 'audiosep' and settings.model_id == 'audiosep-base-4M'
    assert settings.suppression == 'conservative' and settings.suppression_strength == .9
    assert settings.chunk_batch == 1
    monkeypatch.setenv('AUDIO_BACKEND','mock')
    assert load_settings().suppression == 'off'


def test_existing_checkpoint_is_reused_and_missing_explicit_path_fails(tmp_path, monkeypatch):
    from sam_audio_service.backends.audiosep import checkpoint_path
    folder=tmp_path/'checkpoint';folder.mkdir()
    checkpoint=folder/'weights.ckpt';checkpoint.write_bytes(b'fixture')
    monkeypatch.delenv('AUDIO_AUDIOSEP_CHECKPOINT',raising=False)
    assert checkpoint_path(tmp_path,'AUDIO_AUDIOSEP_CHECKPOINT','weights.ckpt') == checkpoint
    monkeypatch.setenv('AUDIO_AUDIOSEP_CHECKPOINT',str(tmp_path/'missing.ckpt'))
    with pytest.raises(RuntimeError,match='missing checkpoint'):
        checkpoint_path(tmp_path,'AUDIO_AUDIOSEP_CHECKPOINT','weights.ckpt')


def test_checkpoint_download_is_pinned_to_official_revision(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from sam_audio_service.backends.audiosep import checkpoint_path, CHECKPOINT_REVISION
    calls=[]
    def download(*args,**kwargs):
        calls.append((args,kwargs));return str(tmp_path/'downloaded')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(hf_hub_download=download))
    monkeypatch.delenv('AUDIO_AUDIOSEP_CHECKPOINT',raising=False)
    assert checkpoint_path(tmp_path,'AUDIO_AUDIOSEP_CHECKPOINT','weights.ckpt') == tmp_path/'downloaded'
    assert calls == [(('Audio-AGI/AudioSep','checkpoint/weights.ckpt'),{'repo_type':'space','revision':CHECKPOINT_REVISION})]
