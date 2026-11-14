import numpy as np
from fastapi.testclient import TestClient

from sam_audio_service.api import create_app
from sam_audio_service.audio_io import DecodedAudio, encode_wav_float32
from sam_audio_service.backends.audiosep import AudioSepBackend
from sam_audio_service.backends.base import BackendResult
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine


def fixture():
    time=np.arange(48000)/48000
    speech=.1*np.sin(2*np.pi*300*time)
    barking=.3*np.sin(2*np.pi*2400*time)*(time>.5)
    source=np.stack([speech+barking, .7*speech+barking]).astype(np.float32)
    estimate=(barking+.01*speech).astype(np.float32)
    backend=AudioSepBackend('', 'audiosep-base-4M', device='cpu')
    backend.model=object()  # Injected estimate tests engine integration, not real inference.
    backend.separate=lambda mixture,prompt,options: BackendResult(target=estimate.copy())
    return source,estimate,backend


def test_default_audiosep_suppression_reaches_api_and_preserves_stereo_timeline():
    source,estimate,backend=fixture()
    settings=Settings(backend='audiosep',model_id='audiosep-base-4M',load_on_startup=False,chunk_seconds_env=0)
    engine=Engine(settings,backend)
    audio=DecodedAudio(source,48000,48000,2)
    result=engine.separate(audio,SeparationMetadata.model_validate({'prompts':['dog barking'],'options':{'refine':'mask'}}))
    assert result.model['suppression']['strength']==.9
    assert result.model['refine'] is None  # The legacy mask is not applied a second time.
    assert len(result.stems)==2
    np.testing.assert_allclose(sum(s.samples for s in result.stems),source,atol=1e-7)
    residual=result.stems[-1].samples
    np.testing.assert_array_equal(residual[:,1000:20000],source[:,1000:20000])
    assert not np.array_equal(residual[0],residual[1])
    assert np.max(np.abs(result.stems[0].samples))>0
    with TestClient(create_app(settings,engine)) as client:
        response=client.post('/separate',files={'audio':('mix.wav',encode_wav_float32(source,48000),'audio/wav')},data={'metadata':'{"prompts":["dog barking"]}'})
        assert response.status_code==200,response.text
        body=response.json()
        assert body['model']['suppression']['mode']=='conservative'
        assert all(s['sample_count']==48000 and s['channels']==2 for s in body['stems'])


def test_explicit_raw_mode_remains_available_for_reproduction():
    source,estimate,backend=fixture()
    engine=Engine(Settings(backend='audiosep',model_id='audiosep-base-4M',chunk_seconds_env=0),backend)
    result=engine.separate(DecodedAudio(source,48000,48000,2),SeparationMetadata.model_validate({'prompts':['dog barking'],'options':{'suppression':'off','refine':'off'}}))
    assert result.model['suppression'] is None
    np.testing.assert_array_equal(result.stems[-1].samples,source-estimate[None])
