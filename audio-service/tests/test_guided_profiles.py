import numpy as np
import pytest
from sam_audio_service.audio_io import DecodedAudio
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.backends.mock import MockBackend
from sam_audio_service.errors import InvalidRequest


def test_chunk_relative_anchors_and_stereo_reconstruction():
    settings=Settings(chunk_seconds=10,chunk_overlap_seconds=2,chunk_batch=2)
    backend=MockBackend();engine=Engine(settings,backend)
    rng=np.random.default_rng(42);samples=rng.normal(0,.03,(2,22*48000)).astype(np.float32)
    seen=[];original=backend.separate_many
    def capture(signals,prompts,options):
        seen.extend(p.anchors for p in prompts)
        return original(signals,prompts,options)
    backend.separate_many=capture
    result=engine.separate(DecodedAudio(samples,48000,48000,2),SeparationMetadata.model_validate({'prompts':['dog'],'anchors':[{'token':'+','start':9,'end':11},{'token':'-','start':18,'end':19}],'options':{'chunk_seconds':10,'chunk_batch':2}}))
    assert seen==[[('+',9,10)],[('+',1,3)],[('-',2,3)]]
    assert np.max(np.abs(sum(s.samples for s in result.stems)-samples))<1e-6
    assert not np.array_equal(result.stems[-1].samples[0],result.stems[-1].samples[1])
    assert result.model['effective_chunk_batch']==2 and result.model['chunk_seconds']==10


def test_service_rejects_outside_anchor():
    engine=Engine(Settings(),MockBackend())
    audio=DecodedAudio(np.ones((1,48000),np.float32)*.01,48000,48000,1)
    with pytest.raises(InvalidRequest,match='anchor extends'):
        engine.separate(audio,SeparationMetadata.model_validate({'prompts':['voice'],'anchors':[{'start':.5,'end':2}]}))


def test_explicit_long_profile_uses_batch_one():
    backend=MockBackend();engine=Engine(Settings(),backend)
    calls=[];base=backend.separate_many
    def capture(signals,prompts,options):
        calls.append(len(signals));return base(signals,prompts,options)
    backend.separate_many=capture
    audio=DecodedAudio(np.ones((2,24*48000),np.float32)*.01,48000,48000,2)
    result=engine.separate(audio,SeparationMetadata.model_validate({'prompts':['voice'],'options':{'chunk_seconds':20,'chunk_batch':1}}))
    assert calls==[1,1] and result.model['effective_chunk_batch']==1


def test_visual_request_never_silently_becomes_text_only():
    engine=Engine(Settings(),MockBackend())
    audio=DecodedAudio(np.ones((1,48000),np.float32)*.01,48000,48000,1)
    with pytest.raises(InvalidRequest,match='Visual guidance could not be used'):
        engine.separate(audio,SeparationMetadata.model_validate({'prompts':['voice'],'visual_prompt':{'kind':'visual_selection','interval':{'start_seconds':0,'end_seconds':1},'label':'person'}}))


def test_category_gate_blocks_regression_and_missing_heldout():
    from benchmarks.guided import CATEGORIES, gate
    rows=[]
    for category in CATEGORIES:
        for profile,score in [('text',5),('judge',4.6)]:
            rows.append(dict(id=category,category=category,profile=profile,split='heldout',target_si_sdri_db=score,background_si_sdri_db=score))
    assert gate(rows)['judge']['numerical_gate_passes']
    assert not gate(rows)['judge']['promoted']
    rows[-1]['target_si_sdri_db']=4.4
    assert not gate(rows)['judge']['numerical_gate_passes']
    assert not gate(rows[:-2])['judge']['numerical_gate_passes']
