import numpy as np
import pytest
from app.event_detection import onset_features,find_similar_events
from app.media import write_wav


def clip():
    rate=16000;x=np.zeros(rate*4,dtype=np.float32)
    for t,f,a in [(.5,1200,.8),(1.2,1200,.4),(2.3,1200,.6),(3.2,300,.7)]:
        n=round(.08*rate);i=round(t*rate);x[i:i+n]+=a*np.exp(-np.arange(n)/(rate*.018))*np.sin(2*np.pi*f*np.arange(n)/rate)
    return x

def test_candidate_onsets_and_example_similarity(tmp_path):
    samples=clip();times,_=onset_features(samples)
    assert np.allclose(times,[.496,1.196,2.296,3.196],atol=.02)
    path=tmp_path/'fixture.wav';write_wav(path,samples[:,None],16000)
    result=find_similar_events(path,4,.5)
    assert np.allclose(result['times'],[.496,1.196,2.296],atol=.02)
    assert result['method']=='spectral-example-matching'

def test_no_clear_example_fails(tmp_path):
    path=tmp_path/'silence.wav';write_wav(path,np.zeros((16000,1)),16000)
    with pytest.raises(ValueError,match='clear sound'):find_similar_events(path,1,.5)

def test_extreme_duration_is_bounded():
    with pytest.raises(ValueError,match='10 minutes'):find_similar_events('unused.wav',601,1)


def test_flexsed_optional_adapter_bounds_and_intervals(tmp_path,monkeypatch):
    import json
    from app.event_detection import flexsed_intervals
    root=tmp_path/'model';root.mkdir();python=tmp_path/'python';python.touch();checkpoint=tmp_path/'weights';checkpoint.touch()
    monkeypatch.setenv('SOUNDCANVAS_FLEXSED_ROOT',str(root));monkeypatch.setenv('SOUNDCANVAS_FLEXSED_CHECKPOINT',str(checkpoint));monkeypatch.setenv('SOUNDCANVAS_FLEXSED_PYTHON',str(python))
    class Result:stdout='other logs\nSOUNDCANVAS_FLEXSED='+json.dumps({'scores':[0,.8,.8,0,0,.7,0],'fps':25,'seconds':1,'peak_gpu_bytes':10})
    monkeypatch.setattr('app.event_detection.subprocess.run',lambda *a,**kw:Result())
    result=flexsed_intervals('ignored.wav',1,'knife chopping')
    assert result['method']=='flexsed-experimental' and len(result['intervals'])==2
    assert result['intervals'][0]['start']==0 and result['intervals'][0]['end']<=.21
