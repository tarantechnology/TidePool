import time
import subprocess
import numpy as np
import pytest

@pytest.fixture
def client(tmp_path,monkeypatch):
    import app.core as core
    core.ROOT=tmp_path/'storage';core.ROOT.mkdir();core.DB=core.ROOT/'db.sqlite'
    monkeypatch.setenv('STORAGE_ROOT',str(core.ROOT));monkeypatch.setenv('INFERENCE_ADAPTER','fake')
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:yield c

def wait(client,cid,h):
    for _ in range(300):
        result=client.get('/api/candidates/'+cid,headers=h).json()
        if result['status'] in ('completed','failed'):return result
        time.sleep(.05)
    pytest.fail('candidate timed out')

def project(client,tmp_path):
    result=client.post('/api/projects').json();pid=result['project_id'];h={'Authorization':'Bearer '+result['access_token']}
    path=tmp_path/'tone.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=12','-ac','2','-ar','48000','-c:a','pcm_f32le',str(path)],check=True)
    with path.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    return pid,h

def test_timestamp_limits_inference_and_preserves_stereo_outside(client,tmp_path,monkeypatch):
    import app.media as media
    import app.candidates as candidates
    real=media.separate;observed=[]
    def instrument(source,directory,payload,report=None):
        from app.core import probe
        observed.append(probe(source)['duration'])
        return real(source,directory,payload,report)
    monkeypatch.setattr(candidates,'separate',instrument)
    monkeypatch.delenv('SOUNDCANVAS_YAMNET_MODEL',raising=False);monkeypatch.delenv('SOUNDCANVAS_YAMNET_ONNX',raising=False)
    pid,h=project(client,tmp_path)
    assert client.post(f'/api/projects/{pid}/candidates',json={'prompt':'dog','processing_time':4}).status_code==401
    res=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking','processing_time':4,'edit_scope':{'action':'mute'}})
    assert res.status_code==202,res.text
    j=wait(client,res.json()['candidate_id'],h);assert j['status']=='completed',j
    assert j['provenance']['processing_interval']=={'start':3.0,'end':5.0}
    assert observed and observed[0] <= 4.01
    a=tmp_path/'a.wav';b=tmp_path/'b.wav';a.write_bytes(client.get(j['original_url'],headers=h).content);b.write_bytes(client.get(j['mix_url'],headers=h).content)
    original,sr=media.read_wav(a);edited,rate=media.read_wav(b)
    assert rate==sr==48000 and original.shape==edited.shape and original.shape[1]==2
    assert np.array_equal(original[:3*sr],edited[:3*sr])
    assert np.array_equal(original[5*sr:],edited[5*sr:])
    assert np.max(np.abs(original[3*sr:5*sr]-edited[3*sr:5*sr]))>0.01

def test_range_validation(client,tmp_path):
    pid,h=project(client,tmp_path);url=f'/api/projects/{pid}/candidates'
    for payload in [ {'prompt':'dog','processing_time':13}, {'prompt':'dog','processing_interval':{'start':1,'end':13}}, {'prompt':'dog','processing_interval':{'start':1,'end':2},'processing_time':1.5}, {'prompt':'dog','processing_interval':{'start':1,'end':2},'visual_prompt':{'track_id':1}} ]:
        assert client.post(url,headers=h,json=payload).status_code==422


def test_anchor_is_made_relative_to_cropped_audio(client,tmp_path,monkeypatch):
    import app.candidates as candidates
    import app.media as media
    real=media.separate;seen=[]
    def instrument(source,directory,payload,report=None):
        seen.append(payload['anchors'])
        return real(source,directory,payload,report)
    monkeypatch.setattr(candidates,'separate',instrument)
    pid,h=project(client,tmp_path)
    req={'prompt':'dog','processing_interval':{'start':4,'end':6},'guidance':{'anchors':[{'token':'+','start':4.5,'end':5.5}]}}
    result=client.post(f'/api/projects/{pid}/candidates',headers=h,json=req)
    j=wait(client,result.json()['candidate_id'],h);assert j['status']=='completed',j
    assert seen==[[{'token':'+','start':1.5,'end':2.5}]]
    assert j['provenance']['inference_interval']=={'start':3.0,'end':7.0}
    assert j['provenance']['anchors'][0]['start']==4.5
