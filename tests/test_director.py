import subprocess
import time
import numpy as np
import pytest

@pytest.fixture
def client(tmp_path,monkeypatch):
    import app.core as core
    core.ROOT=tmp_path/'storage';core.ROOT.mkdir();core.DB=core.ROOT/'db.sqlite'
    monkeypatch.setenv('STORAGE_ROOT',str(core.ROOT));monkeypatch.delenv('OPENAI_API_KEY',raising=False);monkeypatch.delenv('OPENAI_API_KEY_FILE',raising=False)
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:yield c

def upload(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    path=tmp_path/'tone.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=4','-ac','2','-ar','48000','-c:a','pcm_f32le',str(path)],check=True)
    with path.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    return pid,h

def wait(client,cid,h):
    for _ in range(300):
        data=client.get('/api/candidates/'+cid,headers=h).json()
        if data['status'] in ('completed','failed'):return data
        time.sleep(.05)
    pytest.fail('director timed out')

def test_director_raises_detected_event_and_keeps_surroundings(client,tmp_path,monkeypatch):
    import app.discovery as discovery
    import app.candidates as candidates
    from app.core import run,uid
    def detected(*args,**kwargs):return {'sources':[{'label':'boing','prompt':'boing','intervals':[{'start':1,'end':1.5}],'interval_scores':[.9],'score':.9}]}
    def separate(source,directory,payload,report=None):
        target=directory/(uid()+'.wav');residual=directory/(uid()+'.wav')
        run('ffmpeg','-v','error','-y','-i',str(source),'-af','volume=0.25','-c:a','pcm_f32le',str(target))
        run('ffmpeg','-v','error','-y','-i',str(source),'-af','volume=0.75','-c:a','pcm_f32le',str(residual))
        if report is not None:report.update(model={'backend':'fake','is_mock':True},warnings=['Mock model'])
        return [{'name':'boing','path':target},{'name':'residual','path':residual}]
    monkeypatch.setattr(discovery,'discover',detected);monkeypatch.setattr(candidates,'separate',separate)
    pid,h=upload(client,tmp_path)
    assert client.post(f'/api/projects/{pid}/director-preview',json={'request':'emphasize the funny sounds','intent':'funny'}).status_code==401
    r=client.post(f'/api/projects/{pid}/director-preview',headers=h,json={'request':'emphasize the funny sounds','intent':'funny'})
    assert r.status_code==202,r.text
    cid=r.json()['candidate_id'];job=wait(client,cid,h)
    assert job['status']=='completed',job
    assert job['provenance']['kind']=='director' and job['provenance']['changes'][0]['label']=='boing'
    assert job['inference']['model']['is_mock'] is True
    from app.media import read_wav
    a=tmp_path/'before.wav';b=tmp_path/'after.wav';a.write_bytes(client.get(job['original_url'],headers=h).content);b.write_bytes(client.get(job['mix_url'],headers=h).content)
    before,sr=read_wav(a);after,_=read_wav(b)
    assert np.array_equal(before[:int(.8*sr)],after[:int(.8*sr)])
    assert np.array_equal(before[int(1.7*sr):],after[int(1.7*sr):])
    assert np.max(np.abs(before[int(.9*sr):int(1.6*sr)]-after[int(.9*sr):int(1.6*sr)]))>0.001
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert len(rows)==2 and any(s['name']=='boing' and s['volume']==1.35 for s in rows)
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]

def test_director_does_not_invent_funny_sounds(client,tmp_path,monkeypatch):
    import app.discovery as discovery
    monkeypatch.setattr(discovery,'discover',lambda *args,**kwargs:{'sources':[]})
    pid,h=upload(client,tmp_path)
    cid=client.post(f'/api/projects/{pid}/director-preview',headers=h,json={'request':'make it funny','intent':'funny'}).json()['candidate_id']
    result=wait(client,cid,h)
    assert result['status']=='failed' and 'could not reliably' in result['error'].lower()
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_repeated_events_merge_into_one_editable_layer(client,tmp_path,monkeypatch):
    import app.discovery as discovery
    import app.candidates as candidates
    import app.director as director
    from app.core import run,uid
    monkeypatch.setattr(discovery,'discover',lambda *a,**k:{'sources':[{'label':'boing','prompt':'boing','intervals':[{'start':.6,'end':1},{'start':2,'end':2.4}],'interval_scores':[.9,.8],'score':.9}]})
    monkeypatch.setattr(director,'select',lambda request,events,cuts,visual=None:([{**e,'action':'raise'} for e in events],''))
    def separate(source,directory,payload,report=None):
        a=directory/(uid()+'.wav');b=directory/(uid()+'.wav')
        for path,volume in ((a,.25),(b,.75)):
            run('ffmpeg','-v','error','-y','-i',str(source),'-af',f'volume={volume}','-c:a','pcm_f32le',str(path))
        if report is not None:report.update(model={'backend':'fake','is_mock':True})
        return [{'name':'boing','path':a},{'name':'residual','path':b}]
    monkeypatch.setattr(candidates,'separate',separate)
    pid,h=upload(client,tmp_path)
    cid=client.post(f'/api/projects/{pid}/director-preview',headers=h,json={'request':'emphasize funny sounds','intent':'funny'}).json()['candidate_id']
    job=wait(client,cid,h);assert job['status']=='completed',job
    assert len(job['provenance']['changes'])==1 and len(job['provenance']['changes'][0]['intervals'])==2
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    stems=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert len(stems)==2 and len([s for s in stems if s['name']=='boing'])==1


def test_director_adjust_only_accepts_own_proposed_changes(client,tmp_path,monkeypatch):
    import app.discovery as discovery
    import app.candidates as candidates
    from app.core import run,uid
    monkeypatch.setattr(discovery,'discover',lambda *a,**k:{'sources':[{'label':'boing','prompt':'boing','intervals':[{'start':1,'end':1.5}],'interval_scores':[.9],'score':.9}]})
    def separate(source,directory,payload,report=None):
        a=directory/(uid()+'.wav');b=directory/(uid()+'.wav')
        for path,volume in ((a,.25),(b,.75)):run('ffmpeg','-v','error','-y','-i',str(source),'-af',f'volume={volume}','-c:a','pcm_f32le',str(path))
        if report is not None:report.update(model={'backend':'fake','is_mock':True})
        return [{'name':'boing','path':a},{'name':'residual','path':b}]
    monkeypatch.setattr(candidates,'separate',separate)
    pid,h=upload(client,tmp_path)
    cid=client.post(f'/api/projects/{pid}/director-preview',headers=h,json={'request':'emphasize funny sounds','intent':'funny'}).json()['candidate_id']
    first=wait(client,cid,h);assert first['status']=='completed',first
    sid=first['provenance']['changes'][0]['target_stem_id']
    assert client.post(f'/api/candidates/{cid}/director-adjust',json={'stem_id':sid,'enabled':False}).status_code==401
    assert client.post(f'/api/candidates/{cid}/director-adjust',headers=h,json={'stem_id':'invalid','enabled':False}).status_code==422
    res=client.post(f'/api/candidates/{cid}/director-adjust',headers=h,json={'stem_id':sid,'enabled':False})
    assert res.status_code==202,res.text
    new=wait(client,res.json()['candidate_id'],h);assert new['status']=='completed',new
    assert new['provenance']['changes'][0]['enabled'] is False
    assert client.post('/api/candidates/'+new['id']+'/apply',headers=h).status_code==200
    stems=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert next(s for s in stems if s['id']==sid)['volume']==1
