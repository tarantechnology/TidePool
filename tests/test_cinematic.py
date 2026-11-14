import subprocess
import time
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('INFERENCE_ADAPTER','fake')
    monkeypatch.setenv('AUDIO_BACKEND','mock')
    import app.core as core
    core.ROOT = tmp_path/'storage'; core.ROOT.mkdir(); core.DB = core.ROOT/'db.sqlite'
    monkeypatch.setenv('STORAGE_ROOT',str(core.ROOT))
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c: yield c


def uploaded(client, tmp_path):
    p = client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    path=tmp_path/'clip.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=1','-ac','2','-ar','48000',str(path)],check=True)
    with path.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('clip.wav',f)}).status_code==200
    return pid,h


def wait(client,cid,h):
    for _ in range(200):
        j=client.get('/api/candidates/'+cid,headers=h).json()
        if j['status'] in ('completed','failed'): return j
        time.sleep(.05)
    pytest.fail('preview timed out')


def test_preview_apply_undo_authorization_and_export(client,tmp_path):
    pid,h=uploaded(client,tmp_path)
    assert client.post(f'/api/projects/{pid}/cinematic-preview').status_code==401
    assert client.post(f'/api/projects/{pid}/cinematic-preview',headers=h,json={'style':'invalid'}).status_code==422
    response=client.post(f'/api/projects/{pid}/cinematic-preview',headers=h,json={'style':'cinematic'})
    assert response.status_code==202,response.text
    cid=response.json()['candidate_id']; job=wait(client,cid,h)
    assert job['status']=='completed',job
    assert job['provenance']['kind']=='cinematic' and job['provenance']['analysis']['mean_dbfs'] < 0
    assert client.get(job['mix_url']).status_code==401
    assert client.get(job['mix_url'],headers=h).content[:4]==b'RIFF'
    assert client.get(job['original_url'],headers=h).content[:4]==b'RIFF'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    assert len(client.get(f'/api/projects/{pid}/stems',headers=h).json())==1
    result=client.post(f'/api/candidates/{cid}/undo',headers=h)
    assert result.status_code==200,result.text
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_stale_preview_rejected_and_existing_edits_preserved(client,tmp_path):
    pid,h=uploaded(client,tmp_path)
    j=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['music']}).json()['job_id']
    for _ in range(200):
        result=client.get('/api/jobs/'+j,headers=h).json()
        if result['status'] in ('completed','failed'): break
        time.sleep(.05)
    assert result['status']=='completed',result
    stems=client.get(f'/api/projects/{pid}/stems',headers=h).json(); sid=stems[0]['id']
    changed=client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'expected_revision':stems[0]['revision'],'volume':.5,'automation':[{'time':0,'value':.8}]}).json()
    cid=client.post(f'/api/projects/{pid}/cinematic-preview',headers=h,json={'style':'subtle'}).json()['candidate_id']
    assert wait(client,cid,h)['status']=='completed'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    after=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert len(after)==len(stems)
    same=next(s for s in after if s['id']==sid)
    assert same['volume']==.5 and same['automation']==[{'time':0,'value':.8}]
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    cid2=client.post(f'/api/projects/{pid}/cinematic-preview',headers=h).json()['candidate_id']
    assert wait(client,cid2,h)['status']=='completed'
    current=client.get(f'/api/projects/{pid}/stems',headers=h).json()[0]
    assert client.patch(f'/api/projects/{pid}/stems/{current["id"]}',headers=h,json={'expected_revision':current['revision'],'volume':.25}).status_code==200
    assert client.post(f'/api/candidates/{cid2}/apply',headers=h).status_code==409


def test_silence_fails_without_changing_project(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    path=tmp_path/'silence.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-t','1',str(path)],check=True)
    with path.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('silence.wav',f)}).status_code==200
    cid=client.post(f'/api/projects/{pid}/cinematic-preview',headers=h).json()['candidate_id']
    result=wait(client,cid,h)
    assert result['status']=='failed' and 'silent' in result['error'].lower()
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==409


def test_agent_can_plan_cinematic_without_unvalidated_operations(client,monkeypatch):
    import json
    import app.agent as agent
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key')
    output={'kind':'cinematic','description':None,'remove':False,'actions':[],'message':'Prepare a preview.'}
    class MockResponse:
        is_error=False
        def json(self): return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(output)}]}]}
    monkeypatch.setattr(agent.httpx,'post',lambda *args,**kwargs:MockResponse())
    body={'prompt':'make it feel like a film','stems':[],'tracks':[],'duration':10}
    result=client.post(f'/api/projects/{pid}/agent',headers=h,json=body)
    assert result.status_code==200 and result.json()['kind']=='cinematic'
    output['actions']=[{'type':'set_gain','stemId':'invented','value':1}]
    assert client.post(f'/api/projects/{pid}/agent',headers=h,json=body).status_code==502


def test_cinematic_planner_does_not_ignore_extra_constraints(client,monkeypatch):
    import json
    import app.agent as agent
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key')
    output={'kind':'cinematic','description':None,'remove':False,'actions':[],'message':'Make cinematic.'}
    class Response:
        is_error=False
        def json(self): return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(output)}]}]}
    monkeypatch.setattr(agent.httpx,'post',lambda *args,**kwargs:Response())
    body={'prompt':'make it cinematic but keep the music unchanged','stems':[],'tracks':[],'duration':10}
    result=client.post(f'/api/projects/{pid}/agent',headers=h,json=body)
    assert result.status_code==200 and result.json()['kind']=='clarify'


def test_video_cinematic_export_keeps_video_and_audio(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    path=tmp_path/'video.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(path)],check=True)
    with path.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('video.mp4',f)}).status_code==200
    cid=client.post(f'/api/projects/{pid}/cinematic-preview',headers=h).json()['candidate_id']
    assert wait(client,cid,h)['status']=='completed'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    response=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'mp4','include_video':True})
    assert response.status_code==202,response.text
    eid=response.json()['export_id']
    for _ in range(200):
        job=client.get('/api/exports/'+eid,headers=h).json()
        if job['status'] in ('completed','failed'): break
        time.sleep(.05)
    assert job['status']=='completed',job
    result=client.get(job['download_url'],headers=h)
    out=tmp_path/'out.mp4';out.write_bytes(result.content)
    from app.core import probe
    metadata=probe(out)
    assert metadata['has_video'] and metadata['duration']>0.9


def test_manual_effects_preview_only_changes_selected_layer_and_undo(client,tmp_path,monkeypatch):
    monkeypatch.setenv('INFERENCE_ADAPTER','fake')
    pid,h=uploaded(client,tmp_path)
    j=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['music','speech']}).json()['job_id']
    for _ in range(200):
        result=client.get('/api/jobs/'+j,headers=h).json()
        if result['status'] in ('completed','failed'): break
        time.sleep(.05)
    assert result['status']=='completed',result
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json(); assert len(rows)==2
    chosen,other=rows
    before=client.get(chosen['audio_url'],headers=h).content
    other_before=client.get(other['audio_url'],headers=h).content
    url=f'/api/projects/{pid}/effects-preview'
    assert client.post(url,json={'stem_id':chosen['id'],'bass':6}).status_code==401
    assert client.post(url,headers=h,json={'stem_id':chosen['id'],'bass':13}).status_code==422
    assert client.post(url,headers=h,json={'stem_id':'not-a-layer'}).status_code==422
    cid=client.post(url,headers=h,json={'stem_id':chosen['id'],'bass':6,'treble':-2,'highpass':45,'compression':1.5}).json()['candidate_id']
    result=wait(client,cid,h);assert result['status']=='completed',result
    assert result['provenance']['kind']=='effects'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    after=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert client.get(next(r['audio_url'] for r in after if r['id']==chosen['id']),headers=h).content != before
    assert client.get(next(r['audio_url'] for r in after if r['id']==other['id']),headers=h).content == other_before
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    assert client.get(chosen['audio_url'],headers=h).content==before


def test_manual_effects_on_unseparated_audio_and_stale_preview(client,tmp_path):
    pid,h=uploaded(client,tmp_path);url=f'/api/projects/{pid}/effects-preview'
    cid=client.post(url,headers=h,json={'bass':4}).json()['candidate_id']
    assert wait(client,cid,h)['status']=='completed'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    row=client.get(f'/api/projects/{pid}/stems',headers=h).json()[0]
    cid2=client.post(url,headers=h,json={'stem_id':row['id'],'treble':3}).json()['candidate_id']
    assert wait(client,cid2,h)['status']=='completed'
    assert client.patch(f'/api/projects/{pid}/stems/{row["id"]}',headers=h,json={'expected_revision':row['revision'],'volume':.5}).status_code==200
    assert client.post(f'/api/candidates/{cid2}/apply',headers=h).status_code==409
