import os
import subprocess
import time
import pytest

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('STORAGE_ROOT',str(tmp_path/'storage'))
    import app.core as core
    core.ROOT=tmp_path/'storage'; core.ROOT.mkdir(exist_ok=True); core.DB=core.ROOT/'db.sqlite'
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c: yield c

def wait(client,url,headers):
    for _ in range(200):
        data=client.get(url,headers=headers).json()
        if data['status'] in ('completed','failed'): return data
        time.sleep(.05)
    pytest.fail('job timed out')

def test_workflow(client,tmp_path):
    p=client.post('/api/projects').json(); pid=p['project_id']; headers={'Authorization':'Bearer '+p['access_token']}
    assert client.get('/api/projects/'+pid).status_code==401
    assert client.post(f'/api/projects/{pid}/upload',headers=headers,files={'file':('x.txt',b'x')}).status_code==415
    video=tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(video)],check=True)
    with video.open('rb') as f: r=client.post(f'/api/projects/{pid}/upload',headers=headers,files={'file':('input.mp4',f,'video/mp4')})
    assert r.status_code==200, r.text
    assert r.json()['metadata']['has_video']
    r=client.post(f'/api/projects/{pid}/process',headers=headers,json={'prompts':['dog','music']}); assert r.status_code==202
    job=wait(client,'/api/jobs/'+r.json()['job_id'],headers); assert job['status']=='completed',job
    sid=job['stems'][0]['id']
    r=client.patch(f'/api/projects/{pid}/stems/{sid}',headers=headers,json={'volume':0.5,'automation':[{'time':0,'value':1},{'time':0.9,'value':0.5}]}); assert r.json()['volume']==0.5
    assert client.get(f'/api/projects/{pid}/stems/{sid}/audio',headers=headers).status_code==200
    for text in ['Mute the dog','Unmute the dog','Mute the dog from 0.1 to 0.5 seconds']:
        assert client.post('/api/webhooks/local',json={'project_id':pid,'token':p['access_token'],'text':text}).json()['status']=='applied'
    assert client.post('/api/webhooks/local',json={'project_id':pid,'token':'bad','text':'Mute the dog'}).status_code==401
    assert client.post('/api/webhooks/local',json={'project_id':pid,'token':p['access_token'],'text':'Mute the missing'}).json()['status']=='clarification'
    r=client.post(f'/api/projects/{pid}/export',headers=headers,json={'format':'mp4','include_video':True}); assert r.status_code==202
    result=wait(client,'/api/exports/'+r.json()['export_id'],headers); assert result['status']=='completed',result
    output=client.get(result['download_url'],headers=headers); assert output.status_code==200
    path=tmp_path/'out.mp4'; path.write_bytes(output.content)
    import app.core as core
    assert core.probe(path)['has_video']
    assert client.patch(f'/api/projects/{pid}/stems/bad',headers=headers,json={'muted':True}).status_code==404
    assert client.get('/api/projects/bad',headers=headers).status_code==404

def test_parser():
    from app.commands import parse
    assert parse('Lower the siren volume to 30%')['volume']==.3
    assert parse('Make the music louder')['action']=='louder'
    assert parse('Mute the dog from 15 to 10 seconds') is None
    assert parse('Export the video')['action']=='export'

def test_failed_inference(client,tmp_path,monkeypatch):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'a.wav'; subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://127.0.0.1:1/separate')
    j=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['sound']}).json()['job_id']
    result=wait(client,'/api/jobs/'+j,h); assert result['status']=='failed' and result['error']

def test_audio_mute_export(client,tmp_path):
    import wave
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=1','-ac','2','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    j=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['tone']}).json()['job_id']
    sid=wait(client,'/api/jobs/'+j,h)['stems'][0]['id']
    assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'muted':True}).status_code==200
    eid=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'wav','include_video':False}).json()['export_id']
    result=wait(client,'/api/exports/'+eid,h); assert result['status']=='completed',result
    out=tmp_path/'muted.wav'; out.write_bytes(client.get(result['download_url'],headers=h).content)
    with wave.open(str(out),'rb') as w: assert not any(w.readframes(w.getnframes()))


def test_revision_conflict_and_automation_limit(client, tmp_path):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    j=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog']}).json()['job_id']
    s=wait(client,'/api/jobs/'+j,h)['stems'][0]; sid=s['id']; assert s['revision']==0
    remote=client.post('/api/webhooks/local',json={'project_id':pid,'token':p['access_token'],'text':'Mute the dog'})
    assert remote.json()['status']=='applied'
    stale=client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'expected_revision':0,'volume':.5,'muted':False})
    assert stale.status_code==409
    current=client.get(f'/api/projects/{pid}/stems',headers=h).json()[0]
    assert current['muted'] and current['revision']==1 and current['volume']==1
    fresh=client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'expected_revision':1,'volume':.5})
    assert fresh.json()['revision']==2 and fresh.json()['muted']
    points=[{'time':i*.01,'value':1} for i in range(65)]
    assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'automation':points}).status_code==422


def test_http_inference_requires_prompt_before_job(client, tmp_path, monkeypatch):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http')
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={}).status_code==422
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['  ']}).status_code==422


def test_inference_timeout_is_retryable_by_new_request(client, tmp_path, monkeypatch):
    import httpx
    import app.media as media
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://example.invalid')
    def timeout(*args, **kwargs): raise httpx.ReadTimeout('timed out')
    monkeypatch.setattr(media.httpx, 'post', timeout)
    for _ in range(2):
        response=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['tone']})
        result=wait(client,'/api/jobs/'+response.json()['job_id'],h)
        assert result['status']=='failed' and 'timed out' in result['error']
