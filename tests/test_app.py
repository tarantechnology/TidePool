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
    samples=subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','f32le','-'])
    assert not any(samples)


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
    remote_range=client.post('/api/webhooks/local',json={'project_id':pid,'token':p['access_token'],'text':'Mute the dog from 0.1 to 0.5 seconds'})
    assert remote_range.json()['status']=='applied'
    assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'mute_ranges':[[0.8,0.2]]}).status_code==422
    assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'mute_ranges':[[0.1,2]]}).status_code==422
    saved=client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'expected_revision':3,'name':'bark','start_offset':0.2,'mute_ranges':[]})
    assert saved.status_code==200 and saved.json()['name']=='bark' and saved.json()['mute_ranges']==[]
    points=[{'time':i*.01,'value':1} for i in range(65)]
    assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'automation':points}).status_code==422
    points=[{'time':i*.01,'value':float(i % 2)} for i in range(64)]
    assert client.patch(f'/api/projects/{pid}/stems/{sid}',headers=h,json={'automation':points,'expected_revision':4}).status_code==200
    eid=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'wav','include_video':False}).json()['export_id']
    assert wait(client,'/api/exports/'+eid,h)['status']=='completed'


def test_http_inference_requires_prompt_before_job(client, tmp_path, monkeypatch):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http')
    # With no prompts, only prompted mode is refused up front; auto lets the service route.
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'prompted'}).status_code==422
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


def test_existing_database_gains_revision_column(tmp_path, monkeypatch):
    import sqlite3
    import app.core as core
    path=tmp_path/'legacy.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE stems(id TEXT PRIMARY KEY)')
        db.execute("INSERT INTO stems(id) VALUES('old')")
    monkeypatch.setattr(core,'DB',path)
    core.init()
    with core.connect() as db:
        assert db.execute("SELECT revision FROM stems WHERE id='old'").fetchone()['revision']==0
    core.init()  # migration is safe to repeat


def test_project_resume_and_original_media(client, tmp_path):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    video=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=s=64x64:r=10:d=1','-f','lavfi','-i','sine=duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(video)],check=True)
    with video.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.mp4',f)}).status_code==200
    assert client.get(f'/api/projects/{pid}/original').status_code==401
    response=client.get(f'/api/projects/{pid}/original',headers=h)
    assert response.status_code==200 and response.content==video.read_bytes()
    assert client.get(f'/api/projects/{pid}/preview').status_code==401
    import app.core as core
    prepared=core.project_dir(pid)/'extracted'/'preview.mp4'
    assert prepared.exists(), 'upload should transcode the preview, not GET /preview'
    preview=client.get(f'/api/projects/{pid}/preview',headers=h); assert preview.status_code==200
    path=tmp_path/'preview.mp4'; path.write_bytes(preview.content)
    codec=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name','-of','csv=p=0',str(path)],text=True).strip()
    assert codec=='h264'
    job=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog']}).json()['job_id']
    assert wait(client,'/api/jobs/'+job,h)['status']=='completed'
    project=client.get(f'/api/projects/{pid}',headers=h).json()
    assert project['latest_job']=={'id':job,'status':'completed'} and project['last_completed_job_id']==job


def test_preview_of_odd_dimension_video(client, tmp_path):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    video=tmp_path/'odd.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=green:s=65x65:r=10:d=1',
                    '-f','lavfi','-i','sine=duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(video)],check=True)
    with video.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('odd.mp4',f)}).status_code==200
    import app.core as core
    prepared=core.project_dir(pid)/'extracted'/'preview.mp4'
    assert prepared.exists()
    response=client.get(f'/api/projects/{pid}/preview',headers=h)
    assert response.status_code==200, response.text
    out=tmp_path/'preview.mp4'; out.write_bytes(response.content)
    # The preview is video-only, so probe the stream directly: yuv420p needs even sides.
    size=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                                  'stream=width,height,codec_name','-of','csv=p=0',str(out)],text=True).strip()
    assert size=='h264,64,64', size
    # Pre-existing projects without a stored preview transcode lazily on first request.
    prepared.unlink()
    assert client.get(f'/api/projects/{pid}/preview',headers=h).status_code==200
    assert prepared.exists()


def test_visual_analysis_job_orchestration(client, tmp_path, monkeypatch):
    import json
    from pathlib import Path
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    video=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=s=64x64:r=10:d=1','-f','lavfi','-i','sine=duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(video)],check=True)
    with video.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.mp4',f)}).status_code==200
    import subprocess as sp
    import app.detector as detector
    monkeypatch.setattr(detector,'_import_error',None)  # Installed and importable.
    def fake_spawn(args, timeout, **kwargs):
        assert 'soundcanvas_cv' in args and '--video-id' in args
        output=Path(args[args.index('--output-dir')+1]); output.mkdir(parents=True)
        (output/'detections.json').write_text(json.dumps({'analyzed_frames':10,'processed_duration_seconds':1,'frames':[]}))
        return sp.CompletedProcess(args,0,'','')
    monkeypatch.setattr(detector,'_spawn',fake_spawn)
    response=client.post(f'/api/projects/{pid}/visual-analyze',headers=h); assert response.status_code==202
    job=wait(client,'/api/visual-jobs/'+response.json()['job_id'],h)
    assert job['status']=='completed' and job['result']['analyzed_frames']==10
    assert client.get(f'/api/projects/{pid}/visual-metadata',headers=h).json()['analyzed_frames']==10


def test_agent_requires_project_access_and_sanitizes_model_output(client, monkeypatch):
    import json
    import app.agent as agent
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key')
    response={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'kind':'separate','description':'a person narrating','remove':True,'actions':[],'message':'I will separate the narration.'})}]}]}
    class MockResponse:
        is_error=False
        def json(self): return response
    captured={}
    def fake_post(url, **kwargs):
        captured.update(kwargs['json'])
        return MockResponse()
    monkeypatch.setattr(agent.httpx,'post',fake_post)
    body={'prompt':'remove the narrator','stems':[],'tracks':[],'duration':10,'has_project_stems':False}
    assert client.post(f'/api/projects/{pid}/agent',json=body).status_code==401
    r=client.post(f'/api/projects/{pid}/agent',headers=h,json=body)
    assert r.status_code==200 and r.json()['description']=='a person narrating'
    assert captured['model']=='gpt-6-luna' and captured['store'] is False
    response['output'][0]['content'][0]['text']='not json'
    assert client.post(f'/api/projects/{pid}/agent',headers=h,json=body).status_code==502


def test_add_separation_preserves_existing_layers_and_edits(client, tmp_path, monkeypatch):
    import shutil
    import app.worker as worker
    def separate(source, directory, payload, report=None):
        directory.mkdir(exist_ok=True, parents=True)
        result=[]
        for name in (payload['prompts'][0], 'residual'):
            path=directory/(worker.uid()+'.wav'); shutil.copyfile(source,path); result.append((name,path))
        return result
    monkeypatch.setattr(worker,'separate',separate)
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'input.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('input.wav',f)}).status_code==200
    first=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['voice']}).json()['job_id']
    first_stems=wait(client,'/api/jobs/'+first,h)['stems']; voice=first_stems[0]; residual=first_stems[1]
    assert client.patch(f"/api/projects/{pid}/stems/{voice['id']}",headers=h,json={'volume':.4}).status_code==200
    assert client.patch(f"/api/projects/{pid}/stems/{residual['id']}",headers=h,json={'volume':.8}).status_code==200
    second=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog'],'append':True}).json()['job_id']
    result=wait(client,'/api/jobs/'+second,h); assert result['status']=='completed',result
    stems={s['name']:s for s in result['stems']}
    assert set(stems)=={'voice','dog','residual'}
    assert stems['voice']['id']==voice['id'] and stems['voice']['volume']==.4
    assert stems['dog']['volume']==.8 and stems['residual']['volume']==.8
    assert stems['residual']['id']!=residual['id']


def test_add_separation_fails_if_residual_changed_during_inference(client, tmp_path, monkeypatch):
    import shutil
    import app.worker as worker
    def separate(source,directory,payload,report=None):
        if payload['prompts']==['dog']:
            with worker.connect() as db: db.execute("UPDATE stems SET revision=revision+1 WHERE name='residual'")
        result=[]
        for name in (payload['prompts'][0], 'residual'):
            path=directory/(worker.uid()+'.wav'); shutil.copyfile(source,path); result.append((name,path))
        return result
    monkeypatch.setattr(worker,'separate',separate)
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'input.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('input.wav',f)}).status_code==200
    first=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['voice']}).json()['job_id'];wait(client,'/api/jobs/'+first,h)
    second=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog'],'append':True}).json()['job_id']
    result=wait(client,'/api/jobs/'+second,h);assert result['status']=='failed' and 'preserved' in result['error']
    assert {s['name'] for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}=={'voice','residual'}


def test_prepare_visual_copy_preserves_original_and_validates_timeline(client,tmp_path):
    from pathlib import Path
    import app.core as core
    from app.visual import validate_timeline
    video=tmp_path/'offset.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=blue:s=64x64:r=10:d=1','-f','lavfi','-i','sine=duration=1','-c:v','mpeg4','-c:a','aac','-avoid_negative_ts','make_zero','-shortest',str(video)],check=True)
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    with video.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('offset.mp4',f)}).status_code==200
    with core.connect() as db: old=db.execute('SELECT original FROM projects WHERE id=?',(pid,)).fetchone()['original']
    started=client.post(f'/api/projects/{pid}/prepare-visual',headers=h);assert started.status_code==202
    job=wait(client,'/api/prepare-jobs/'+started.json()['job_id'],h);assert job['status']=='completed',job
    with core.connect() as db: new=db.execute('SELECT original,metadata FROM projects WHERE id=?',(pid,)).fetchone()
    assert old!=new['original'] and Path(old).exists() and Path(new['original']).exists()
    import json
    validate_timeline(new['original'],json.loads(new['metadata'])['frame_rate'])
    assert client.get(f'/api/projects/{pid}/original-audio',headers=h).status_code==200


def test_retry_preview_preserves_edits_and_rejects_stale_apply(client, tmp_path, monkeypatch):
    import shutil
    import app.worker as worker
    def separate(source, directory, payload, report=None):
        directory.mkdir(exist_ok=True, parents=True)
        result = []
        for name in (payload['prompts'][0], 'residual'):
            path = directory / (worker.uid() + '.wav')
            shutil.copyfile(source, path)
            result.append((name, path))
        return result
    monkeypatch.setattr(worker, 'separate', separate)
    project = client.post('/api/projects').json()
    pid = project['project_id']
    headers = {'Authorization': 'Bearer ' + project['access_token']}
    audio = tmp_path / 'input.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=1',str(audio)],check=True)
    with audio.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/upload', headers=headers, files={'file':('input.wav',f)}).status_code == 200
    job = client.post(f'/api/projects/{pid}/process', headers=headers, json={'prompts':['voice']}).json()['job_id']
    before = wait(client, '/api/jobs/' + job, headers)['stems']
    stem_id = before[0]['id']
    assert client.patch(f'/api/projects/{pid}/stems/{stem_id}',headers=headers,json={'volume':.4,'muted':True}).status_code == 200
    def generate():
        response = client.post(f'/api/projects/{pid}/candidates',headers=headers,json={'stem_id':stem_id,'prompt':'narration'})
        assert response.status_code == 202, response.text
        cid = response.json()['candidate_id']
        result = wait(client, '/api/candidates/' + cid, headers)
        assert result['status'] == 'completed', result
        assert 'path' not in str(result)
        assert client.get(result['audio_url'],headers=headers).status_code == 200
        assert client.get(result['audio_url']).status_code == 401
        return cid
    first = generate()
    assert [s['id'] for s in client.get(f'/api/projects/{pid}/stems',headers=headers).json()] == [s['id'] for s in before]
    client.patch(f'/api/projects/{pid}/stems/{stem_id}',headers=headers,json={'volume':.7})
    assert client.post(f'/api/candidates/{first}/apply',headers=headers).status_code == 409
    second = generate()
    assert client.post(f'/api/candidates/{second}/apply',headers=headers).status_code == 200
    after = client.get(f'/api/projects/{pid}/stems',headers=headers).json()
    assert [s['id'] for s in after] == [s['id'] for s in before]
    assert after[0]['volume'] == .7 and after[0]['muted']
    assert client.post(f'/api/candidates/{second}/apply',headers=headers).status_code == 409

def test_audio_service_error_code_and_retryable_are_exposed(client, tmp_path, monkeypatch):
    import httpx
    import app.media as media
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    body={'detail':'native frame is missing','error':{'code':'INVALID_REQUEST','message':'native frame is missing','retryable':False,'details':{'frame':10}}}
    monkeypatch.setattr(media.httpx,'post',lambda url, **kwargs: httpx.Response(400,request=httpx.Request('POST',url),json=body))
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog']}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h)
    assert job['status']=='failed'
    assert job['error']=='INVALID_REQUEST: native frame is missing'
    assert job['error_code']=='INVALID_REQUEST' and job['retryable'] is False


def test_unstructured_failures_have_no_error_code(client, tmp_path, monkeypatch):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://127.0.0.1:1/separate')
    job=wait(client,'/api/jobs/'+client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog']}).json()['job_id'],h)
    assert job['status']=='failed' and job['error_code'] is None and job['retryable'] is None


def test_prompt_cap_rejects_over_limit_jobs_before_queueing(client, tmp_path, monkeypatch):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.1',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':[f'p{i}' for i in range(9)]}).status_code==422
    monkeypatch.setenv('MAX_PROMPTS','2')
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['a','b','c']}).status_code==422
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['a','b']}).status_code==202


def test_stem_metadata_from_the_audio_service_is_persisted(client, tmp_path, monkeypatch):
    import base64
    import httpx
    import app.media as media
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.2','-ac','2','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    def mock_post(url, files, **kwargs):
        raw=base64.b64encode(files['audio'][1].read()).decode()
        return httpx.Response(200,request=httpx.Request('POST',url),json={'stems':[
            {'name':'siren','audio_base64':raw,'stem_id':'abc123','role':'target','prompt':'siren','label':'siren',
             'start_time':0.0,'duration':0.2,'sample_rate':48000,'channels':2,'sample_count':9600,
             'format':'wav/pcm_f32le','spans':[[0.0,0.2]],'source_asset_id':pid},
            {'name':'residual','audio_base64':raw,'stem_id':'def456','role':'residual','label':'everything else',
             'start_time':0.0,'duration':0.2,'sample_rate':48000,'channels':2,'format':'wav/pcm_f32le'}],
            'model':{'is_mock':True},'warnings':[],'contract_version':'2025-11-draft-1'})
    monkeypatch.setattr(media.httpx,'post',mock_post)
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['siren']}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='completed',job
    target,residual=job['stems']
    assert (target['role'],target['source_stem_id'],target['prompt'],target['label'])==('target','abc123','siren','siren')
    assert (target['sample_rate'],target['channels'],target['format'])==(48000,2,'wav/pcm_f32le')
    assert target['spans']==[[0.0,0.2]]
    assert residual['role']=='residual' and residual['label']=='everything else' and residual['spans'] is None
    assert residual['prompt'] is None
    listed=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert [s['role'] for s in listed]==['target','residual']
    assert client.get(f'/api/projects/{pid}',headers=h).json()['stems'][0]['spans']==[[0.0,0.2]]


def test_fake_adapter_fills_role_prompt_and_label(client, tmp_path):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=0.2',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog','music']}).json()['job_id']
    stems=wait(client,'/api/jobs/'+jid,h)['stems']
    assert [(s['name'],s['role'],s['prompt'],s['label']) for s in stems]==[('dog','target','dog','dog'),('music','target','music','music')]
    assert all(s['role']!='residual' for s in stems)
    assert all(s['spans'] is None and s['source_stem_id'] is None for s in stems)


def test_planner_asks_instead_of_separating_over_existing_stems(monkeypatch):
    import json
    import app.agent as agent
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key')
    plan={'kind':'separate','description':'a person narrating','remove':False,'actions':[],'message':'I will separate the narration.'}
    class MockResponse:
        is_error=False
        def json(self): return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(plan)}]}]}
    monkeypatch.setattr(agent.httpx,'post',lambda url, **kwargs: MockResponse())
    body=agent.AgentRequest(prompt='isolate the narrator',stems=[],tracks=[],duration=10,has_project_stems=True)
    result=agent.plan(body)
    assert result['kind']=='separate' and result['description']=='a person narrating'
    body.has_project_stems=False
    assert agent.plan(body)['kind']=='separate'


def test_existing_database_gains_stem_and_job_metadata_columns(tmp_path, monkeypatch):
    import sqlite3
    import app.core as core
    path=tmp_path/'legacy.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE stems(id TEXT PRIMARY KEY)')
        db.execute("INSERT INTO stems(id) VALUES('old')")
        db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, status TEXT, progress INTEGER)')
        db.execute("INSERT INTO jobs(id,project_id,kind,status,progress) VALUES('j','p','process','queued',0)")
    monkeypatch.setattr(core,'DB',path)
    core.init(); core.init()  # migration is safe to repeat
    with core.connect() as db:
        row=db.execute("SELECT * FROM stems WHERE id='old'").fetchone()
        assert row['role'] is None and row['spans'] is None and row['source_stem_id'] is None
        assert row['prompt'] is None and row['label'] is None and row['format'] is None
        assert row['sample_rate'] is None and row['channels'] is None
        job=db.execute("SELECT * FROM jobs WHERE id='j'").fetchone()
        assert job['error_code'] is None and job['retryable'] is None
