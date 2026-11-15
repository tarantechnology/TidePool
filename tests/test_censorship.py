import pytest
from test_cinematic import client, uploaded, wait


def test_exact_word_detection_and_window_union():
    from app.censorship import matches, windows
    words=[{'word':'class','start':0.,'end':.1},{'word':' Fuck!','start':.2,'end':.3},{'word':'shit','start':.35,'end':.4},{'word':'shittake','start':.5,'end':.6}]
    found=matches(words)
    assert [w['word'] for w in found]==[' Fuck!','shit']
    assert windows(found,1)==[[pytest.approx(.12),pytest.approx(.48)]]


def test_censorship_preview_apply_undo_and_auth(client,tmp_path,monkeypatch):
    import app.censorship as c
    monkeypatch.setattr(c,'available',lambda: True)
    monkeypatch.setattr(c,'transcribe',lambda path,duration: ([{'word':'shit','start':.3,'end':.5}], 'en'))
    pid,h=uploaded(client,tmp_path)
    url=f'/api/projects/{pid}/censorship-preview'
    assert client.post(url).status_code==401
    response=client.post(url,headers=h); assert response.status_code==202,response.text
    cid=response.json()['candidate_id']; job=wait(client,cid,h)
    assert job['status']=='completed',job
    assert job['provenance']['matches'][0]['word']=='shit'
    assert client.get(job['mix_url']).status_code==401
    assert client.get(job['mix_url'],headers=h).content[:4]==b'RIFF'
    # The generated mix keeps the sine unchanged outside the window and adds a
    # 1 kHz beep while attenuating it inside the window.
    import subprocess
    import numpy as np
    def pcm(content):
        raw=subprocess.run(['ffmpeg','-v','error','-i','pipe:0','-f','f32le','-ac','1','-ar','48000','pipe:1'],input=content,capture_output=True,check=True).stdout
        return np.frombuffer(raw,dtype='<f4')
    original=pcm(client.get(job['original_url'],headers=h).content)
    edited=pcm(client.get(job['mix_url'],headers=h).content)
    assert np.max(np.abs(original[:4800]-edited[:4800])) < .0001
    t=np.arange(12000)/48000
    tone=np.sin(2*np.pi*1000*t)
    assert abs(np.dot(edited[12000:24000],tone))/len(t) > .04
    assert abs(np.dot(edited[12000:24000],np.sin(2*np.pi*440*t)))/len(t) < .01
    applied=client.post(f'/api/candidates/{cid}/apply',headers=h)
    assert applied.status_code==200,applied.text
    stems=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert len(stems)==2
    assert next(s for s in stems if s['name']=='Original audio')['mute_ranges'][0]==pytest.approx([.22,.58])
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_no_matches_does_not_modify_project(client,tmp_path,monkeypatch):
    import app.censorship as c
    monkeypatch.setattr(c,'available',lambda: True)
    monkeypatch.setattr(c,'transcribe',lambda path,duration: ([{'word':'class','start':.3,'end':.5}], 'en'))
    pid,h=uploaded(client,tmp_path)
    cid=client.post(f'/api/projects/{pid}/censorship-preview',headers=h).json()['candidate_id']
    job=wait(client,cid,h)
    assert job['status']=='failed' and 'No listed' in job['error']
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_existing_layer_edits_and_stale_preview(client,tmp_path,monkeypatch):
    import app.censorship as c
    monkeypatch.setattr(c,'available',lambda: True)
    monkeypatch.setattr(c,'transcribe',lambda path,duration: ([{'word':'fuck','start':.3,'end':.5}], 'en'))
    pid,h=uploaded(client,tmp_path)
    job=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['voice','music']}).json()['job_id']
    import time
    for _ in range(200):
        result=client.get('/api/jobs/'+job,headers=h).json()
        if result['status'] in ('completed','failed'): break
        time.sleep(.05)
    assert result['status']=='completed',result
    stems=client.get(f'/api/projects/{pid}/stems',headers=h).json(); target=next(s for s in stems if s['name']=='voice')
    assert client.patch(f'/api/projects/{pid}/stems/{target["id"]}',headers=h,json={'expected_revision':target['revision'],'volume':.5,'automation':[{'time':0,'value':.8}]}).status_code==200
    cid=client.post(f'/api/projects/{pid}/censorship-preview',headers=h).json()['candidate_id']
    assert wait(client,cid,h)['status']=='completed'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    after=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    voice=next(s for s in after if s['id']==target['id']); residual=next(s for s in after if s['name']=='music')
    assert voice['volume']==.5 and voice['automation']==[{'time':0,'value':.8}]
    assert voice['mute_ranges'] and not residual['mute_ranges']
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    cid=client.post(f'/api/projects/{pid}/censorship-preview',headers=h).json()['candidate_id']
    assert wait(client,cid,h)['status']=='completed'
    current=client.get(f'/api/projects/{pid}/stems',headers=h).json()[0]
    assert client.patch(f'/api/projects/{pid}/stems/{current["id"]}',headers=h,json={'expected_revision':current['revision'],'volume':.25}).status_code==200
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==409


def test_missing_transcription_runtime_is_clear(client,tmp_path,monkeypatch):
    monkeypatch.delenv('SOUNDCANVAS_TRANSCRIPTION_MODEL',raising=False)
    pid,h=uploaded(client,tmp_path)
    response=client.post(f'/api/projects/{pid}/censorship-preview',headers=h)
    assert response.status_code==503 and 'not configured' in response.json()['detail']


def test_isolated_runner_protocol_validates_word_times(tmp_path,monkeypatch):
    import json, subprocess
    import app.censorship as c
    monkeypatch.setenv('SOUNDCANVAS_TRANSCRIPTION_MODEL',str(tmp_path))
    monkeypatch.setenv('SOUNDCANVAS_TRANSCRIPTION_PYTHON','/bin/sh')
    def fake_run(args,**kwargs):
        request=json.loads(kwargs['input'])
        assert request['model']==str(tmp_path) and request['path']=='clip.wav'
        return subprocess.CompletedProcess(args,0,stdout=json.dumps({'words':[{'word':'hello','start':.1,'end':.2},{'word':'bad','start':-1,'end':2}],'language':'en'}),stderr='')
    monkeypatch.setattr(c.subprocess,'run',fake_run)
    words,language=c.transcribe('clip.wav',1)
    assert words==[{'word':'hello','start':.1,'end':.2}] and language=='en'
