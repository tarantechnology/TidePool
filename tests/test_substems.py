"""Separate the rest: one music track replaced by the instruments inside it and a Rest."""
import base64
import json
from pathlib import Path
import httpx
from test_app import client, wait  # noqa: F401  (fixture)
from test_music_stems import project, samples
from test_stem_cleanup import paths, wavs


def split_song(client, pid, h):
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'music_stems'}).json()['job_id']
    return {s['label']:s for s in wait(client,'/api/jobs/'+jid,h)['stems']}


def test_split_replaces_the_other_track_with_children_that_sum_to_it(client, tmp_path):
    import app.core as core
    pid,h=project(client,tmp_path)
    other=split_song(client,pid,h)['other']
    client.patch(f"/api/projects/{pid}/stems/{other['id']}",headers=h,json={'volume':0.5,'solo':True,'automation':[{'time':0,'value':1},{'time':0.5,'value':0.2}]})
    with core.connect() as db: parent_path=db.execute('SELECT path FROM stems WHERE id=?',(other['id'],)).fetchone()['path']
    assert client.post(f"/api/projects/{pid}/stems/{other['id']}/split").status_code==401
    r=client.post(f"/api/projects/{pid}/stems/{other['id']}/split",headers=h,json={})
    assert r.status_code==202,r.text
    job=wait(client,'/api/jobs/'+r.json()['job_id'],h); assert job['status']=='completed',job
    assert job['result']['routing']['mode']=='music_substems' and job['result']['inference']['model']['is_mock'] is True
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert other['id'] not in {s['id'] for s in rows}
    children=[s for s in rows if s['source_stem_id']==other['id']]
    assert [(c['name'],c['role'],c['label']) for c in children]==[('Piano','target','piano'),('Synth','target','synth'),('Rest','residual','rest')]
    assert all(c['volume']==0.5 and c['solo'] and len(c['automation'])==2 and c['parent_name']==other['name'] for c in children)
    assert [s['name'] for s in rows][:3]==['Vocals','Drums','Bass']
    # The stems job lets the editor reload the whole project as usual.
    stems_job=client.get('/api/jobs/'+job['result']['stems_job_id'],headers=h).json()
    assert len(stems_job['stems'])==6
    with core.connect() as db: child_paths=[r['path'] for r in db.execute('SELECT path FROM stems WHERE source_stem_id=?',(other['id'],))]
    assert abs(sum(samples(p) for p in child_paths)-samples(parent_path)).max()<1e-4
    assert Path(parent_path).is_file()  # Undo can still restore the parent.


def test_undo_restores_the_parent_and_drops_the_children(client, tmp_path):
    pid,h=project(client,tmp_path)
    other=split_song(client,pid,h)['other']
    before=paths(pid)
    jid=client.post(f"/api/projects/{pid}/stems/{other['id']}/split",headers=h,json={'stems':['strings']}).json()['job_id']
    assert wait(client,'/api/jobs/'+jid,h)['status']=='completed'
    assert {s['name'] for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}=={'Vocals','Drums','Bass','Strings','Rest'}
    r=client.post(f'/api/candidates/{jid}/undo',headers=h); assert r.status_code==200,r.text
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert {s['name'] for s in rows}=={'Vocals','Drums','Bass','Piano'} and paths(pid)==before
    assert next(s for s in rows if s['id']==other['id'])['revision']==other['revision']+1
    assert wavs(pid)==paths(pid)  # The children's files are gone.
    assert client.post(f'/api/candidates/{jid}/undo',headers=h).status_code==409


def test_split_refusals_and_failures_leave_the_track(client, tmp_path, monkeypatch):
    import app.core as core
    pid,h=project(client,tmp_path)
    stems=split_song(client,pid,h)
    url=lambda sid: f'/api/projects/{pid}/stems/{sid}/split'
    assert client.post(url('nope'),headers=h).status_code==404
    assert client.post(url(stems['other']['id']),headers=h,json={'stems':['a','a']}).status_code==422
    with core.connect() as db: db.execute("UPDATE stems SET label='dog' WHERE id=?",(stems['drums']['id'],))
    assert client.post(url(stems['drums']['id']),headers=h).status_code==409  # not a music track
    jid=client.post(url(stems['other']['id']),headers=h,json={'stems':['kazoo']}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='failed' and job['error_code']=='INVALID_REQUEST'
    assert stems['other']['id'] in {s['id'] for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}
    # The parent edited while queued: refused, nothing replaced.
    import app.worker as worker
    from app.media import separate
    def racing(*args, **kwargs):
        with core.connect() as db: db.execute('UPDATE stems SET revision=revision+1 WHERE id=?',(stems['other']['id'],))
        return separate(*args, **kwargs)
    monkeypatch.setattr(worker,'separate',racing)
    import app.substems as substems
    monkeypatch.setattr(substems,'separate',racing)
    before=paths(pid)
    jid=client.post(url(stems['other']['id']),headers=h).json()['job_id']
    assert wait(client,'/api/jobs/'+jid,h)['status']=='failed' and paths(pid)==before


def test_http_split_sends_music_substems_and_maps_service_errors(client, tmp_path, monkeypatch):
    import app.media as media
    pid,h=project(client,tmp_path)
    other=split_song(client,pid,h)['other']
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    sent=[]
    def unsupported(url, files, data, **kwargs):
        sent.append(json.loads(data['metadata']))
        return httpx.Response(400,request=httpx.Request('POST',url),json={'error':{'code':'UNSUPPORTED_OPERATION','message':'no nested model','retryable':False}})
    monkeypatch.setattr(media.httpx,'post',unsupported)
    jid=client.post(f"/api/projects/{pid}/stems/{other['id']}/split",headers=h,json={'stems':['Piano']}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h)
    assert job['status']=='failed' and job['error_code']=='UNSUPPORTED_OPERATION' and job['retryable'] is False
    assert sent[0]['options']['mode']=='music_substems' and sent[0]['options']['parent_label']=='other' and sent[0]['options']['stems']==['piano']
    def ok(url, files, data, **kwargs):
        raw=base64.b64encode(files['audio'][1].read()).decode()
        return httpx.Response(200,request=httpx.Request('POST',url),json={'stems':[
            {'name':'Piano','label':'piano','role':'target','audio_base64':raw,'tags':[{'label':'Piano','score':0.9}]},
            {'name':'Rest','label':'rest','role':'residual','audio_base64':raw}],'routing':{'mode':'music_substems','engine':'demucs6'}})
    monkeypatch.setattr(media.httpx,'post',ok)
    jid=client.post(f"/api/projects/{pid}/stems/{other['id']}/split",headers=h).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='completed',job
    assert [s['name'] for s in job['result']['stems']]==['Piano','Rest'] and job['result']['stems'][0]['tags'][0]['label']=='Piano'


def test_capabilities_report_music_substems(client, tmp_path, monkeypatch):
    import app.media as media
    pid,h=project(client,tmp_path)
    assert client.get(f'/api/projects/{pid}/capabilities',headers=h).json()['music_substems']['available'] is True
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    body={'model':{'loaded':True},'music_substems':{'available':True,'model_id':'htdemucs_6s','instruments':['piano','guitar'],'secret':1}}
    monkeypatch.setattr(media.httpx,'get',lambda url, **kw: httpx.Response(200,request=httpx.Request('GET',url),json=body))
    assert client.get(f'/api/projects/{pid}/capabilities',headers=h).json()['music_substems']=={'available':True,'model_id':'htdemucs_6s','instruments':['piano','guitar']}
    body.pop('music_substems')
    assert client.get(f'/api/projects/{pid}/capabilities',headers=h).json()['music_substems']=={'available':False}


def test_planner_maps_split_the_rest_to_the_other_track(monkeypatch):
    import pytest
    from app.agent import AgentError, SCHEMA
    from test_music_stems import planned
    assert 'split_stem_id' in SCHEMA['required'] and 'music_substems' in SCHEMA['properties']['separation_mode']['enum']
    stems=[{'id':'v','label':'Vocals','canonical_label':'vocals'},{'id':'o','label':'Piano','canonical_label':'other','tags':['Piano','Strings']},{'id':'x','label':'dog','canonical_label':'dog'}]
    result,request=planned(monkeypatch,{'separation_mode':'music_substems','split_stem_id':'o'},'split the rest',stems=stems)
    assert (result['kind'],result['separation_mode'],result['split_stem_id'],result['description'])==('separate','music_substems','o','piano')
    assert 'music_substems' in request['instructions']
    for bad in ({'separation_mode':'music_substems','split_stem_id':'missing'}, {'separation_mode':'music_substems','split_stem_id':'x'},
                {'separation_mode':'music_substems','split_stem_id':None}, {'separation_mode':'music_stems','split_stem_id':'o'},
                {'separation_mode':'music_substems','split_stem_id':'o','remove':True}):
        with pytest.raises(AgentError): planned(monkeypatch,bad,'split the rest',stems=stems)
