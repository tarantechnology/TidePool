import base64
import json
import subprocess
import httpx
import pytest
from test_app import client, wait  # noqa: F401  (fixture)


def project(client, tmp_path, name='song.wav', video=False):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    path=tmp_path/name
    if video: args=['-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','mpeg4','-c:a','aac','-shortest']
    else: args=['-f','lavfi','-i','sine=frequency=440:duration=1','-ac','2','-ar','48000']
    subprocess.run(['ffmpeg','-v','error','-y',*args,str(path)],check=True)
    with path.open('rb') as f: r=client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':(name,f)})
    assert r.status_code==200, r.text
    return pid,h


def samples(path):
    import numpy as np
    return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','f32le','-']),'<f4')


@pytest.mark.parametrize('name',['song.flac','song.m4a','song.mp3'])
def test_audio_only_upload_works_end_to_end(client, tmp_path, name):
    pid,h=project(client,tmp_path,name)
    info=client.get(f'/api/projects/{pid}',headers=h).json()
    assert info['metadata']['has_video'] is False
    assert client.get(f'/api/projects/{pid}/preview',headers=h).status_code==404
    assert client.get(f'/api/projects/{pid}/original-audio',headers=h).status_code==200
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'music_stems'}).json()['job_id']
    assert wait(client,'/api/jobs/'+jid,h)['status']=='completed'
    assert client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'mp4','include_video':True}).status_code==422
    for fmt in ('wav','mp3'):
        eid=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':fmt,'include_video':False}).json()['export_id']
        assert wait(client,'/api/exports/'+eid,h)['status']=='completed'


def test_fake_full_split_returns_named_stems_that_sum_to_the_mix(client, tmp_path):
    import app.core as core
    pid,h=project(client,tmp_path)
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'music_stems'}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='completed',job
    assert [(s['name'],s['role'],s['label']) for s in job['stems']]==[('Vocals','target','vocals'),('Drums','target','drums'),('Bass','target','bass'),('Other','target','other')]
    assert job['inference']['model']['is_mock'] is True and any('no source separation' in w for w in job['inference']['warnings'])
    assert job['routing']['mode']=='music_stems' and job['routing']['stems']==['vocals','drums','bass','other']
    with core.connect() as db: paths=[r['path'] for r in db.execute('SELECT path FROM stems WHERE project_id=?',(pid,))]
    mix=samples(core.ROOT/'projects'/pid/'extracted'/'audio.wav')
    assert abs(sum(samples(p) for p in paths)-mix).max()<1e-4


def test_karaoke_split_mutes_the_vocals_and_keeps_one_residual(client, tmp_path):
    pid,h=project(client,tmp_path)
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'music_stems','stems':['Vocals'],'edit_scope':{'action':'mute'}}).json()['job_id']
    stems=wait(client,'/api/jobs/'+jid,h)['stems']
    assert [(s['name'],s['role'],s['muted']) for s in stems]==[('Vocals','target',True),('residual','residual',False)]


def test_process_mode_validation(client, tmp_path, monkeypatch):
    pid,h=project(client,tmp_path)
    url=f'/api/projects/{pid}/process'
    assert client.post(url,headers=h,json={'mode':'karaoke'}).status_code==422
    assert client.post(url,headers=h,json={'stems':['vocals']}).status_code==422  # stems need music_stems
    assert client.post(url,headers=h,json={'mode':'music_stems','prompts':['vocals']}).status_code==422
    assert client.post(url,headers=h,json={'mode':'music_stems','stems':['vocals','vocals']}).status_code==422
    assert client.post(url,headers=h,json={'mode':'music_stems','append':True}).status_code==422
    # An edit needs exactly one target: a full split has four.
    assert client.post(url,headers=h,json={'mode':'music_stems','edit_scope':{'action':'mute'}}).status_code==422
    # Unknown stem names fail the job with the service's error code.
    jid=client.post(url,headers=h,json={'mode':'music_stems','stems':['kazoo']}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='failed' and job['error_code']=='INVALID_REQUEST' and 'vocals' in job['error']
    monkeypatch.setenv('INFERENCE_ADAPTER','http')
    assert client.post(url,headers=h,json={'mode':'prompted'}).status_code==422
    assert client.post(url,headers=h,json={'prompts':['  ']}).status_code==422


def test_http_forwards_mode_stems_and_source_kind_and_persists_routing(client, tmp_path, monkeypatch):
    import app.media as media
    pid,h=project(client,tmp_path)
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    sent=[]
    routing={'mode':'music_stems','engine':'demucs','model_id':'htdemucs_ft','reason':'no prompts and the content is music','scores':{'music':0.97,'speech':0.02},'stems':['vocals','drums']}
    def mock_post(url, files, data, **kwargs):
        sent.append(json.loads(data['metadata']))
        raw=base64.b64encode(files['audio'][1].read()).decode()
        stems=[{'name':n.title(),'label':n,'role':'target','audio_base64':raw} for n in ('vocals','drums')]
        return httpx.Response(200,request=httpx.Request('POST',url),json={'stems':stems,'routing':routing,'model':{'backend':'demucs','is_mock':False},'warnings':[]})
    monkeypatch.setattr(media.httpx,'post',mock_post)
    # auto with no prompts is the service's call on http.
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='completed',job
    assert job['routing']==routing and [s['name'] for s in job['stems']]==['Vocals','Drums']
    assert 'mode' not in sent[0]['options'] and sent[0]['hints']=={'source_kind':'audio'}
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'music_stems','stems':['vocals','drums']}).json()['job_id']
    assert wait(client,'/api/jobs/'+jid,h)['status']=='completed'
    assert sent[1]['options']['mode']=='music_stems' and sent[1]['options']['stems']==['vocals','drums']
    assert sent[1]['prompts']==[]


def test_route_fake_is_deterministic_by_source_kind(client, tmp_path):
    pid,h=project(client,tmp_path)
    assert client.get(f'/api/projects/{pid}/route').status_code==401
    decision=client.get(f'/api/projects/{pid}/route',headers=h).json()
    assert decision['mode']=='music_stems' and decision['stems']==['vocals','drums','bass','other']
    pid,h=project(client,tmp_path,'clip.mp4',video=True)
    assert client.get(f'/api/projects/{pid}/route',headers=h).json()['mode']=='prompted'


def test_route_http_proxies_the_service_and_whitelists_fields(client, tmp_path, monkeypatch):
    import app.media as media
    pid,h=project(client,tmp_path,'clip.mp4',video=True)
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid:9000/separate')
    seen={}
    def mock_post(url, files, data, **kwargs):
        seen.update(url=url, metadata=json.loads(data['metadata']))
        return httpx.Response(200,request=httpx.Request('POST',url),json={'routing':{'mode':'prompted','engine':'audiosep','model_id':'m','reason':'speech','scores':None,'stems':None,'debug_path':'/secret'}})
    monkeypatch.setattr(media.httpx,'post',mock_post)
    decision=client.get(f'/api/projects/{pid}/route',headers=h).json()
    assert decision=={'mode':'prompted','engine':'audiosep','model_id':'m','reason':'speech','scores':None,'stems':None}
    assert seen['url']=='http://mock.invalid:9000/v1/route' and seen['metadata']['hints']=={'source_kind':'video'}
    monkeypatch.setattr(media.httpx,'post',lambda url, **kw: httpx.Response(404,request=httpx.Request('POST',url)))
    assert client.get(f'/api/projects/{pid}/route',headers=h).status_code==502


def test_capabilities_pass_music_stems_and_routing_through(client, tmp_path, monkeypatch):
    import app.media as media
    pid,h=project(client,tmp_path)
    fake=client.get(f'/api/projects/{pid}/capabilities',headers=h).json()
    assert fake['music_stems']['available'] is True and fake['routing']['default_mode']=='auto'
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    body={'model':{'loaded':True},'music_stems':{'available':True,'model_id':'htdemucs_ft','stems':['vocals','drums','bass','other'],'max_duration_seconds':900,'weights':'/x'},
          'routing':{'default_mode':'auto','classifier':'ast'}}
    monkeypatch.setattr(media.httpx,'get',lambda url, **kw: httpx.Response(200,request=httpx.Request('GET',url),json=body))
    caps=client.get(f'/api/projects/{pid}/capabilities',headers=h).json()
    assert caps['music_stems']=={'available':True,'model_id':'htdemucs_ft','stems':['vocals','drums','bass','other'],'max_duration_seconds':900}
    assert caps['routing']=={'default_mode':'auto','classifier':'ast'}
    body.pop('music_stems'); body.pop('routing')
    assert client.get(f'/api/projects/{pid}/capabilities',headers=h).json()['music_stems']=={'available':False}


def planned(monkeypatch, plan, prompt='split this song into tracks', **extra):
    import app.agent as agent
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key')
    base={'kind':'separate','description':None,'remove':False,'actions':[],'message':'ok','guidance':{'anchors':[]},
          'edit_scope':{'action':'keep','interval':None},'use_visual':False,'separation_mode':None,'stems':None}
    captured={}
    class MockResponse:
        is_error=False
        def json(self): return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({**base,**plan})}]}]}
    def post(url, **kwargs):
        captured.update(kwargs['json']); return MockResponse()
    monkeypatch.setattr(agent.httpx,'post',post)
    result=agent.plan(agent.AgentRequest(prompt=prompt,stems=extra.get('stems',[]),tracks=[],duration=10))
    return result, captured


def test_planner_schema_is_strict_and_carries_separation_mode(monkeypatch):
    from app.agent import SCHEMA
    assert {'separation_mode','stems'} <= set(SCHEMA['required'])
    _,request=planned(monkeypatch,{'separation_mode':'music_stems'})
    assert request['text']['format']['strict'] is True and 'music_stems' in request['instructions']


def test_planner_maps_song_intents(monkeypatch):
    split,_=planned(monkeypatch,{'separation_mode':'music_stems'})
    assert (split['kind'],split['separation_mode'],split['stems'],split['description'])==('separate','music_stems',None,'all instruments')
    karaoke,_=planned(monkeypatch,{'separation_mode':'music_stems','stems':['vocals'],'remove':True,'description':'vocals','edit_scope':{'action':'mute','interval':None}},'make a karaoke version')
    assert karaoke['stems']==['vocals'] and karaoke['remove'] and karaoke['edit_scope']['action']=='mute'
    drums,_=planned(monkeypatch,{'separation_mode':'music_stems','stems':['drums'],'description':'the drums'},'isolate the drums')
    assert drums['stems']==['drums'] and drums['description']=='drums' and not drums['remove']
    speech,_=planned(monkeypatch,{'separation_mode':'prompted','description':'a person speaking'},'isolate the narrator')
    assert speech['separation_mode']=='prompted' and speech['stems'] is None
    edit,_=planned(monkeypatch,{'kind':'edit','actions':[{'type':'set_mute','stemId':'v','muted':True}]},'make a karaoke version',stems=[{'id':'v','label':'Vocals'}])
    assert edit['kind']=='edit' and edit['separation_mode'] is None


@pytest.mark.parametrize('plan',[
    {'separation_mode':'karaoke'},
    {'separation_mode':'prompted','stems':['vocals'],'description':'x'},  # stems need music_stems
    {'separation_mode':'music_stems','stems':['kazoo']},
    {'separation_mode':'music_stems','stems':[]},
    {'separation_mode':'music_stems','remove':True},  # removal needs exactly one stem
    {'kind':'clarify','separation_mode':'music_stems'},
])
def test_planner_rejects_invalid_separation_modes(monkeypatch, plan):
    from app.agent import AgentError
    with pytest.raises(AgentError): planned(monkeypatch,plan)
