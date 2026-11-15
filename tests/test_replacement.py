import io
import time
import subprocess
import pytest

@pytest.fixture
def client(tmp_path,monkeypatch):
    import app.core as core
    core.ROOT=tmp_path/'storage';core.ROOT.mkdir();core.DB=core.ROOT/'db.sqlite'
    monkeypatch.setenv('STORAGE_ROOT',str(core.ROOT));monkeypatch.setenv('INFERENCE_ADAPTER','fake')
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c: yield c

def wait(c,cid,h):
    for _ in range(300):
        j=c.get('/api/candidates/'+cid,headers=h).json()
        if j['status'] in ('completed','failed'):return j
        time.sleep(.05)
    pytest.fail('timed out')

def test_upload_preview_replace_undo_and_validation(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';fx=tmp_path/'fx.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=5','-ac','2','-ar','48000',str(audio)],check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=880:duration=0.3','-ac','2','-ar','48000',str(fx)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('audio.wav',f)}).status_code==200
    with fx.open('rb') as f: result=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)})
    assert result.status_code==200,result.text
    aid=result.json()['asset_id'];assert client.get(f'/api/projects/{pid}/sound-effects/{aid}/audio').status_code==401
    url=f'/api/projects/{pid}/replacement-preview';body={'asset_id':aid,'effect_label':'quack','source_prompt':'punch','mode':'replace','intervals':[{'start':1,'end':1.5},{'start':3,'end':3.5}]}
    assert client.post(url,json=body).status_code==401
    assert client.post(url,headers=h,json={**body,'asset_id':'x'*32}).status_code==422
    assert client.post(url,headers=h,json={**body,'intervals':[{'start':1,'end':1.5},{'start':1.4,'end':2}]}).status_code==422
    res=client.post(url,headers=h,json=body);assert res.status_code==202,res.text
    cid=res.json()['candidate_id'];job=wait(client,cid,h);assert job['status']=='completed',job
    assert job['provenance']['kind']=='replacement' and job['inference']['model']['is_mock'] is True
    assert client.post('/api/candidates/'+cid+'/apply',headers=h).status_code==200
    stems=client.get(f'/api/projects/{pid}/stems',headers=h).json();assert any(s['name']=='quack' for s in stems)
    targets=[s for s in stems if s['role']=='target']
    assert len(targets)==2 and targets[0]['sound_group_id']==targets[1]['sound_group_id'] and targets[0]['sound_group_id']
    assert [s['sound_start'] for s in targets]==[1,3]
    from app.core import connect,restore_legacy_effect_groups
    with connect() as db:
        db.execute("UPDATE stems SET sound_group_id=NULL,sound_group_label=NULL,sound_start=NULL,sound_end=NULL WHERE project_id=? AND role='target'",(pid,))
        restore_legacy_effect_groups(db)
    targets=[s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json() if s['role']=='target']
    assert len(targets)==2 and targets[0]['sound_group_id']==targets[1]['sound_group_id'] and targets[0]['sound_group_id']
    agent={'prompt':'make the punches quieter','stems':[],'tracks':[],'duration':5,'has_project_stems':True}
    plan=client.post(f'/api/projects/{pid}/agent',headers=h,json=agent).json()
    assert plan['kind']=='edit' and {a['stemId'] for a in plan['actions']}=={s['id'] for s in targets}
    eq=client.post(f'/api/projects/{pid}/effects-preview',headers=h,json={'effect_group_id':targets[0]['sound_group_id'],'bass':2})
    assert eq.status_code==202 and wait(client,eq.json()['candidate_id'],h)['status']=='completed'
    assert client.get(f'/api/projects/{pid}',headers=h).json()['latest_undoable_candidate_id']==cid
    assert client.post('/api/candidates/'+cid+'/undo',headers=h).status_code==200
    assert client.get(f'/api/projects/{pid}',headers=h).json()['latest_undoable_candidate_id'] is None
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_generation_unavailable_is_explicit(client):
    p=client.post('/api/projects').json();h={'Authorization':'Bearer '+p['access_token']}
    response=client.post('/api/projects/'+p['project_id']+'/sound-effects/generate',headers=h,json={'prompt':'quack'})
    assert response.status_code==503


def test_generated_effect_job_and_asset_authorization(client,tmp_path,monkeypatch):
    fake=tmp_path/'stable-audio'
    fake.write_text('#!/bin/sh\n[ -z "$PYTHONPATH" ] || exit 99\nfor last do :; done\nffmpeg -v error -y -f lavfi -i sine=frequency=700:duration=0.5 -c:a pcm_f32le "$last"\n')
    fake.chmod(0o755);monkeypatch.setenv('SOUNDCANVAS_SFX_CLI',str(fake))
    monkeypatch.setenv('PYTHONPATH','/incompatible-editor-dependencies')
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    assert client.post(f'/api/projects/{pid}/sound-effects/generate',json={'prompt':'quack'}).status_code==401
    job=client.post(f'/api/projects/{pid}/sound-effects/generate',headers=h,json={'prompt':'quack'}).json()['job_id']
    for _ in range(300):
        result=client.get('/api/sound-effect-jobs/'+job,headers=h).json()
        if result['status'] in ('completed','failed'): break
        time.sleep(.05)
    assert result['status']=='completed',result
    asset=result['result'];assert asset['source']=='generated'
    assert client.get(f'/api/projects/{pid}/sound-effects/{asset["asset_id"]}/audio',headers=h).content[:4]==b'RIFF'
    other=client.post('/api/projects').json();other_h={'Authorization':'Bearer '+other['access_token']}
    assert client.get('/api/sound-effect-jobs/'+job,headers=other_h).status_code==401


def test_quack_prompts_preserve_count_and_description():
    from app.replacement import effect_request
    single=effect_request('cartoon quack');double=effect_request('two cartoon quacks')
    assert single['count']==1 and double['count']==2
    assert 'cartoon quack' in double['prompt'] and 'single' in double['prompt'].lower()
    assert effect_request('three bangs')['count']==3


def test_parse_replacement_and_vague_requests():
    from app.replacement import parse_replacement
    assert parse_replacement('replace every punch with two quacks',10)=={'source':'punch','effect':'two quacks','time':None}
    assert parse_replacement('I want to replace all of the knife chopping sounds with quack sounds',10)=={'source':'knife chopping sounds','effect':'quack sounds','time':None}
    assert 'quack sounds' not in __import__('app.replacement',fromlist=['effect_request']).effect_request('quack sounds')['prompt']
    assert parse_replacement('can you make every punch sound like a quack?',10)=={'source':'punch','effect':'quack','time':None}
    assert parse_replacement('add a quack at 4 seconds',10)=={'source':'','effect':'quack','time':4,'mode':'add'}
    assert parse_replacement('replace the punch at 4 seconds with a quack',10)=={'source':'punch','effect':'quack','time':4}
    assert parse_replacement('Replace all knife chopping sounds with a quack; use the sound at 4 seconds as an example',10)=={'source':'knife chopping sounds','effect':'quack','time':4,'all_occurrences':True}
    assert parse_replacement('replace every punch with a gong, using that sound around 3.5s as an example.',10)=={'source':'punch','effect':'gong','time':3.5,'all_occurrences':True}
    with pytest.raises(ValueError):parse_replacement('replace the punch at 2 seconds with a quack; use the sound at 4 seconds as an example',10)
    with pytest.raises(ValueError):parse_replacement('replace punch with something',10)
    with pytest.raises(ValueError):parse_replacement('replace the punch at 15s with a quack',10)
    with pytest.raises(ValueError):parse_replacement('replace punch with quack at 4s and add music',10)


def test_automatic_replacement_finds_generates_and_previews(client,tmp_path,monkeypatch):
    import app.discovery as discovery
    fake=tmp_path/'stable-audio';fake.write_text('#!/bin/sh\nfor last do :; done\nffmpeg -v error -y -f lavfi -i sine=frequency=700:duration=0.5 -c:a pcm_f32le "$last"\n');fake.chmod(0o755)
    monkeypatch.setenv('SOUNDCANVAS_SFX_CLI',str(fake))
    monkeypatch.setattr(discovery,'discover',lambda *a,**kw:{'sources':[{'label':'impact','intervals':[{'start':1.,'end':1.4},{'start':3.,'end':3.4}]}]})
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=5','-ac','2','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    response=client.post(f'/api/projects/{pid}/automatic-replacement',headers=h,json={'request':'replace every punch with two quacks'})
    assert response.status_code==202,response.text
    cid=response.json()['candidate_id'];job=wait(client,cid,h);assert job['status']=='completed',job
    assert job['provenance']['requested_count']==2 and len(job['provenance']['intervals'])==2
    assert job['inference']['model']['is_mock'] is True
    assert client.post('/api/candidates/'+cid+'/apply',headers=h).status_code==200
    assert client.post('/api/candidates/'+cid+'/undo',headers=h).status_code==200


def test_followup_replaces_known_effects_at_current_positions_and_undoes(client,tmp_path,monkeypatch):
    import app.candidates as candidates
    import app.discovery as discovery
    fake=tmp_path/'stable-audio';fake.write_text('#!/bin/sh\nfor last do :; done\nffmpeg -v error -y -f lavfi -i sine=frequency=700:duration=0.5 -c:a pcm_f32le "$last"\n');fake.chmod(0o755)
    monkeypatch.setenv('SOUNDCANVAS_SFX_CLI',str(fake))
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';fx=tmp_path/'fx.wav'
    for path,freq in [(audio,440),(fx,880)]:
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'sine=frequency={freq}:duration=5','-ac','2','-ar','48000',str(path)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)}).json()['asset_id']
    body={'asset_id':aid,'effect_label':'quack','source_prompt':'sound','mode':'add','intervals':[{'start':1,'end':1.01},{'start':3,'end':3.01}]}
    cid=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=body).json()['candidate_id'];assert wait(client,cid,h)['status']=='completed'
    assert client.post('/api/candidates/'+cid+'/apply',headers=h).status_code==200
    initial=client.get(f'/api/projects/{pid}/stems',headers=h).json();effects=[s for s in initial if s['role']=='effect'];assert len(effects)==2
    assert effects[0]['effect_group_id'] and effects[0]['effect_group_id']==effects[1]['effect_group_id']
    assert effects[0]['effect_asset_id']==aid and effects[0]['effect_label']=='quack'
    # Projects made before effect metadata existed remain resolvable by layer name.
    from app.core import connect
    with connect() as db:db.execute("UPDATE stems SET effect_group_id=NULL,effect_asset_id=NULL,effect_label=NULL WHERE project_id=? AND role='effect'",(pid,))
    first=effects[0];edit={'expected_revision':first['revision'],'start_offset':first['start_offset']+.2,'volume':.6,'muted':True,'automation':[{'time':0,'value':.8}]}
    assert client.patch(f'/api/projects/{pid}/stems/{first["id"]}',headers=h,json=edit).status_code==200
    before=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    monkeypatch.setattr(candidates,'build',lambda *a,**kw:pytest.fail('Known effects must not run separation.'))
    monkeypatch.setattr(discovery,'discover',lambda *a,**kw:pytest.fail('Known effects must not run detection.'))
    request={'request':'Replace all quacks with loud gongs'}
    assert client.post(f'/api/projects/{pid}/automatic-replacement',json=request).status_code==401
    res=client.post(f'/api/projects/{pid}/automatic-replacement',headers=h,json=request);assert res.status_code==202,res.text
    newcid=res.json()['candidate_id'];job=wait(client,newcid,h);assert job['status']=='completed',job
    assert job['provenance']['kind']=='known-effect-replacement' and len(job['provenance']['peak_times'])==2
    assert 'loud gong.' in job['provenance']['generation_prompt']
    assert client.post('/api/candidates/'+newcid+'/apply',headers=h).status_code==200
    after=client.get(f'/api/projects/{pid}/stems',headers=h).json();changed={s['id']:s for s in after}
    assert all(changed[s['id']]['name'].startswith('loud gongs') for s in effects)
    assert changed[first['id']]['muted'] and changed[first['id']]['automation']==edit['automation']
    assert changed[first['id']]['volume']==pytest.approx(.9)
    assert [s for s in before if s['role']!='effect'][0]['name']==[s for s in after if s['role']!='effect'][0]['name']
    assert client.post('/api/candidates/'+newcid+'/undo',headers=h).status_code==200
    restored={s['id']:s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}
    for row in before:
        for key in ('name','start_offset','volume','muted','automation'):assert restored[row['id']][key]==row[key]
    # A follow-up preview must not overwrite a later edit.
    stale=client.post(f'/api/projects/{pid}/automatic-replacement',headers=h,json=request).json()['candidate_id']
    assert wait(client,stale,h)['status']=='completed'
    row=restored[first['id']]
    assert client.patch(f'/api/projects/{pid}/stems/{row["id"]}',headers=h,json={'expected_revision':row['revision'],'volume':.7}).status_code==200
    assert client.post('/api/candidates/'+stale+'/apply',headers=h).status_code==409


def test_langchain_effect_resolution_validates_only_known_labels(monkeypatch):
    import sys,types
    import app.agent as agent
    from app.effect_followup import resolve_effect_reference,EffectReference
    rows=[{'id':'a','role':'effect','name':'quack','effect_label':'quack'}, {'id':'b','role':'effect','name':'quack 2','effect_label':'quack'}, {'id':'c','role':'target','name':'dog'}]
    monkeypatch.setattr(agent,'_key',lambda:'test-key')
    class FakeModel:
        result=EffectReference(decision='match',label='quack')
        def __init__(self,**kwargs):pass
        def with_structured_output(self,*args,**kwargs):return self
        def invoke(self,messages):return self.result
    monkeypatch.setitem(sys.modules,'langchain_openai',types.SimpleNamespace(ChatOpenAI=FakeModel))
    assert [r['id'] for r in resolve_effect_reference('duck sound',rows)]==['a','b']
    FakeModel.result=EffectReference(decision='match',label='dog')
    with pytest.raises(ValueError,match='verified'):resolve_effect_reference('dog',rows)
    FakeModel.result=EffectReference(decision='ambiguous',label=None)
    with pytest.raises(ValueError,match='Several sounds'):resolve_effect_reference('that sound',rows)
    FakeModel.result=EffectReference(decision='none',label=None)
    assert resolve_effect_reference('footsteps',rows)==[]


def test_group_commands_do_not_mix_distinct_groups(monkeypatch):
    from app.effect_group_commands import plan_group_command
    rows=[{'id':'a','role':'effect','name':'gong','effect_label':'gong','effect_group_id':'one','volume':1,'start_offset':1},
          {'id':'b','role':'effect','name':'gong 2','effect_label':'gong','effect_group_id':'two','volume':.5,'start_offset':2}]
    result=plan_group_command('make the gongs quieter',rows)
    assert result['kind']=='clarify' and result['actions']==[]
    result=plan_group_command('move the second gong later by 0.2 seconds',rows)
    assert result['kind']=='clarify' and result['actions']==[]


def test_separate_identical_sound_prompts_share_a_group(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=4','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    for start in (1,2):
        response=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'knife chopping','processing_interval':{'start':start,'end':start+.3}})
        assert response.status_code==202,response.text
        cid=response.json()['candidate_id'];assert wait(client,cid,h)['status']=='completed';assert client.post('/api/candidates/'+cid+'/apply',headers=h).status_code==200
    targets=[s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json() if s['role']=='target']
    assert len(targets)==2 and targets[0]['sound_group_id']==targets[1]['sound_group_id'] and targets[0]['sound_group_id']
    assert [s['sound_start'] for s in targets]==[1,2]


def test_process_append_groups_same_explicit_sound(client,tmp_path,monkeypatch):
    import shutil
    import app.worker as worker
    def separate(source,directory,payload,report=None):
        result=[]
        for name in (payload['prompts'][0],'residual'):
            path=directory/(worker.uid()+'.wav');shutil.copyfile(source,path)
            result.append({'name':name,'path':path,'role':'residual' if name=='residual' else 'target','prompt':name,'label':name})
        return result
    monkeypatch.setattr(worker,'separate',separate)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=2','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    for append in (False,True):
        response=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog barking'],'append':append})
        assert response.status_code==202,response.text
        jid=response.json()['job_id']
        for _ in range(300):
            job=client.get('/api/jobs/'+jid,headers=h).json()
            if job['status'] in ('completed','failed'):break
            time.sleep(.05)
        assert job['status']=='completed',job
    targets=[s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json() if s['role']=='target']
    assert len(targets)==2 and targets[0]['sound_group_id']==targets[1]['sound_group_id'] and targets[0]['sound_group_id']


def test_group_effects_preview_apply_and_undo(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';fx=tmp_path/'fx.wav'
    for path,duration in [(audio,4),(fx,.2)]:
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'sine=frequency=440:duration={duration}','-ac','2','-ar','48000',str(path)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('fx.wav',f)}).json()['asset_id']
    body={'asset_id':aid,'effect_label':'gong','source_prompt':'sound','mode':'add','intervals':[{'start':1,'end':1.01},{'start':2,'end':2.01}]}
    cid=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=body).json()['candidate_id'];assert wait(client,cid,h)['status']=='completed'
    assert client.post('/api/candidates/'+cid+'/apply',headers=h).status_code==200
    before=client.get(f'/api/projects/{pid}/stems',headers=h).json();clips=[s for s in before if s['role']=='effect'];gid=clips[0]['effect_group_id']
    url=f'/api/projects/{pid}/effects-preview'
    assert client.post(url,json={'effect_group_id':gid,'bass':4}).status_code==401
    assert client.post(url,headers=h,json={'effect_group_id':'not-a-group','bass':4}).status_code==422
    response=client.post(url,headers=h,json={'effect_group_id':gid,'bass':4});assert response.status_code==202,response.text
    eqid=response.json()['candidate_id'];assert wait(client,eqid,h)['status']=='completed'
    assert client.post('/api/candidates/'+eqid+'/apply',headers=h).status_code==200
    after=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert [(s['id'],s['start_offset'],s['effect_group_id']) for s in after]==[(s['id'],s['start_offset'],s['effect_group_id']) for s in before]
    assert client.post('/api/candidates/'+eqid+'/undo',headers=h).status_code==200
    restored=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert [(s['id'],s['start_offset'],s['effect_group_id']) for s in restored]==[(s['id'],s['start_offset'],s['effect_group_id']) for s in before]


def test_group_commands_and_clip_edits_are_revision_checked_and_undoable(client,tmp_path):
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';fx=tmp_path/'fx.wav'
    for path,duration in [(audio,4),(fx,.2)]:subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'sine=frequency=440:duration={duration}','-ac','2','-ar','48000',str(path)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('fx.wav',f)}).json()['asset_id']
    body={'asset_id':aid,'effect_label':'gong','source_prompt':'sound','mode':'add','intervals':[{'start':1,'end':1.01},{'start':2,'end':2.01}]}
    cid=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=body).json()['candidate_id'];assert wait(client,cid,h)['status']=='completed';assert client.post('/api/candidates/'+cid+'/apply',headers=h).status_code==200
    before=client.get(f'/api/projects/{pid}/stems',headers=h).json();clips=[s for s in before if s['role']=='effect']
    # Recover legacy rows only when an applied candidate still references their exact files.
    from app.core import connect,restore_legacy_effect_groups
    with connect() as db:
        db.execute("UPDATE stems SET effect_group_id=NULL,effect_asset_id=NULL,effect_label=NULL WHERE project_id=? AND role='effect'",(pid,))
        restore_legacy_effect_groups(db)
    before=client.get(f'/api/projects/{pid}/stems',headers=h).json();clips=[s for s in before if s['role']=='effect']
    assert clips[0]['effect_group_id'] and clips[0]['effect_group_id']==clips[1]['effect_group_id']
    agent={'prompt':'make the gongs quieter','stems':[],'tracks':[],'duration':4,'has_project_stems':True}
    plan=client.post(f'/api/projects/{pid}/agent',headers=h,json=agent).json();assert plan['kind']=='edit' and {a['stemId'] for a in plan['actions']}=={c['id'] for c in clips}
    agent['prompt']='move the second gong later';assert 'How many seconds' in client.post(f'/api/projects/{pid}/agent',headers=h,json=agent).json()['message']
    agent['prompt']='move the second gong later by 0.2 seconds';plan=client.post(f'/api/projects/{pid}/agent',headers=h,json=agent).json()
    assert plan['kind']=='clip_edit' and plan['clip_edit']['stem_id']==clips[1]['id'] and plan['clip_edit']['start_offset']==pytest.approx(clips[1]['start_offset']+.2)
    url=f'/api/projects/{pid}/effect-clip-edit';body={'stem_id':clips[1]['id'],'expected_revision':clips[1]['revision'],'operation':'move','start_offset':clips[1]['start_offset']+.2}
    assert client.post(url,json=body).status_code==401
    assert client.post(url,headers=h,json={**body,'expected_revision':999}).status_code==409
    result=client.post(url,headers=h,json=body);assert result.status_code==202,result.text
    move=result.json()['candidate_id'];assert wait(client,move,h)['status']=='completed';assert client.post('/api/candidates/'+move+'/apply',headers=h).status_code==200
    moved=client.get(f'/api/projects/{pid}/stems',headers=h).json();assert next(s for s in moved if s['id']==clips[1]['id'])['start_offset']==pytest.approx(body['start_offset'])
    assert client.post('/api/candidates/'+move+'/undo',headers=h).status_code==200
    restored=client.get(f'/api/projects/{pid}/stems',headers=h).json();clip=next(s for s in restored if s['id']==clips[1]['id'])
    deletion=client.post(url,headers=h,json={'stem_id':clip['id'],'expected_revision':clip['revision'],'operation':'delete'}).json()['candidate_id']
    assert wait(client,deletion,h)['status']=='completed';assert client.post('/api/candidates/'+deletion+'/apply',headers=h).status_code==200
    assert clip['id'] not in {s['id'] for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}
    assert client.post('/api/candidates/'+deletion+'/undo',headers=h).status_code==200
    assert clip['id'] in {s['id'] for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}


def test_automatic_no_matches_asks_for_time(client,tmp_path,monkeypatch):
    import app.discovery as discovery
    monkeypatch.setenv('SOUNDCANVAS_SFX_CLI','/unused')
    monkeypatch.setattr(discovery,'discover',lambda *a,**kw:{'sources':[]})
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=5','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    response=client.post(f'/api/projects/{pid}/automatic-replacement',headers=h,json={'request':'replace punches with quack'})
    job=wait(client,response.json()['candidate_id'],h);assert job['status']=='failed' and 'timestamp' in job['error']
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_two_quacks_use_two_separated_copies(client,tmp_path,monkeypatch):
    import numpy as np
    from app.replacement import asset_path, prepare_effect
    from app.media import read_wav
    fake=tmp_path/'stable-audio';fake.write_text('#!/bin/sh\nfor last do :; done\nffmpeg -v error -y -f lavfi -i sine=frequency=700:duration=0.5 -c:a pcm_f32le "$last"\n');fake.chmod(0o755)
    monkeypatch.setenv('SOUNDCANVAS_SFX_CLI',str(fake))
    pid=client.post('/api/projects').json()['project_id']
    single=prepare_effect({'id':pid},{'prompt':'quack'})
    double=prepare_effect({'id':pid},{'prompt':'two quacks'})
    a,sr=read_wav(asset_path(pid,single['asset_id']));b,rate=read_wav(asset_path(pid,double['asset_id']))
    assert rate==sr and double['requested_count']==2 and len(b)>len(a)*2
    gap=b[len(a)+round(sr*.05):len(a)+round(sr*.2)]
    assert np.max(np.abs(gap))<1e-5


def test_automatic_add_uses_supplied_time_without_separation(client,tmp_path,monkeypatch):
    import app.candidates as candidates
    fake=tmp_path/'stable-audio';fake.write_text('#!/bin/sh\nfor last do :; done\nffmpeg -v error -y -f lavfi -i sine=frequency=700:duration=0.5 -c:a pcm_f32le "$last"\n');fake.chmod(0o755)
    monkeypatch.setenv('SOUNDCANVAS_SFX_CLI',str(fake))
    monkeypatch.setattr(candidates,'build',lambda *a,**kw:pytest.fail('Add must not separate the original.'))
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'audio.wav';subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=5','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('a.wav',f)}).status_code==200
    result=client.post(f'/api/projects/{pid}/automatic-replacement',headers=h,json={'request':'add a quack at 4 seconds'})
    assert result.status_code==202,result.text
    job=wait(client,result.json()['candidate_id'],h);assert job['status']=='completed',job
    assert job['provenance']['mode']=='add' and job['provenance']['intervals'][0]['start']==4


def test_generic_agent_never_downgrades_replacement_to_separation(client,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False);monkeypatch.delenv('OPENAI_API_KEY_FILE',raising=False)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    body={'prompt':'I want to replace all the knife chopping sounds with quack sounds','stems':[],'tracks':[],'duration':10}
    res=client.post(f'/api/projects/{pid}/agent',headers=h,json=body)
    assert res.status_code==200 and res.json()['kind']=='clarify'


def test_peak_detection_finds_multiple_transients_and_ignores_carrier_cycles():
    import numpy as np
    from app.replacement import transient_peaks
    rate=48000;samples=np.zeros((rate*2,2))
    for t,amplitude in [(.4,.8),(.75,.3),(1.1,.65)]:
        n=round(.07*rate);i=round(t*rate);wave=amplitude*np.exp(-np.arange(n)/(.015*rate))*np.sin(2*np.pi*440*np.arange(n)/rate)
        samples[i:i+n,0]=wave;samples[i:i+n,1]=wave*.7
    peaks,fallback=transient_peaks(samples,rate,0,2)
    assert not fallback and len(peaks)==3
    assert np.allclose(np.array(peaks)/rate,[.4,.75,1.1],atol=.01)
    constant=np.ones((rate,2))*.1
    peaks,fallback=transient_peaks(constant,rate,0,1)
    assert fallback and len(peaks)==1
    assert transient_peaks(np.zeros((rate,2)),rate,0,1)[0]==[]


def test_replacement_aligns_effect_peak_to_each_detected_peak(client,tmp_path):
    import numpy as np
    from app.media import write_wav,read_wav
    rate=48000;source=np.zeros((rate*2,2),dtype=np.float32)
    for t,amp in [(.03,.7),(.6,.4),(1.1,.8)]:
        n=round(.06*rate);i=round(t*rate);v=amp*np.exp(-np.arange(n)/(.015*rate))*np.sin(2*np.pi*440*np.arange(n)/rate)
        source[i:i+n,0]=v;source[i:i+n,1]=v*.8
    effect=np.zeros((round(.2*rate),2),dtype=np.float32)
    effect[round(.12*rate),:]=.8
    src=tmp_path/'source.wav';fx=tmp_path/'fx.wav';write_wav(src,source,rate);write_wav(fx,effect,rate)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with src.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.wav',f)}).status_code==200
    with fx.open('rb') as f: asset=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)}).json()['asset_id']
    req={'asset_id':asset,'effect_label':'gong','source_prompt':'impact','mode':'replace','intervals':[{'start':0,'end':1.5}]}
    r=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=req);assert r.status_code==202,r.text
    j=wait(client,r.json()['candidate_id'],h);assert j['status']=='completed',j
    points=j['provenance']['peak_times'];assert len(points)==3 and np.allclose(points,[.03,.6,1.1],atol=.01)
    target=tmp_path/'target.wav';target.write_bytes(client.get(j['audio_url'],headers=h).content);placed,sr=read_wav(target)
    assert sr==rate and len(placed)==len(source)
    for point in points:
        index=round(point*rate);chunk=placed[max(0,index-3):index+4]
        offset=int(np.argmax(np.max(np.abs(chunk),axis=1)))-min(index,3)
        assert abs(offset)<=1 and np.max(np.abs(chunk))>.7
    assert any('trimmed' in w for w in j['inference']['warnings'])
    mix=tmp_path/'mix.wav';mix.write_bytes(client.get(j['mix_url'],headers=h).content)
    assert client.post('/api/candidates/'+j['id']+'/apply',headers=h).status_code==200
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    clips=[row for row in rows if row.get('role')=='effect']
    assert len(clips)==3 and len({row['id'] for row in clips})==3
    assert all(row['duration']<.21 for row in clips)
    export=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'wav','include_video':False}).json()['export_id']
    for _ in range(300):
        status=client.get('/api/exports/'+export,headers=h).json()
        if status['status'] in ('completed','failed'):break
        time.sleep(.05)
    assert status['status']=='completed',status
    rendered=tmp_path/'rendered.wav';rendered.write_bytes(client.get(status['download_url'],headers=h).content)
    expected,_=read_wav(mix);actual,_=read_wav(rendered)
    assert expected.shape==actual.shape and np.max(np.abs(expected-actual))<1e-6


def test_silent_selected_target_fails_without_applying_effect(client,tmp_path):
    import numpy as np
    from app.media import write_wav
    rate=48000;source=np.zeros((rate*2,2),dtype=np.float32);source[rate:,0]=.1;source[rate:,1]=.1
    effect=np.ones((rate//4,2),dtype=np.float32)*.1
    src=tmp_path/'source.wav';fx=tmp_path/'effect.wav';write_wav(src,source,rate);write_wav(fx,effect,rate)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with src.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)}).json()['asset_id']
    req={'asset_id':aid,'effect_label':'bell','source_prompt':'impact','mode':'replace','intervals':[{'start':0,'end':.5}]}
    r=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=req);j=wait(client,r.json()['candidate_id'],h)
    assert j['status']=='failed' and 'silent' in j['error'].lower()
    assert j['recovery']['mode']=='add' and j['audio_url'] is None
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]
    retry=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=j['recovery'])
    preview=wait(client,retry.json()['candidate_id'],h)
    assert preview['status']=='completed' and preview['provenance']['mode']=='add'
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


def test_add_aligns_effect_peaks_and_reports_overlap_and_trim(client,tmp_path):
    import numpy as np
    from app.media import write_wav,read_wav
    rate=48000;source=np.zeros((rate*2,2),dtype=np.float32)
    effect=np.zeros((rate//2,2),dtype=np.float32);effect[round(.1*rate),:]=.8
    src=tmp_path/'source.wav';fx=tmp_path/'fx.wav';write_wav(src,source,rate);write_wav(fx,effect,rate)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with src.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)}).json()['asset_id']
    req={'asset_id':aid,'effect_label':'bell','source_prompt':'sound','mode':'add','intervals':[{'start':1.6,'end':1.7},{'start':1.9,'end':1.95}]}
    r=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=req);j=wait(client,r.json()['candidate_id'],h)
    assert j['status']=='completed',j
    assert j['provenance']['peak_times']==[1.6,1.9]
    assert any('overlap' in w for w in j['inference']['warnings'])
    assert any('tail was trimmed' in w for w in j['inference']['warnings'])
    path=tmp_path/'placed.wav';path.write_bytes(client.get(j['audio_url'],headers=h).content);placed,sr=read_wav(path)
    for t in (1.6,1.9): assert np.max(np.abs(placed[round(t*sr)]))>.7


def test_adjust_placement_restores_original_and_never_reruns_separator(client,tmp_path,monkeypatch):
    import numpy as np
    import app.candidates as candidates
    from app.media import write_wav,read_wav
    rate=48000;source=np.zeros((rate*2,2),dtype=np.float32)
    for t in (.4,.9,1.4):
        n=round(.04*rate);i=round(t*rate);v=.7*np.exp(-np.arange(n)/(.01*rate))*np.sin(2*np.pi*600*np.arange(n)/rate)
        source[i:i+n,:]=v[:,None]
    effect=np.zeros((4800,2),np.float32);effect[480,:]=.8
    src=tmp_path/'source.wav';fx=tmp_path/'effect.wav';write_wav(src,source,rate);write_wav(fx,effect,rate)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with src.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)}).json()['asset_id']
    req={'asset_id':aid,'effect_label':'bell','source_prompt':'impact','mode':'replace','intervals':[{'start':.2,'end':1.7}]}
    r=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=req);cid=r.json()['candidate_id'];job=wait(client,cid,h)
    assert job['status']=='completed',job
    events=job['provenance']['events'];assert len(events)==3
    monkeypatch.setattr(candidates,'build',lambda *a,**kw:pytest.fail('Adjustment must not run separation.'))
    url=f'/api/candidates/{cid}/replacement-adjust'
    assert client.post(url,json={'operation':'toggle','event_id':events[1]['id'],'enabled':False}).status_code==401
    assert client.post(url,headers=h,json={'operation':'toggle','event_id':'other','enabled':False}).status_code==422
    r=client.post(url,headers=h,json={'operation':'toggle','event_id':events[1]['id'],'enabled':False});assert r.status_code==202,r.text
    j=wait(client,r.json()['candidate_id'],h);assert j['status']=='completed',j
    assert j['provenance']['events'][1]['enabled'] is False
    before=tmp_path/'before.wav';after=tmp_path/'after.wav'
    before.write_bytes(client.get(j['original_url'],headers=h).content);after.write_bytes(client.get(j['mix_url'],headers=h).content)
    original,_=read_wav(before);edited,_=read_wav(after)
    assert np.max(np.abs(original[round(.88*rate):round(.96*rate)]-edited[round(.88*rate):round(.96*rate)]))<1e-5
    r=client.post(f"/api/candidates/{j['id']}/replacement-adjust",headers=h,json={'operation':'alignment','alignment':'onset'})
    j2=wait(client,r.json()['candidate_id'],h);assert j2['status']=='completed' and j2['provenance']['alignment']=='onset'
    r=client.post(f"/api/candidates/{j2['id']}/replacement-adjust",headers=h,json={'operation':'move','event_id':events[0]['id'],'time':.43})
    j3=wait(client,r.json()['candidate_id'],h);assert j3['status']=='completed'
    r=client.post(f"/api/candidates/{j3['id']}/replacement-adjust",headers=h,json={'operation':'add','time':1.2})
    j4=wait(client,r.json()['candidate_id'],h);assert j4['status']=='completed' and len(j4['provenance']['events'])==4
    bad=client.post(f"/api/candidates/{j4['id']}/replacement-adjust",headers=h,json={'operation':'add','time':1.9})
    assert wait(client,bad.json()['candidate_id'],h)['status']=='failed'
    assert client.post(f"/api/candidates/{j4['id']}/apply",headers=h).status_code==200
    assert client.post(f"/api/candidates/{j4['id']}/undo",headers=h).status_code==200


def test_bad_selection_is_skipped_while_other_selection_is_previewed(client,tmp_path):
    import numpy as np
    from app.media import write_wav
    rate=48000;source=np.zeros((rate*2,2),np.float32);n=2400;t=np.arange(n)/rate
    source[rate:rate+n,:]=(.6*np.exp(-t/.01)*np.sin(2*np.pi*600*t))[:,None]
    effect=np.zeros((4800,2),np.float32);effect[480,:]=.8
    src=tmp_path/'source.wav';fx=tmp_path/'fx.wav';write_wav(src,source,rate);write_wav(fx,effect,rate)
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with src.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('source.wav',f)}).status_code==200
    with fx.open('rb') as f:aid=client.post(f'/api/projects/{pid}/sound-effects/upload',headers=h,files={'file':('effect.wav',f)}).json()['asset_id']
    req={'asset_id':aid,'effect_label':'bell','source_prompt':'impact','mode':'replace','intervals':[{'start':0,'end':.5},{'start':.9,'end':1.3}]}
    r=client.post(f'/api/projects/{pid}/replacement-preview',headers=h,json=req);j=wait(client,r.json()['candidate_id'],h)
    assert j['status']=='completed',j
    assert len(j['provenance']['events'])==1 and any('Skipped' in w for w in j['inference']['warnings'])


def test_example_time_searches_the_rest_of_clip_and_verifies_targets(client,tmp_path,monkeypatch):
    import numpy as np
    from app.media import write_wav
    rate=48000;source=np.zeros((rate*2,2),np.float32)
    for t in (.4,.9,1.4):
        n=round(.07*rate);i=round(t*rate);v=.8*np.exp(-np.arange(n)/(.015*rate))*np.sin(2*np.pi*1200*np.arange(n)/rate)
        source[i:i+n,:]=v[:,None]
    audio=tmp_path/'source.wav';write_wav(audio,source,rate)
    fake=tmp_path/'generator';fake.write_text('#!/bin/sh\nfor last do :; done\nffmpeg -v error -y -f lavfi -i sine=frequency=700:duration=0.5 -c:a pcm_f32le "$last"\n');fake.chmod(0o755)
    monkeypatch.setenv('SOUNDCANVAS_SFX_CLI',str(fake))
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    with audio.open('rb') as f:assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('audio.wav',f)}).status_code==200
    res=client.post(f'/api/projects/{pid}/automatic-replacement',headers=h,json={'request':'replace all chops with bell at 0.4 seconds'})
    assert res.status_code==202,res.text
    j=wait(client,res.json()['candidate_id'],h);assert j['status']=='completed',j
    assert j['provenance']['detection']['method']=='spectral-example-matching'
    times=[x['source_peak'] for x in j['provenance']['events']]
    assert len(times)==3 and np.allclose(times,[.4,.9,1.4],atol=.02)
