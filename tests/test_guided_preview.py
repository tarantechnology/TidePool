import json
import subprocess
import pytest
from test_app import client, wait


def uploaded(client,tmp_path):
    p=client.post('/api/projects').json(); pid=p['project_id']; headers={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'sound.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=duration=2','-ac','2',str(audio)],check=True)
    with audio.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/upload',headers=headers,files={'file':('sound.wav',f)}).status_code==200
    return pid,headers


def preview(client,pid,h,**changes):
    response=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking',**changes})
    assert response.status_code==202,response.text
    cid=response.json()['candidate_id'];result=wait(client,'/api/candidates/'+cid,h)
    assert result['status']=='completed',result
    return cid,result


def test_guidance_scope_preview_undo_and_multitrack(client,tmp_path):
    pid,h=uploaded(client,tmp_path)
    cid,result=preview(client,pid,h,guidance={'anchors':[{'token':'+','start':.1,'end':.4}]},edit_scope={'action':'mute','interval':{'start':1,'end':1.5}})
    assert result['provenance']['anchors'][0]['end']==.4
    assert result['provenance']['edit_scope']['interval']['start']==1
    for key in ('audio_url','mix_url','original_url'):
        assert client.get(result[key],headers=h).status_code==200
        assert client.get(result[key]).status_code==401
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]
    assert client.post(f'/api/candidates/{cid}/apply').status_code==401
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    target=next(r for r in rows if r['name']!='residual')
    assert target['mute_ranges']==[[1,1.5]] and not target['muted']
    assert next(r for r in rows if r['name']=='residual')['mute_ranges']==[]
    second,_=preview(client,pid,h,prompt='music')
    assert client.post(f'/api/candidates/{second}/apply',headers=h).status_code==200
    assert len(client.get(f'/api/projects/{pid}/stems',headers=h).json())==3
    assert client.post(f'/api/projects/{pid}/candidates',headers=h,json={'stem_id':target['id'],'prompt':'dog'}).status_code==409
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==409
    assert client.post(f'/api/candidates/{second}/undo',headers=h).status_code==200
    assert client.post(f'/api/candidates/{second}/undo',headers=h).status_code==409
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert len(rows)==2 and next(r for r in rows if r['id']==target['id'])['mute_ranges']==[[1,1.5]]


@pytest.mark.parametrize('guidance',[{'anchors':[{'start':1,'end':1}]},{'anchors':[{'start':-1,'end':1}]},{'anchors':[{'start':0,'end':3}]},{'anchors':[{'token':'bad','start':0,'end':1}]}])
def test_invalid_anchors_rejected(client,tmp_path,guidance):
    pid,h=uploaded(client,tmp_path)
    assert client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog','guidance':guidance}).status_code==422
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog'],'guidance':guidance}).status_code==422


def test_undo_stale_and_candidate_failure_recovery(client,tmp_path,monkeypatch):
    import app.candidates as candidates
    pid,h=uploaded(client,tmp_path)
    cid,_=preview(client,pid,h)
    client.post(f'/api/candidates/{cid}/apply',headers=h)
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    client.patch(f"/api/projects/{pid}/stems/{rows[0]['id']}",headers=h,json={'volume':.4,'automation':[{'time':0,'value':.3},{'time':1,'value':1}]})
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==409
    before=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    def fail(*args,**kwargs): raise RuntimeError('Model unavailable. Retry with text and time.')
    monkeypatch.setattr(candidates,'separate',fail)
    cid=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'voice'}).json()['candidate_id']
    assert wait(client,'/api/candidates/'+cid,h)['status']=='failed'
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==before


def test_timed_command_and_ambiguity(client,tmp_path):
    pid,h=uploaded(client,tmp_path)
    body={'prompt':'mute the dog from 1 to 1.5 seconds','stems':[],'tracks':[],'duration':2}
    plan=client.post(f'/api/projects/{pid}/agent',headers=h,json=body).json()
    assert plan['kind']=='separate' and plan['guidance']['anchors']==[]
    cid,_=preview(client,pid,h);client.post(f'/api/candidates/{cid}/apply',headers=h)
    plan=client.post(f'/api/projects/{pid}/agent',headers=h,json=body).json()
    assert plan['kind']=='edit' and plan['actions'][0]['type']=='mute_range'
    cid,_=preview(client,pid,h,prompt='other dog');client.post(f'/api/candidates/{cid}/apply',headers=h)
    plan=client.post(f'/api/projects/{pid}/agent',headers=h,json=body).json()
    assert plan['kind']=='clarify' and 'other dog' in plan['message']


def test_backend_rejects_generated_operations():
    from app.agent import AgentRequest, validate_operations
    body=AgentRequest(prompt='mute',stems=[{'id':'known'}],tracks=[],duration=2)
    for action in [{'type':'set_mute','stemId':'wrong','muted':True},{'type':'set_gain','stemId':'known','value':float('nan')},{'type':'mute_range','stemId':'known','start':1,'end':3}]:
        with pytest.raises(ValueError): validate_operations([action],body)


def test_initial_undo_can_resume_and_retry(client,tmp_path):
    pid,h=uploaded(client,tmp_path)
    cid,_=preview(client,pid,h)
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    info=client.get(f'/api/projects/{pid}',headers=h).json()
    assert info['status']=='uploaded' and info['last_completed_job_id'] is None and info['stems']==[]
    assert client.get(f'/api/projects/{pid}/uploaded-original',headers=h).status_code==200
    assert client.get(f'/api/projects/{pid}/uploaded-original').status_code==401
    preview(client,pid,h)
