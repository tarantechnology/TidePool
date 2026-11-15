import json
from pathlib import Path
import pytest
from test_app import client, wait
from test_guided_preview import uploaded


def selection(client,pid,h):
    from app.core import connect
    from app.visual import file_sha256
    with connect() as db: project=db.execute('SELECT * FROM projects WHERE id=?',(pid,)).fetchone()
    data={'selection_only':True,'video_id':pid,'analysis_id':'sam-test','original_source_sha256':file_sha256(project['original']),
          'detector':{'model':'sam3.1'},'frames':[{'frame_index':0,'objects':[{'detection_id':'dog-1','track_id':1,'label':'dog','mask':{'counts':[0,1],'size':[1,1]}}]}]}
    assert client.put(f'/api/projects/{pid}/visual-metadata',headers=h,json=data).status_code==200
    return {'analysis_id':'sam-test','detection_id':'dog-1','track_id':1,'frame_index':0}


def test_object_click_uses_identical_text_transport_and_preserves_preview_apply_undo(client,tmp_path,monkeypatch):
    from app import candidates,media
    pid,h=uploaded(client,tmp_path);chosen=selection(client,pid,h);seen=[];real=media.separate
    def capture(source,directory,payload,report=None):
        seen.append(json.loads(json.dumps(payload)))
        return real(source,directory,payload,report)
    monkeypatch.setattr(candidates,'separate',capture)
    results=[]
    for context in [{},{'object_selection':chosen}]:
        r=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking','edit_scope':{'action':'mute'},**context})
        assert r.status_code==202,r.text
        cid=r.json()['candidate_id'];result=wait(client,'/api/candidates/'+cid,h)
        assert result['status']=='completed',result
        results.append((cid,result,client.get(result['mix_url'],headers=h).content))
    assert results[0][2]==results[1][2]
    for payload in seen:
        assert not payload.get('visual_prompt') and '_object_context' not in payload
        assert payload['prompts']==['dog barking']
        assert 'source' not in payload and '_processing_interval' not in payload
    a,b=seen
    assert {k:v for k,v in a['options'].items() if k!='seed'}=={k:v for k,v in b['options'].items() if k!='seed'}
    context=results[1][1]['provenance']['object_selection']
    assert context['audio_conditioning']=='text_only' and context['label']=='dog'
    cid=results[1][0]
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==[]


@pytest.mark.parametrize('override',[{'analysis_id':'stale'},{'track_id':2},{'detection_id':'not-a-dog'},{'frame_index':5}])
def test_stale_or_unobserved_object_is_rejected(client,tmp_path,override):
    pid,h=uploaded(client,tmp_path);chosen={**selection(client,pid,h),**override}
    r=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking','object_selection':chosen})
    assert r.status_code==422


@pytest.mark.parametrize('analysis_request', [{'concept':'dog'}, {}])
def test_remote_detection_uses_proxy_without_replacing_original_audio_or_layers(client,tmp_path,monkeypatch,analysis_request):
    import subprocess
    from app import object_selection,detector,core
    import httpx
    p=client.post('/api/projects').json();pid=p['project_id'];h={'Authorization':'Bearer '+p['access_token']}
    clip=tmp_path/'clip.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=1','-f','lavfi','-i','sine=duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(clip)],check=True)
    with clip.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('clip.mp4',f)}).status_code==200
    r=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['music']})
    assert wait(client,'/api/jobs/'+r.json()['job_id'],h)['status']=='completed'
    audio_before=client.get(f'/api/projects/{pid}/original-audio',headers=h).content
    stems_before=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    original_before=client.get(f'/api/projects/{pid}/original',headers=h).content
    monkeypatch.setattr(detector,'status',lambda:{'available':True,'selection_only':True,'automatic':True})
    monkeypatch.setenv('VISION_SERVICE_URL','http://vision.test')
    monkeypatch.delenv('OBJECT_ANALYSIS_FPS', raising=False)
    def remote(url,files,data,**kwargs):
        assert data['concept']==analysis_request.get('concept','auto')
        proxy=files['video'][1]
        # Verify the real proxy's cadence; original media/audio must stay native.
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=r_frame_rate,nb_frames','-of','json',proxy.name],text=True))['streams'][0]
        assert info['r_frame_rate']=='10/1' and int(info['nb_frames'])==10
        import hashlib
        digest=hashlib.sha256(proxy.read()).hexdigest()
        return httpx.Response(200,request=httpx.Request('POST',url),json={'selection_only':True,'video_id':data['video_id'],
            'source_sha256':digest,'analysis_id':'sam-test','frames':[],'analyzed_frames':10,'processed_duration_seconds':1})
    monkeypatch.setattr(object_selection.httpx,'post',remote)
    r=client.post(f'/api/projects/{pid}/visual-analyze',headers=h,json=analysis_request)
    assert r.status_code==202 and r.json()['prepared'] is False
    assert wait(client,'/api/visual-jobs/'+r.json()['job_id'],h)['status']=='completed'
    assert client.get(f'/api/projects/{pid}/original-audio',headers=h).content==audio_before
    assert client.get(f'/api/projects/{pid}/original',headers=h).content==original_before
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==stems_before
    monkeypatch.setattr(detector,'status',lambda:{'available':True,'selection_only':True,'automatic':False})
    assert client.post(f'/api/projects/{pid}/visual-analyze',headers=h,json={}).status_code==503
    monkeypatch.setattr(detector,'status',lambda:{'available':True,'selection_only':True,'automatic':True,'prompted':False})
    assert client.post(f'/api/projects/{pid}/visual-analyze',headers=h,json={'concept':'dog'}).status_code==503


@pytest.mark.parametrize('value,expected', [(None,10), ('15',15), ('7.5',7.5)])
def test_selection_cadence_can_restore_detail(monkeypatch,value,expected):
    from app.object_selection import analysis_fps
    monkeypatch.delenv('OBJECT_ANALYSIS_FPS',raising=False)
    if value is not None:monkeypatch.setenv('OBJECT_ANALYSIS_FPS',value)
    assert analysis_fps()==expected


@pytest.mark.parametrize('value', ['0', '-1', '31', 'nan', 'inf', 'fast'])
def test_invalid_selection_cadence_is_rejected(monkeypatch,value):
    from app.object_selection import analysis_fps
    monkeypatch.setenv('OBJECT_ANALYSIS_FPS',value)
    with pytest.raises(ValueError):analysis_fps()


def test_clicking_an_existing_sound_reuses_audio_and_undo_preserves_all_layers(client,tmp_path,monkeypatch):
    from app import candidates
    pid,h=uploaded(client,tmp_path);chosen=selection(client,pid,h)
    r=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking'})
    cid=r.json()['candidate_id'];assert wait(client,'/api/candidates/'+cid,h)['status']=='completed'
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    original=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    audio_before={s['id']:client.get(s['audio_url'],headers=h).content for s in original}
    def fail(*a,**kw):raise AssertionError('Existing sound must not trigger inference')
    monkeypatch.setattr(candidates,'separate',fail)
    r=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking','object_selection':chosen,'edit_scope':{'action':'mute'}})
    cid=r.json()['candidate_id'];result=wait(client,'/api/candidates/'+cid,h)
    assert result['status']=='completed',result
    assert result['provenance']['reused_target_id']
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    after=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    assert len(after)==len(original)
    for row in after:
        assert row['muted']==(row['role']=='target')
        assert client.get(row['audio_url'],headers=h).content==audio_before[row['id']]
    assert client.post(f'/api/candidates/{cid}/undo',headers=h).status_code==200
    assert all(not s['muted'] for s in client.get(f'/api/projects/{pid}/stems',headers=h).json())
