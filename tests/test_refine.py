import base64
import json
import subprocess
import numpy as np
import pytest
from test_app import client, wait


def uploaded(client, tmp_path, seconds=1):
    p=client.post('/api/projects').json(); pid=p['project_id']; h={'Authorization':'Bearer '+p['access_token']}
    audio=tmp_path/'tone.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'sine=frequency=440:duration={seconds}','-ac','2','-ar','48000',str(audio)],check=True)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload',headers=h,files={'file':('tone.wav',f)}).status_code==200
    return pid,h


def applied(client, pid, h, **body):
    """Fake-adapter preview + apply: one target and one (silent) residual."""
    r=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog barking',**body}); assert r.status_code==202,r.text
    cid=r.json()['candidate_id']; result=wait(client,'/api/candidates/'+cid,h); assert result['status']=='completed',result
    assert client.post(f'/api/candidates/{cid}/apply',headers=h).status_code==200
    rows=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    return next(r for r in rows if r['role']=='target'), next(r for r in rows if r['role']=='residual'), result


def paths(pid):
    from app.core import connect
    with connect() as db: return {r['id']:r['path'] for r in db.execute('SELECT id,path FROM stems WHERE project_id=?',(pid,))}


def leak_into_residual(pid, residual_id, gain=.25):
    """Simulate SAM Audio attenuating rather than removing: the residual keeps a quieter copy."""
    from app.media import read_wav, write_wav
    path=paths(pid)[residual_id]
    tone=np.sin(2*np.pi*440*np.arange(48000)/48000)[:,None].repeat(2,1)*gain
    write_wav(path,tone,48000)
    return read_wav(path)[0]


def refine(client, pid, h, sid, body=None):
    r=client.post(f'/api/projects/{pid}/stems/{sid}/refine',headers=h,json=body)
    assert r.status_code==202,r.text
    return wait(client,'/api/refine-jobs/'+r.json()['job_id'],h)


def wav_bytes(samples, rate=48000):
    from app.media import encode_wav
    return encode_wav(samples,rate)


def http_stub(monkeypatch, tmp_path, calls, split=.6, refinement=True):
    """Audio-service stand-in: target = split * input, residual = the rest."""
    import httpx
    import app.media as media
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    def post(url, files, data, **kwargs):
        metadata=json.loads(data['metadata']); calls.append(metadata)
        source=tmp_path/f'in{len(calls)}.wav'; source.write_bytes(files['audio'][1].read())
        samples,rate=media.read_wav(source)
        passes=metadata['options'].get('passes',1)
        block={'passes_requested':passes,'passes_run':passes,'per_pass_energy_db':[-6.0-i for i in range(passes)],'stopped_early':False}
        target={'name':metadata['prompts'][0],'role':'target','prompt':metadata['prompts'][0],'audio_base64':base64.b64encode(wav_bytes(samples*split,rate)).decode()}
        if refinement: target['refinement']=block
        return httpx.Response(200,request=httpx.Request('POST',url),json={'stems':[target,
            {'name':'residual','role':'residual','audio_base64':base64.b64encode(wav_bytes(samples*(1-split),rate)).decode()}],
            'model':{'is_mock':True,'passes':passes},'warnings':[]})
    monkeypatch.setattr(media.httpx,'post',post)


def test_wav_round_trip_is_float32_and_exact(tmp_path):
    from app.media import read_wav, write_wav
    from app.core import probe
    samples=np.random.default_rng(1).uniform(-1,1,(1000,2)).astype(np.float32)
    write_wav(tmp_path/'x.wav',samples,48000)
    back,rate=read_wav(tmp_path/'x.wav')
    assert rate==48000 and np.array_equal(back,samples.astype(np.float64))
    info=probe(tmp_path/'x.wav'); assert (info['sample_rate'],info['channels'])==(48000,2)


def test_passes_forwarded_and_refinement_persisted(client, tmp_path, monkeypatch):
    pid,h=uploaded(client,tmp_path,.2)
    calls=[]; http_stub(monkeypatch,tmp_path,calls)
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['siren'],'passes':2}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h); assert job['status']=='completed',job
    assert calls[0]['options']['passes']==2 and calls[0]['options']['residual']=='mixture_minus_targets'
    expected={'passes_requested':2,'passes_run':2,'per_pass_energy_db':[-6.0,-7.0],'stopped_early':False}
    target=next(s for s in job['stems'] if s['role']=='target')
    assert target['refinement']==expected and next(s for s in job['stems'] if s['role']=='residual')['refinement'] is None
    assert next(s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json() if s['role']=='target')['refinement']==expected
    assert next(s for s in client.get(f'/api/projects/{pid}',headers=h).json()['stems'] if s['role']=='target')['refinement']==expected
    # Omitted passes are not forwarded; the service defaults to one.
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['siren']}).json()['job_id']
    assert wait(client,'/api/jobs/'+jid,h)['status']=='completed' and 'passes' not in calls[1]['options']
    # Candidates forward passes too and record it in provenance.
    cid=client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'horn','passes':3}).json()['candidate_id']
    result=wait(client,'/api/candidates/'+cid,h); assert result['status']=='completed',result
    assert calls[2]['options']['passes']==3
    assert result['provenance']['passes']==3 and result['provenance']['refinement']['passes_run']==3


def test_passes_out_of_range_rejected(client, tmp_path, monkeypatch):
    pid,h=uploaded(client,tmp_path,.2)
    for passes in (0,4):
        assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog'],'passes':passes}).status_code==422
        assert client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'dog','passes':passes}).status_code==422
    monkeypatch.setenv('MAX_PASSES','5')
    r=client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['dog'],'passes':4}); assert r.status_code==202
    assert wait(client,'/api/jobs/'+r.json()['job_id'],h)['stems'][0]['refinement']['passes_run']==4


def test_fake_adapter_fills_refinement(client, tmp_path):
    pid,h=uploaded(client,tmp_path,.2)
    target,residual,result=applied(client,pid,h,passes=2)
    assert target['refinement']=={'passes_requested':2,'passes_run':2,'per_pass_energy_db':[0.0,0.0],'stopped_early':False}
    assert residual['refinement'] is None
    assert result['provenance']['passes']==2 and result['inference']['model']['passes']==2


def test_capabilities_report_passes_and_refine(client, tmp_path, monkeypatch):
    import httpx
    import app.media as media
    pid,h=uploaded(client,tmp_path,.2)
    caps=client.get(f'/api/projects/{pid}/capabilities',headers=h).json()
    assert caps['max_passes']==3 and caps['refine'] is True
    monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
    service={'model':{'loaded':True,'text_ranker_loaded':True},'prompting_modes':{},'defaults':{'passes':1},'limits':{'passes':2}}
    monkeypatch.setattr(media.httpx,'get',lambda url, **kwargs: httpx.Response(200,request=httpx.Request('GET',url),json=service))
    caps=client.get(f'/api/projects/{pid}/capabilities',headers=h).json()
    assert caps['max_passes']==2 and caps['default_passes']==1 and caps['refine'] is True


def test_refine_folds_residual_leak_back_into_target(client, tmp_path):
    from app.media import read_wav
    pid,h=uploaded(client,tmp_path)
    target,residual,_=applied(client,pid,h)
    client.patch(f"/api/projects/{pid}/stems/{target['id']}",headers=h,json={'volume':.7,'automation':[{'time':0,'value':1},{'time':.5,'value':.5}],'mute_ranges':[[.1,.2]]})
    target=next(s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json() if s['id']==target['id'])
    before=paths(pid)
    old_target=read_wav(before[target['id']])[0]
    old_residual=leak_into_residual(pid,residual['id'])
    job=refine(client,pid,h,target['id'])
    assert job['status']=='completed',job
    out=job['result']
    assert (out['stem_id'],out['residual_stem_id'])==(target['id'],residual['id'])
    expected_db=10*np.log10(np.mean(old_residual**2))
    assert abs(out['extracted_energy_db']-expected_db)<1e-2
    rows={s['id']:s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json()}
    t,r=rows[target['id']],rows[residual['id']]
    assert t['revision']==target['revision']+1 and r['revision']==residual['revision']+1
    assert (t['volume'],t['automation'],t['mute_ranges'])==(.7,target['automation'],[[.1,.2]])
    assert t['refinement']['passes_run']==2 and t['refinement']['per_pass_energy_db']==[0.0,0.0]  # fake reports zeros
    assert t['refinement']['refine_jobs'][0]['extracted_energy_db']==out['extracted_energy_db']
    after=paths(pid)
    new_target,new_residual=read_wav(after[target['id']])[0],read_wav(after[residual['id']])[0]
    assert np.max(np.abs((new_target+new_residual)-(old_target+old_residual)))<1e-6
    assert np.max(np.abs(new_residual))<1e-6 and np.max(np.abs(new_target-(old_target+old_residual)))<1e-6
    from pathlib import Path
    assert not Path(before[target['id']]).exists() and not Path(before[residual['id']]).exists()
    assert client.get(f"/api/projects/{pid}/stems/{target['id']}/audio",headers=h).status_code==200
    # The completed stem-loader job reflects the refined layers, and no other job blocks the next refine.
    info=client.get(f'/api/projects/{pid}',headers=h).json()
    assert info['last_completed_job_id']==out['stems_job_id']
    second=refine(client,pid,h,target['id'])
    assert second['status']=='completed' and len(rows_after:=client.get(f'/api/projects/{pid}/stems',headers=h).json())==2
    assert next(s for s in rows_after if s['id']==target['id'])['refinement']['passes_run']==3


def test_refine_preconditions(client, tmp_path):
    from app.core import connect
    pid,h=uploaded(client,tmp_path,.5)
    assert client.post(f'/api/projects/{pid}/stems/missing/refine',headers=h).status_code==404
    target,residual,_=applied(client,pid,h)
    assert client.post(f"/api/projects/{pid}/stems/{target['id']}/refine").status_code==401
    r=client.post(f"/api/projects/{pid}/stems/{residual['id']}/refine",headers=h); assert r.status_code==409 and 'target' in r.json()['detail']
    assert client.post(f"/api/projects/{pid}/stems/{target['id']}/refine",headers=h,json={'passes':4}).status_code==422
    client.patch(f"/api/projects/{pid}/stems/{residual['id']}",headers=h,json={'start_offset':.1})
    r=client.post(f"/api/projects/{pid}/stems/{target['id']}/refine",headers=h); assert r.status_code==409 and 'offset' in r.json()['detail']
    client.patch(f"/api/projects/{pid}/stems/{residual['id']}",headers=h,json={'start_offset':0})
    with connect() as db: db.execute("INSERT INTO jobs(id,project_id,kind,status) VALUES('busy',?,'export','queued')",(pid,))
    r=client.post(f"/api/projects/{pid}/stems/{target['id']}/refine",headers=h); assert r.status_code==409 and 'export' in r.json()['detail']
    with connect() as db:
        db.execute("DELETE FROM jobs WHERE id='busy'")
        db.execute("UPDATE stems SET role=NULL,name='residual' WHERE id=?",(target['id'],))  # legacy row named residual
    r=client.post(f"/api/projects/{pid}/stems/{target['id']}/refine",headers=h); assert r.status_code==409
    with connect() as db: db.execute("UPDATE stems SET role=NULL,name='dog' WHERE id=?",(target['id'],))  # legacy target
    with connect() as db: db.execute("UPDATE stems SET role='target',name='other' WHERE id=?",(residual['id'],))
    r=client.post(f"/api/projects/{pid}/stems/{target['id']}/refine",headers=h); assert r.status_code==409 and 'residual' in r.json()['detail']
    with connect() as db: db.execute("UPDATE stems SET role='residual',name='residual' WHERE id=?",(residual['id'],))
    r=client.post(f"/api/projects/{pid}/stems/{target['id']}/refine",headers=h); assert r.status_code==202
    assert wait(client,'/api/refine-jobs/'+r.json()['job_id'],h)['status']=='completed'


def test_refine_blocks_new_separations_while_queued(client, tmp_path):
    from app.core import connect
    pid,h=uploaded(client,tmp_path,.5)
    target,residual,_=applied(client,pid,h)
    with connect() as db: db.execute("INSERT INTO jobs(id,project_id,kind,status,payload) VALUES('r',?,'refine','processing','{}')",(pid,))
    assert client.post(f'/api/projects/{pid}/candidates',headers=h,json={'prompt':'cat'}).status_code==409
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['cat'],'append':True}).status_code==409


def test_refine_with_http_adapter(client, tmp_path, monkeypatch):
    from app.media import read_wav
    pid,h=uploaded(client,tmp_path)
    target,residual,_=applied(client,pid,h,guidance={'anchors':[{'token':'+','start':.1,'end':.4}]})
    before=paths(pid); old_target=read_wav(before[target['id']])[0]
    old_residual=leak_into_residual(pid,residual['id'])
    calls=[]; http_stub(monkeypatch,tmp_path,calls)
    job=refine(client,pid,h,target['id'],{'passes':2})
    assert job['status']=='completed',job
    sent=calls[0]
    assert sent['prompts']==['dog barking'] and sent['options']['passes']==2 and 'seed' not in sent['options']
    assert sent['anchors']==[{'token':'+','start':.1,'end':.4}] and sent['visual_prompt'] is None
    after=paths(pid)
    new_target,new_residual=read_wav(after[target['id']])[0],read_wav(after[residual['id']])[0]
    assert np.max(np.abs(new_target-(old_target+.6*old_residual)))<1e-6
    assert np.max(np.abs((new_target+new_residual)-(old_target+old_residual)))<1e-6
    t=next(s for s in client.get(f'/api/projects/{pid}/stems',headers=h).json() if s['id']==target['id'])
    assert t['refinement']['passes_run']==3 and t['refinement']['per_pass_energy_db']==[0.0,-6.0,-7.0]
    assert job['result']['warnings']==[]


def test_refine_without_origin_warns_and_pads_short_target(client, tmp_path, monkeypatch):
    import httpx
    import app.media as media
    from app.core import connect
    pid,h=uploaded(client,tmp_path)
    target,residual,_=applied(client,pid,h)
    with connect() as db: db.execute("UPDATE jobs SET payload='{}' WHERE kind='candidate'")  # origin lost (e.g. pruned history)
    leak_into_residual(pid,residual['id'])
    calls=[]; http_stub(monkeypatch,tmp_path,calls)
    real=media.httpx.post
    def short(url, **kwargs):
        response=real(url,**kwargs); data=response.json()
        samples,rate=media.read_wav(_decode(data['stems'][0]['audio_base64'],tmp_path))
        data['stems'][0]['audio_base64']=base64.b64encode(wav_bytes(samples[:-480],rate)).decode()
        return httpx.Response(200,request=response.request,json=data)
    monkeypatch.setattr(media.httpx,'post',short)
    job=refine(client,pid,h,target['id'])
    assert job['status']=='completed',job
    warnings=job['result']['warnings']
    assert any('not found' in w for w in warnings) and any('padded' in w for w in warnings)
    from app.media import read_wav
    assert len(read_wav(paths(pid)[target['id']])[0])==48000


def _decode(text, tmp_path):
    path=tmp_path/'decoded.wav'; path.write_bytes(base64.b64decode(text)); return path


@pytest.mark.parametrize('failure',['service','mismatch','commit'])
def test_refine_failure_leaves_stems_untouched(client, tmp_path, monkeypatch, failure):
    import httpx
    import app.media as media
    import app.refine as refine_module
    pid,h=uploaded(client,tmp_path,.5)
    target,residual,_=applied(client,pid,h)
    leak_into_residual(pid,residual['id'])
    stems_dir=next(iter(paths(pid).values())).rsplit('/',1)[0]
    import os
    files_before=sorted(os.listdir(stems_dir)); rows_before=client.get(f'/api/projects/{pid}/stems',headers=h).json()
    before=paths(pid); contents={k:open(v,'rb').read() for k,v in before.items()}
    if failure=='service':
        monkeypatch.setenv('INFERENCE_ADAPTER','http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://mock.invalid/separate')
        monkeypatch.setattr(media.httpx,'post',lambda url, **kw: httpx.Response(503,request=httpx.Request('POST',url),json={'error':{'code':'model_unavailable','message':'down','retryable':True}}))
    elif failure=='mismatch':
        calls=[]; http_stub(monkeypatch,tmp_path,calls)
        real=media.httpx.post
        def mono(url, **kwargs):
            response=real(url,**kwargs); data=response.json()
            data['stems'][0]['audio_base64']=base64.b64encode(wav_bytes(np.zeros((48000,1)),48000)).decode()
            return httpx.Response(200,request=response.request,json=data)
        monkeypatch.setattr(media.httpx,'post',mono)
    else:
        def boom(db, pid): raise RuntimeError('disk full')
        monkeypatch.setattr(refine_module,'completed_job',boom)
    job=refine(client,pid,h,target['id'])
    assert job['status']=='failed' and job['error']
    if failure=='service': assert job['error_code']=='model_unavailable' and job['retryable'] is True
    assert client.get(f'/api/projects/{pid}/stems',headers=h).json()==rows_before
    assert paths(pid)==before and {k:open(v,'rb').read() for k,v in before.items()}==contents
    assert sorted(os.listdir(stems_dir))==files_before


def test_unusable_visual_guidance_is_skipped_with_warning():
    from app.refine import request
    project={'id':'0'*32,'visual':None,'metadata':'{}','original':None}
    target={'prompt':None,'name':'dog'}
    original={'prompts':[{'text':'dog barking','name':'dog','anchors':[{'token':'-','start':0,'end':.1}]}],
              'anchors':[{'token':'+','start':.2,'end':.3}],'visual_prompt':{'selection':{'track_id':1}},
              'options':{'reranking_candidates':2,'seed':7}}
    warnings=[]
    body=request(project,target,original,2,'job',warnings)
    assert body['prompts']==['dog'] and body['visual_prompt'] is None and 'source' not in body
    assert body['options']=={'reranking_candidates':2,'passes':2}
    assert [a['token'] for a in body['anchors']]==['+','-']
    assert len(warnings)==1 and 'Visual guidance' in warnings[0]
