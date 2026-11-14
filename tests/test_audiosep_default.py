import httpx
import pytest

from test_app import client, wait
from test_guided_preview import uploaded


def test_audiosep_default_options_are_single_candidate_sequential(monkeypatch):
    from app.guidance import options
    monkeypatch.delenv('AUDIO_BACKEND',raising=False)
    monkeypatch.setenv('QUALITY_CANDIDATES','8')
    monkeypatch.setenv('QUALITY_RERANKING','true')
    for profile in ['preview','quality']:
        settings=options(profile)
        assert settings['candidates']==settings['reranking_candidates']==settings['chunk_batch']==1
        assert settings['refine']=='off'


@pytest.mark.parametrize('localization', [{}, {'processing_interval':{'start':.25,'end':1.25}}, {'processing_time':1.0}])
def test_real_audiosep_capabilities_drive_tidepool_preview(client,tmp_path,monkeypatch,localization):
    from app import media, candidates
    pid,headers=uploaded(client,tmp_path)
    monkeypatch.setenv('INFERENCE_ADAPTER','http')
    monkeypatch.setenv('SAM_AUDIO_SERVICE_URL','http://audio.test/separate')
    caps={'backend':'audiosep','is_mock':False,'independent_candidates':False,
          'model':{'loaded':True,'model_id':'audiosep-base-4M','backend':'audiosep'},
          'prompting_modes':{'visual':False,'span_anchors':False},
          'defaults':{'suppression':{'mode':'conservative','strength':.9}},'limits':{}}
    monkeypatch.setattr(media.httpx,'get',lambda url,**kw:httpx.Response(200,request=httpx.Request('GET',url),json=caps))
    advertised=client.get(f'/api/projects/{pid}/capabilities',headers=headers).json()
    assert advertised['backend']=='audiosep' and advertised['model_id']=='audiosep-base-4M'
    assert advertised['suppression']['mode']=='conservative'
    assert advertised['quality'] is False and advertised['visual'] is False
    seen=[]
    def capture(project,payload,cid):
        seen.append(payload)
        raise RuntimeError('Intentional test stop before model inference')
    monkeypatch.setattr(candidates,'build',capture)
    response=client.post(f'/api/projects/{pid}/candidates',headers=headers,json={'prompt':'dog barking','edit_scope':{'action':'mute'},**localization})
    assert response.status_code==202
    assert wait(client,'/api/candidates/'+response.json()['candidate_id'],headers)['status']=='failed'
    assert seen[0]['options']['candidates']==seen[0]['options']['chunk_batch']==1
    assert seen[0]['options']['refine']=='off'
    assert seen[0]['_processing_interval']==localization.get('processing_interval')
    assert seen[0]['_processing_time']==localization.get('processing_time')
    assert client.post(f'/api/projects/{pid}/candidates',headers=headers,json={'prompt':'dog barking','profile':'quality'}).status_code==422
