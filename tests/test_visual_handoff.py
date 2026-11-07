"""Project trust/identity regressions. The external audio model is explicitly mocked."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp('media')/'clip.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=red:s=64x48:r=30:d=1',
                    '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(path)],check=True)
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch, video):
    import app.core as core
    from app.main import app
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    monkeypatch.setattr(core, 'DB', tmp_path/'db.sqlite')
    monkeypatch.setenv('INFERENCE_ADAPTER', 'http')
    monkeypatch.setenv('SAM_AUDIO_SERVICE_URL', 'http://mock.invalid/separate')
    core.init()
    client = TestClient(app)  # Work is driven synchronously by work_once.
    project = client.post('/api/projects').json()
    pid = project['project_id']
    headers = {'Authorization':'Bearer '+project['access_token']}
    response = client.post(f'/api/projects/{pid}/upload',headers=headers,files={'file':('clip.mp4', video.read_bytes())})
    assert response.status_code == 200, response.text
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    assert response.json()['metadata']['source_sha256'] == sha
    mask = {'encoding':'rle','order':'C','foreground_value':1,'size':[48,64],'counts':[0,3072]}
    frames = [{'frame_index':i,'timestamp_seconds':i/30,'valid_until_seconds':(i+1)/30,
               'objects':[{'detection_id':f'frame:{i}:det:0','track_id':7,'label':'person','mask':mask}]} for i in range(30)]
    data = {'schema_version':'1.0','analysis_id':'analysis-a','video_id':pid,'source_sha256':sha,
            'width':64,'height':48,'fps':30,'processed_duration_seconds':1.,'frames':frames}
    selection = {'schema_version':'1.0','kind':'visual_selection','analysis_id':'analysis-a',
                 'video_id':pid,'source_sha256':sha,'frame_index':9,'timestamp_seconds':.3,
                 'frame_width':64,'frame_height':48,'interval':{'start_seconds':.2,'end_seconds':.8},
                 **deepcopy(frames[9]['objects'][0])}
    return client, pid, headers, data, selection


def save_and_process(setup, detections=None, selection=None):
    client, pid, h, data, chosen = setup
    response = client.put(f'/api/projects/{pid}/visual-metadata',headers=h,json=data if detections is None else detections)
    assert response.status_code == 200, response.text
    return client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['person speaking'], 'visual_prompt':chosen if selection is None else selection})


def test_trusted_snapshot_and_source_integrity(setup, monkeypatch):
    from app.worker import work_once
    import app.media as media
    client, pid, h, data, selection = setup
    def mock_post(url, files, data, **kwargs):
        payload = json.loads(data['metadata'])
        assert payload['visual_prompt'] == {'selection':selection,'detections':setup[3]}
        assert payload['source']['asset_id'] == pid
        assert payload['asset_id'] == pid
        path = Path(payload['source']['original_ref'].removeprefix('file://'))
        assert path.parent.name == 'original' and path.parent.parent.name == pid
        assert hashlib.sha256(path.read_bytes()).hexdigest() == payload['source']['sha256']
        assert payload['request_id']
        raw = files['audio'][1].read()
        return httpx.Response(200,request=httpx.Request('POST',url),json={'stems':[{'name':'MOCK copy','audio_base64':base64.b64encode(raw).decode()}],
            'model':{'is_mock':True},'warnings':['MOCK: no separation']})
    monkeypatch.setattr(media.httpx, 'post', mock_post)
    start = save_and_process(setup)
    assert start.status_code == 202, start.text
    # Later analysis edits cannot change the job's stored masks.
    assert client.put(f'/api/projects/{pid}/visual-metadata',headers=h,json={}).status_code == 200
    assert work_once()
    job = client.get('/api/jobs/'+start.json()['job_id'],headers=h).json()
    assert job['status'] == 'completed', job
    assert job['inference']['model']['is_mock'] is True
    assert job['stems'][0]['start_offset'] == 0


@pytest.mark.parametrize('field,value', [('source_sha256','0'*64),('video_id','another-project'),
    ('analysis_id','another-analysis'),('detection_id','invented'),('track_id',99),('track_id',None),
    ('timestamp_seconds',.31),('frame_width',99),('frame_index',1),('mask',None)])
def test_tampered_selection_rejected(setup, field, value):
    selection = deepcopy(setup[4]); selection[field] = value
    assert save_and_process(setup, selection=selection).status_code == 422


@pytest.mark.parametrize('damage', ['hash','project','sparse','missing_mask','rle','geometry','empty','duplicate','timestamp','fps'])
def test_invalid_detections_rejected(setup, damage):
    data = deepcopy(setup[3])
    obj = data['frames'][10]['objects'][0]
    if damage == 'hash': data['source_sha256'] = '0'*64
    if damage == 'project': data['video_id'] = 'another-project'
    if damage == 'sparse': del data['frames'][10]
    if damage == 'missing_mask': obj['mask'] = None
    if damage == 'rle': obj['mask']['counts'] = [3073]
    if damage == 'geometry': obj['mask']['size'] = [64,48]
    if damage == 'empty':
        for frame in data['frames']: frame['objects'][0]['mask']['counts'] = [3072]
    if damage == 'duplicate': data['frames'][10]['objects'].append(deepcopy(obj))
    if damage == 'timestamp': data['frames'][10]['timestamp_seconds'] += .02
    if damage == 'fps': data['fps'] = 10
    assert save_and_process(setup, detections=data).status_code == 422


@pytest.mark.parametrize('interval', [(-.1,.8),(.2,.21),(.2,1.1),(.21,.8),(.4,.8)])
def test_invalid_interval_rejected(setup, interval):
    selection = deepcopy(setup[4])
    selection['interval'] = dict(zip(('start_seconds','end_seconds'),interval))
    assert save_and_process(setup, selection=selection).status_code == 422


@pytest.mark.parametrize('key', ['video_ref','original_ref','detections_ref','mask_video_ref','audio_ref'])
def test_client_cannot_read_other_project_media(setup, key):
    client, pid, h, _, selection = setup
    other = client.post('/api/projects').json()['project_id']
    raw = {'selection':selection,'nested':{key:f'file:///data/projects/{other}/original/private.mp4'}}
    assert client.post(f'/api/projects/{pid}/process',headers=h,json={'prompts':['sound'],'visual_prompt':raw}).status_code == 422
    assert client.put(f'/api/projects/{pid}/visual-metadata',headers=h,json=raw).status_code == 422


def test_track_interruption_is_allowed_without_invented_masks(setup):
    data = deepcopy(setup[3]); data['frames'][10]['objects'] = []
    response = save_and_process(setup, detections=data)
    assert response.status_code == 202, response.text


def test_uploaded_bytes_rechecked_before_inference(setup, monkeypatch):
    import app.core as core
    import app.media as media
    from app.worker import work_once
    client, pid, h, _, _ = setup
    start = save_and_process(setup)
    assert start.status_code == 202
    original = next((core.project_dir(pid)/'original').iterdir())
    with original.open('ab') as stream: stream.write(b'changed after enqueue')
    monkeypatch.setattr(media.httpx,'post',lambda *a, **k: pytest.fail('Changed bytes reached audio service'))
    assert work_once()
    job = client.get('/api/jobs/'+start.json()['job_id'],headers=h).json()
    assert job['status'] == 'failed' and 'hash changed' in job['error']


def test_audio_error_details_are_preserved(setup, monkeypatch):
    import app.media as media
    from app.worker import work_once
    client, _, h, _, _ = setup
    monkeypatch.setattr(media.httpx,'post',lambda url, **kwargs: httpx.Response(400,request=httpx.Request('POST',url),
        json={'error':{'code':'INVALID_REQUEST','message':'native frame is missing'}}))
    start = save_and_process(setup)
    assert work_once()
    job = client.get('/api/jobs/'+start.json()['job_id'],headers=h).json()
    assert job['status'] == 'failed' and 'INVALID_REQUEST: native frame is missing' in job['error']


@pytest.mark.parametrize('damage', ['frames', 'frame', 'objects', 'object', 'mask', 'interval', 'envelope'])
def test_malformed_visual_json_is_a_validation_error(setup, damage):
    data, selection = deepcopy(setup[3]), deepcopy(setup[4])
    if damage == 'frames': data['frames'] = None
    if damage == 'frame': data['frames'][10] = 'invalid'
    if damage == 'objects': data['frames'][10]['objects'] = None
    if damage == 'object': data['frames'][10]['objects'] = [None]
    if damage == 'mask': data['frames'][10]['objects'][0]['mask'] = [1]
    if damage == 'interval': selection['interval'] = [0, 1]
    if damage == 'envelope': selection = {'selection': ['invalid']}
    response = save_and_process(setup, detections=data, selection=selection)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize('damage', ['one_sample_short', 'offset', 'sample_rate', 'channels'])
def test_misaligned_visual_stem_fails_the_job(setup, monkeypatch, damage):
    import io
    import wave
    import app.media as media
    from app.worker import work_once
    client, _, headers, _, _ = setup
    def mock_post(url, files, **kwargs):
        original = files['audio'][1].read()
        with wave.open(io.BytesIO(original)) as source:
            params = source.getparams()
            samples = source.readframes(source.getnframes())
        result = io.BytesIO()
        with wave.open(result, 'wb') as output:
            output.setparams(params)
            if damage == 'one_sample_short': samples = samples[:-params.nchannels * params.sampwidth]
            if damage == 'sample_rate': output.setframerate(24000)
            if damage == 'channels': output.setnchannels(1)
            output.writeframes(samples)
        return httpx.Response(200, request=httpx.Request('POST', url), json={'stems': [{
            'name': 'MOCK malformed', 'start_time': .2 if damage == 'offset' else 0,
            'audio_base64': base64.b64encode(result.getvalue()).decode()}]})
    monkeypatch.setattr(media.httpx, 'post', mock_post)
    start = save_and_process(setup)
    assert start.status_code == 202, start.text
    assert work_once()
    job = client.get('/api/jobs/' + start.json()['job_id'], headers=headers).json()
    assert job['status'] == 'failed' and 'Visual stem must start at zero' in job['error']


def test_preexisting_upload_metadata_can_be_resolved(setup):
    import app.core as core
    client, pid, headers, _, _ = setup
    with core.connect() as db:
        row = db.execute('SELECT metadata FROM projects WHERE id=?', (pid,)).fetchone()
        info = json.loads(row['metadata'])
        for key in ('source_sha256', 'video_id', 'width', 'height', 'frame_rate'):
            info.pop(key)
        db.execute('UPDATE projects SET metadata=? WHERE id=?', (json.dumps(info), pid))
    response = save_and_process(setup)
    assert response.status_code == 202, response.text


@pytest.mark.parametrize('damage', ['offset', 'rotation', 'vfr'])
def test_unsupported_video_timelines_are_rejected(tmp_path, video, damage, monkeypatch):
    import app.visual as visual
    from app.visual import validate_timeline
    if damage == 'rotation':
        # ffmpeg versions differ in how rotation is written; pin ffprobe's contract.
        monkeypatch.setattr(visual, 'run', lambda *a: json.dumps({'streams': [
            {'codec_type': 'video', 'start_time': '0', 'side_data_list': [{'rotation': 90}]}]}))
        with pytest.raises(ValueError, match='rotation'):
            validate_timeline(video, 30)
        return
    output = tmp_path / 'changed.mp4'
    args = ['ffmpeg', '-v', 'error', '-y', '-i', str(video)]
    if damage == 'offset': args += ['-c', 'copy', '-output_ts_offset', '0.2']
    if damage == 'vfr': args += ['-vf', 'select=not(eq(n\\,10))', '-fps_mode', 'vfr', '-c:a', 'copy']
    subprocess.run([*args, str(output)], check=True)
    with pytest.raises(ValueError, match='starting at zero|rotation|constant frame rate'):
        validate_timeline(output, 30)
