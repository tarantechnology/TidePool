"""Real backend routes/SQLite/FFmpeg/worker; external separation is explicitly mocked."""
import json
import shutil
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.selection import select_at_point
from test_contract import Detector
from soundcanvas_cv.masks import encode_mask
import numpy as np


class Segmented(Detector):
    def detect(self, frame):
        objects = super().detect(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[4:24, 4:20] = 1
        objects[0].mask = encode_mask(mask)
        return objects


def run_handoff(tmp_path, monkeypatch, video, metadata, explicit_override):
    import app.core as core
    from app.main import app
    from app.worker import work_once
    import app.media as media
    core_root = tmp_path/'storage'; core_root.mkdir()
    monkeypatch.setattr(core, 'ROOT', core_root)
    monkeypatch.setattr(core, 'DB', core_root/'db.sqlite')
    core.init()
    monkeypatch.setenv('INFERENCE_ADAPTER', 'http')
    monkeypatch.setenv('SAM_AUDIO_SERVICE_URL', 'http://mock-audio.invalid/separate')
    obj = metadata['frames'][0]['objects'][0]
    x1,y1,x2,y2 = obj['bbox_normalized']
    selection = select_at_point(metadata, 0, (x1+x2)/2, (y1+y2)/2)
    stored = {'detections':metadata, 'selection':selection}
    expected = stored
    captured = []
    def mock_audio_post(url, files, data, timeout):
        import base64
        import httpx
        assert url == 'http://mock-audio.invalid/separate'
        payload = json.loads(data['metadata']); captured.append(payload)
        assert payload['visual_prompt'] == expected
        assert payload['source']['sha256'] == metadata['source_sha256']
        assert payload['source']['asset_id'] == pid
        assert payload['source']['original_ref'].startswith((core_root/'projects'/pid/'original').as_uri())
        # The adapter uploads a lossless FLAC encode; decode it back to WAV like the service would.
        assert files['audio'][0] == 'audio.flac' and files['audio'][2] == 'audio/flac'
        flac = tmp_path/'upload.flac'; flac.write_bytes(files['audio'][1].read())
        wav = tmp_path/'upload.wav'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(flac),'-c:a','pcm_s16le',str(wav)],check=True)
        raw = wav.read_bytes()
        return httpx.Response(200, request=httpx.Request('POST', url), json={'stems':[{'name':'MOCK copy; not separated','audio_base64':base64.b64encode(raw).decode()}]})
    monkeypatch.setattr(media.httpx, 'post', mock_audio_post)
    # Deliberately run the worker synchronously: same work_once as the real background thread.
    client = TestClient(app)
    project = client.post('/api/projects').json(); pid = project['project_id']
    headers = {'Authorization':'Bearer '+project['access_token']}
    with video.open('rb') as stream:
        assert client.post(f'/api/projects/{pid}/upload', headers=headers,
                           files={'file':('input.mp4',stream,'video/mp4')}).status_code == 200
    assert client.put(f'/api/projects/{pid}/visual-metadata', json=stored).status_code == 401
    assert client.put(f'/api/projects/{pid}/visual-metadata', headers=headers, json=stored).status_code == 200
    assert client.get(f'/api/projects/{pid}',headers=headers).json()['visual_metadata'] == stored
    body = {'prompts':['person'], **({'visual_prompt':selection} if explicit_override else {})}
    response = client.post(f'/api/projects/{pid}/process', headers=headers, json=body)
    assert response.status_code == 202, response.text
    assert work_once()
    job = client.get('/api/jobs/'+response.json()['job_id'],headers=headers).json()
    assert job['status'] == 'completed', job
    assert len(captured) == 1 and len(job['stems']) == 1
    assert job['stems'][0]['name'].startswith('MOCK')


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='Backend integration requires FFmpeg')
@pytest.mark.parametrize('explicit_override', [False, True])
def test_selection_through_backend_to_mock_audio_service(tmp_path, monkeypatch, explicit_override):
    video = tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x48:r=10:d=1',
                    '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-pix_fmt','yuv420p',
                    '-c:a','aac','-shortest',str(video)], check=True)
    metadata = process_media(video, tmp_path/'cv', Segmented(), 10, 1)
    run_handoff(tmp_path, monkeypatch, video, metadata, explicit_override)


@pytest.mark.real
@pytest.mark.skipif(__import__('os').getenv('SC_CV_REAL_TESTS') != '1', reason='Set SC_CV_REAL_TESTS=1 after real validation')
def test_real_yolo_metadata_to_backend(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'outputs/real_seg/detections.json').read_text())
    run_handoff(tmp_path, monkeypatch, root/'examples/media/pedestrians.mp4', data, True)
