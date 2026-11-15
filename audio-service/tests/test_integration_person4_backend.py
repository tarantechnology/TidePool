"""Cross-team Test 3 (Backend -> Audio) and the audio half of Test 4 (cinematic demo).

Runs Person 4's real FastAPI backend from ../app with INFERENCE_ADAPTER=http pointing at a
live instance of this service, then walks the documented application flow:
create project -> upload -> process -> poll job -> fetch stem audio -> patch stem -> export.
Person 4's code is imported, never modified.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, wait_for_job

# The live audio service (``audio_service``) and Person 4's backend (``backend_client``,
# ``storage_root``) come from ``tests/conftest.py``; both modules that drive the real
# backend share that one copy.
pytestmark = pytest.mark.skipif(not (REPO / "app" / "main.py").exists(), reason="Person 4 backend not present")


def _demo_video(path: Path) -> Path:
    """Two 'voices' (tones) plus a siren sweep over a colour video, like the cinematic demo."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=2",
        "-f", "lavfi", "-i", "aevalsrc=0.3*sin(2*PI*220*t)+0.3*sin(2*PI*330*t)+0.3*sin(2*PI*(900+300*sin(2*PI*1.5*t))*t):s=48000:c=stereo:d=2",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path


def test_backend_to_audio_and_export(backend_client, tmp_path):
    client, core = backend_client
    p = client.post("/api/projects").json()
    pid, headers = p["project_id"], {"Authorization": "Bearer " + p["access_token"]}
    video = _demo_video(tmp_path / "demo.mp4")
    with video.open("rb") as f:
        r = client.post(f"/api/projects/{pid}/upload", headers=headers, files={"file": ("demo.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    source = r.json()["metadata"]

    # Person 2 style visual metadata forwarded unchanged by Person 4
    assert client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json={"object_id": "person_2", "label": "person", "time_range": {"start": 0.5, "end": 1.5}}).status_code == 200

    r = client.post(f"/api/projects/{pid}/process", headers=headers, json={"prompts": ["siren"]})
    assert r.status_code == 202, r.text
    job = wait_for_job(client, "/api/jobs/" + r.json()["job_id"], headers)
    assert job["status"] == "completed", job
    names = [s["name"] for s in job["stems"]]
    assert names == ["siren", "residual"], names  # target + "everything else", as documented

    # Stems are consumable through Person 4's API with the source's timing
    for stem in job["stems"]:
        assert abs(stem["duration"] - source["duration"]) < 0.05, (stem["duration"], source["duration"])
        audio = client.get(stem["audio_url"], headers=headers)
        assert audio.status_code == 200
        from sam_audio_service.audio_io import decode
        decoded = decode(data=audio.content)
        assert decoded.sample_rate == 48000 and decoded.channels == 2
    siren = next(s for s in job["stems"] if s["name"] == "siren")

    # Editor-style edits (Person 3 operations through Person 4's API) and the cinematic export
    r = client.patch(f"/api/projects/{pid}/stems/{siren['id']}", headers=headers, json={"volume": 0.2, "automation": [{"time": 0, "value": 1}, {"time": 1.5, "value": 0.1}]})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/export", headers=headers, json={"format": "mp4", "include_video": True})
    assert r.status_code == 202, r.text
    result = wait_for_job(client, "/api/exports/" + r.json()["export_id"], headers)
    assert result["status"] == "completed", result
    out = tmp_path / "export.mp4"
    out.write_bytes(client.get(result["download_url"], headers=headers).content)
    info = core.probe(out)
    assert info["has_video"] and abs(info["duration"] - source["duration"]) < 0.1


def test_backend_surfaces_audio_service_error(backend_client, tmp_path):
    """Invalid prompt payload -> this service returns 400 INVALID_REQUEST -> Person 4 marks the job failed."""
    client, _ = backend_client
    p = client.post("/api/projects").json()
    pid, headers = p["project_id"], {"Authorization": "Bearer " + p["access_token"]}
    tone = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-ac", "2", "-ar", "48000", str(tone)], check=True)
    with tone.open("rb") as f:
        assert client.post(f"/api/projects/{pid}/upload", headers=headers, files={"file": ("tone.wav", f, "audio/wav")}).status_code == 200
    r = client.post(f"/api/projects/{pid}/process", headers=headers, json={"prompts": ["a" * 600]})
    job = wait_for_job(client, "/api/jobs/" + r.json()["job_id"], headers)
    assert job["status"] == "failed" and job["error"]


def test_backend_visual_masks_and_export(backend_client, tmp_path, monkeypatch):
    """30 fps visual selection through the real HTTP boundary, with a mock model."""
    import hashlib
    from sam_audio_service.backends.mock import MockBackend
    from sam_audio_service.visual import prepare_masked_video

    client, core = backend_client
    original = tmp_path/'visual.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=red:s=64x48:r=30:d=1',
                    '-f','lavfi','-i','sine=frequency=440:duration=1', '-c:v','mpeg4','-c:a','aac','-shortest',str(original)],check=True)
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    mask = {'encoding':'rle','order':'C','size':[48,64],'counts':[0,3072],'foreground_value':1}
    frames = [{'frame_index':i, 'timestamp_seconds':i/30, 'valid_until_seconds':(i+1)/30,
               'objects':[{'track_id':7,'detection_id':f'frame:{i}:det:0','mask':mask}]} for i in range(30)]
    detections = {'analysis_id':'run1','source_sha256':sha,'fps':30,'width':64,'height':48,
                  'processed_duration_seconds':1.0,'frames':frames}
    selection = {'kind':'visual_selection','analysis_id':'run1','source_sha256':sha,'track_id':7,
                 'detection_id':'frame:9:det:0','label':'person','interval':{'start_seconds':.2,'end_seconds':.8}}
    calls=[]
    base = MockBackend.separate
    def validate_masks(self, mixture, prompt, options):
        assert prompt.visual is not None
        assert mixture.shape == (28800,)
        assert prompt.visual.interval == (.2, .8)
        assert Path(prompt.visual.video_path).parent.parent.name == pid
        assert hashlib.sha256(Path(prompt.visual.video_path).read_bytes()).hexdigest() == sha
        decoded, masks = prepare_masked_video(prompt.visual, size=64)
        assert decoded.shape[0] == masks.shape[0] == 18
        assert (masks == 0).all()
        calls.append(True)
        return base(self, mixture, prompt, options)
    monkeypatch.setattr(MockBackend, 'separate', validate_masks)
    project=client.post('/api/projects').json(); pid=project['project_id']; h={'Authorization':'Bearer '+project['access_token']}
    with original.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/upload', headers=h, files={'file':('visual.mp4',f,'video/mp4')}).status_code==200
    detections.update(schema_version='1.0', video_id=pid)
    selection.update(schema_version='1.0', video_id=pid, frame_width=64, frame_height=48,
                     frame_index=9, timestamp_seconds=.3, mask=mask)
    assert client.put(f'/api/projects/{pid}/visual-metadata',headers=h,json=detections).status_code==200
    request={'prompts':['person'], 'visual_prompt':selection}
    start=client.post(f'/api/projects/{pid}/process',headers=h,json=request)
    assert start.status_code==202, start.text
    result=wait_for_job(client, '/api/jobs/'+start.json()['job_id'],h)
    assert result['status']=='completed', result
    assert calls
    assert result['inference']['model']['is_mock'] is True
    assert result['inference']['warnings']
    assert result['inference']['source']['asset_id'] == pid
    from sam_audio_service.audio_io import decode
    source_audio = decode(data=(core.project_dir(pid)/'extracted/audio.wav').read_bytes())
    stems = [decode(data=client.get(s['audio_url'],headers=h).content).samples for s in result['stems']]
    assert all(s.shape == source_audio.samples.shape for s in stems)
    # The backend uploads FLAC (24-bit) and asks for 16-bit stems: the sum matches to a 16-bit step, not float precision.
    np.testing.assert_allclose(sum(stems), source_audio.samples, atol=2/32768)
    assert not stems[0][:,:9600].any() and not stems[0][:,38400:].any()
    stem=result['stems'][0]
    assert client.patch(f"/api/projects/{pid}/stems/{stem['id']}",headers=h,json={'volume':.5}).status_code==200
    export=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'mp4','include_video':True}).json()['export_id']
    result=wait_for_job(client,'/api/exports/'+export,h)
    assert result['status']=='completed', result
    out=tmp_path/'visual-export.mp4'; out.write_bytes(client.get(result['download_url'],headers=h).content)
    streams=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','csv=p=0',str(out)],text=True)
    assert 'audio' in streams and 'video' in streams
    import json
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(out)]))
    assert abs(float(probe['format']['duration']) - 1.) < 1/30
    assert all(abs(float(s['start_time'])) < 1/48000 for s in probe['streams'])
    assert next(s for s in probe['streams'] if s['codec_type']=='video')['nb_frames'] == '30'
    exported = decode(data=out.read_bytes()).samples
    expected = .5*stems[0]+stems[1]
    n = min(exported.shape[1], expected.shape[1])
    assert abs(exported.shape[1] - expected.shape[1]) <= 1024  # AAC padding
    assert np.sqrt(np.mean((exported[:,:n] - expected[:,:n])**2)) < .01
    for edit, expected in [({'muted':True},stems[1]), ({'muted':False,'solo':True},.5*stems[0])]:
        assert client.patch(f"/api/projects/{pid}/stems/{stem['id']}",headers=h,json=edit).status_code==200
        eid=client.post(f'/api/projects/{pid}/export',headers=h,json={'format':'wav','include_video':False}).json()['export_id']
        done=wait_for_job(client,'/api/exports/'+eid,h)
        assert done['status']=='completed',done
        edited=decode(data=client.get(done['download_url'],headers=h).content).samples
        np.testing.assert_allclose(edited[:,:expected.shape[1]],expected,atol=1/32768)
