"""Detector availability, auto-prepared visual analysis, job stages and error mapping.

The detector subprocess is stubbed; a real run is documented in the PR, not required here."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TRACEBACK_MISSING = '''Traceback (most recent call last):
  File "<frozen runpy>", line 189, in _run_module_as_main
/usr/bin/python: No module named soundcanvas_cv
'''
TRACEBACK_OFFLINE = '''Traceback (most recent call last):
  File "ultralytics/utils/downloads.py", line 440, in safe_download
ConnectionError: Download failure for https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt. Environment may be offline.
'''


@pytest.fixture(scope='module')
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp('vision-media')
    cfr, vfr = root/'cfr.mp4', root/'vfr.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=red:s=64x48:r=30:d=1',
                    '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','mpeg4','-c:a','aac','-shortest',str(cfr)],check=True)
    # A dropped frame with VFR timestamps, as phone recordings often have.
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(cfr),'-vf','select=not(eq(n\\,10))','-fps_mode','vfr',
                    '-c:v','mpeg4','-c:a','copy',str(vfr)],check=True)
    return {'cfr': cfr, 'vfr': vfr}


@pytest.fixture
def api(tmp_path, monkeypatch):
    import app.core as core
    import app.detector as detector
    from app.main import app
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    monkeypatch.setattr(core, 'DB', tmp_path/'db.sqlite')
    monkeypatch.delenv('VISION_ADAPTER', raising=False)
    monkeypatch.delenv('VISION_MODEL', raising=False)
    monkeypatch.delenv('VISION_DEVICE', raising=False)
    monkeypatch.delenv('YOLO_CONFIG_DIR', raising=False)
    monkeypatch.setattr(detector, '_import_error', ...)
    core.init()
    return TestClient(app)  # No lifespan: tests drive the worker with work_once.


def installed(monkeypatch, present):
    import app.detector as detector
    real = detector.importlib.util.find_spec
    def find_spec(name, *args):
        if name in ('soundcanvas_cv', 'ultralytics'):
            return object() if present else None
        return real(name, *args)
    monkeypatch.setattr(detector.importlib.util, 'find_spec', find_spec)


def upload(client, clip):
    project = client.post('/api/projects').json()
    pid, h = project['project_id'], {'Authorization': 'Bearer '+project['access_token']}
    response = client.post(f'/api/projects/{pid}/upload', headers=h, files={'file': (clip.name, clip.read_bytes())})
    assert response.status_code == 200, response.text
    return pid, h


def detections(source, video_id):
    """Minimal valid schema-1.0 document (see test_visual_handoff.py) for the given source bytes."""
    mask = {'encoding':'rle','order':'C','foreground_value':1,'size':[48,64],'counts':[0,3072]}
    frames = [{'frame_index':i,'timestamp_seconds':i/30,'valid_until_seconds':(i+1)/30,
               'objects':[{'detection_id':f'frame:{i}:det:0','track_id':1,'label':'person','mask':mask}]} for i in range(30)]
    return {'schema_version':'1.0','analysis_id':'stub','video_id':video_id,
            'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'width':64,'height':48,'fps':30,'analyzed_frames':30,'processed_frames':30,
            'processed_duration_seconds':1.0,'frames':frames}


class Detector:
    """Stub for app.detector._spawn: records each call and the job's stage at that moment."""
    def __init__(self, returncode=0, stderr=''):
        self.calls, self.returncode, self.stderr = [], returncode, stderr

    def __call__(self, args, timeout, **kwargs):
        from app.core import connect
        with connect() as db:
            job = db.execute("SELECT status,stage,progress FROM jobs WHERE kind='vision' ORDER BY rowid DESC LIMIT 1").fetchone()
        self.calls.append({'args': args, 'kwargs': kwargs, 'job': dict(job) if job else None})
        if self.returncode == 0 and '--output-dir' in args:
            out = Path(args[args.index('--output-dir')+1]); out.mkdir(parents=True)
            source = args[args.index('--input')+1]
            (out/'detections.json').write_text(json.dumps(detections(source, args[args.index('--video-id')+1])))
        return subprocess.CompletedProcess(args, self.returncode, '', self.stderr)


def test_capabilities_report_missing_detector(api, monkeypatch):
    import app.detector as detector
    installed(monkeypatch, False)
    probe = Detector(); monkeypatch.setattr(detector, '_spawn', probe)
    project = api.post('/api/projects').json()
    body = api.get(f"/api/projects/{project['project_id']}/capabilities",
                   headers={'Authorization':'Bearer '+project['access_token']}).json()
    assert body['vision'] == {'available': False, 'reason': detector.NOT_INSTALLED, 'model': 'yolo26n-seg.pt',
                              'weights_cached': False, 'device': 'cpu'}
    assert 'preview' in body  # Existing capability fields are unchanged.
    assert probe.calls == []  # find_spec short-circuits; nothing is spawned or downloaded.


def test_sam_adapter_reuses_detector_boundary(api, monkeypatch, tmp_path):
    import app.detector as detector
    monkeypatch.setenv('VISION_ADAPTER', 'sam3.1')
    monkeypatch.setenv('VISION_MODEL', str(tmp_path/'sam3.1_multiplex.pt'))
    monkeypatch.setenv('VISION_CONCEPT', 'dog')
    probe = Detector()
    monkeypatch.setattr(detector, '_spawn', probe)
    source = tmp_path/'source.mp4'; source.write_bytes(b'boundary-only')
    result = detector.detect(source, tmp_path/'out', 'asset:test')
    args = probe.calls[0]['args']
    assert args[args.index('--adapter')+1] == 'sam3.1'
    assert args[args.index('--concept')+1] == 'dog'
    assert result['video_id'] == 'asset:test'
    monkeypatch.setenv('VISION_ADAPTER', 'sam2.1')
    monkeypatch.delenv('VISION_SEED_OBJECTS', raising=False)
    with pytest.raises(detector.DetectorError, match='user-supplied points'):
        detector.detect(source, tmp_path/'out2', 'asset:test')


def test_capabilities_report_installed_detector_once_and_weights(api, monkeypatch, tmp_path):
    import sys
    import app.detector as detector
    installed(monkeypatch, True)
    probe = Detector(); monkeypatch.setattr(detector, '_spawn', probe)
    monkeypatch.setenv('VISION_DEVICE', 'mps')
    first = detector.status()
    assert first == {'available': True, 'reason': None, 'model': 'yolo26n-seg.pt', 'weights_cached': False, 'device': 'mps'}
    assert len(probe.calls) == 1 and probe.calls[0]['args'][0] == sys.executable
    assert 'import soundcanvas_cv' in probe.calls[0]['args'][2]
    assert probe.calls[0]['kwargs']['env']['YOLO_CONFIG_DIR'] == str(tmp_path/'vision'/'config')
    # Weights appear where ultralytics downloads bare asset names: the detector's cwd.
    (tmp_path/'vision'/'yolo26n-seg.pt').write_bytes(b'w')
    assert detector.status()['weights_cached'] is True
    assert len(probe.calls) == 1  # Availability is cached.
    # ...or in the weights_dir of the ultralytics settings the detector uses.
    (tmp_path/'vision'/'yolo26n-seg.pt').unlink()
    settings = tmp_path/'vision'/'config'/'Ultralytics'/'settings.json'
    settings.parent.mkdir(parents=True); settings.write_text(json.dumps({'weights_dir': str(tmp_path/'w')}))
    assert detector.status()['weights_cached'] is False
    (tmp_path/'w').mkdir(); (tmp_path/'w'/'yolo26n-seg.pt').write_bytes(b'w')
    assert detector.status()['weights_cached'] is True


def test_capabilities_report_broken_detector_import(api, monkeypatch):
    import app.detector as detector
    installed(monkeypatch, True)
    monkeypatch.setattr(detector, '_spawn', Detector(1, 'Traceback...\nImportError: libGL.so.1: cannot open shared object file: No such file or directory\n'))
    vision = detector.status()
    assert vision['available'] is False
    assert vision['reason'] == 'ImportError: libGL.so.1: cannot open shared object file: No such file or directory'


def test_analyze_returns_503_when_detector_unavailable(api, monkeypatch, clips):
    import app.detector as detector
    from app.core import connect
    installed(monkeypatch, False)
    pid, h = upload(api, clips['cfr'])
    response = api.post(f'/api/projects/{pid}/visual-analyze', headers=h)
    assert response.status_code == 503
    assert response.json() == {'detail': detector.NOT_INSTALLED, 'code': 'vision_unavailable'}
    with connect() as db: assert db.execute("SELECT COUNT(*) FROM jobs WHERE kind='vision'").fetchone()[0] == 0


def job(client, jid, h):
    return client.get(f'/api/visual-jobs/{jid}', headers=h).json()


def test_vfr_upload_is_prepared_then_detected(api, monkeypatch, clips):
    import app.detector as detector
    import app.worker as worker
    from app.core import connect
    from app.visual import validate_timeline
    installed(monkeypatch, True)
    stub = Detector(); monkeypatch.setattr(detector, '_spawn', stub)
    pid, h = upload(api, clips['vfr'])
    with connect() as db: uploaded = db.execute('SELECT original FROM projects WHERE id=?', (pid,)).fetchone()['original']
    with pytest.raises(ValueError, match='constant frame rate'):
        validate_timeline(uploaded, 30)
    stages = []
    real_normalize = worker.normalize
    def normalize(pid_, p, finish=None):
        with connect() as db: stages.append(dict(db.execute("SELECT status,stage,progress FROM jobs WHERE kind='vision'").fetchone()))
        return real_normalize(pid_, p, finish)
    monkeypatch.setattr(worker, 'normalize', normalize)

    response = api.post(f'/api/projects/{pid}/visual-analyze', headers=h)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body['status'] == 'queued' and body['prepared'] is True
    queued = job(api, body['job_id'], h)
    assert (queued['status'], queued['stage'], queued['progress']) == ('queued', 'queued', 5)

    assert worker.work_once()
    assert stages == [{'status': 'processing', 'stage': 'preparing', 'progress': 15}]
    assert stub.calls[-1]['job'] == {'status': 'processing', 'stage': 'detecting', 'progress': 30}
    done = job(api, body['job_id'], h)
    assert done['status'] == 'completed', done
    assert (done['stage'], done['progress'], done['error']) == ('completed', 100, None)
    assert done['result'] == {'analyzed_frames': 30, 'processed_duration_seconds': 1.0, 'prepared': True}

    # Detection ran on the prepared CFR copy, which is now the project's media; the upload is kept.
    with connect() as db: p = db.execute('SELECT original,uploaded_original,metadata FROM projects WHERE id=?', (pid,)).fetchone()
    args = stub.calls[-1]['args']
    assert args[args.index('--input')+1] == p['original'] != uploaded
    assert p['uploaded_original'] == uploaded and Path(uploaded).exists()
    validate_timeline(p['original'], json.loads(p['metadata'])['frame_rate'])
    stored = api.get(f'/api/projects/{pid}/visual-metadata', headers=h).json()
    assert stored['source_sha256'] == json.loads(p['metadata'])['source_sha256']


def test_compatible_upload_skips_preparation(api, monkeypatch, clips):
    import app.detector as detector
    import app.worker as worker
    installed(monkeypatch, True)
    stub = Detector(); monkeypatch.setattr(detector, '_spawn', stub)
    monkeypatch.setattr(worker, 'normalize', lambda *a, **k: pytest.fail('compatible media must not be re-encoded'))
    pid, h = upload(api, clips['cfr'])
    body = api.post(f'/api/projects/{pid}/visual-analyze', headers=h).json()
    assert body['prepared'] is False
    assert worker.work_once()
    assert stub.calls[-1]['job']['stage'] == 'detecting'
    done = job(api, body['job_id'], h)
    assert (done['status'], done['stage'], done['progress']) == ('completed', 'completed', 100)
    assert done['result']['prepared'] is False
    # Weights/settings persist under STORAGE_ROOT.
    assert stub.calls[-1]['kwargs']['cwd'].name == 'vision'


def test_unfixable_rotation_and_existing_layers_are_rejected(api, monkeypatch, clips):
    import app.main as main
    from app.core import connect, uid
    from app.visual import UnfixableTimeline
    installed(monkeypatch, True)
    import app.detector as detector
    monkeypatch.setattr(detector, '_spawn', Detector())
    pid, h = upload(api, clips['vfr'])
    with connect() as db:
        db.execute("INSERT INTO stems(id,project_id,name,path,duration) VALUES(?,?,?,?,?)", (uid(), pid, 'voice', '/dev/null', 1))
    response = api.post(f'/api/projects/{pid}/visual-analyze', headers=h)
    assert response.status_code == 409 and 'existing layers were preserved' in response.json()['detail']
    def rotated(path, fps): raise UnfixableTimeline('Video rotation of 45° cannot be normalized automatically; re-export the clip upright before visual prompting')
    monkeypatch.setattr(main, 'validate_timeline', rotated)
    response = api.post(f'/api/projects/{pid}/visual-analyze', headers=h)
    assert response.status_code == 422 and 'rotation of 45°' in response.json()['detail']


def test_unfixable_rotation_is_detected_from_ffprobe(monkeypatch):
    import app.visual as visual
    monkeypatch.setattr(visual, 'run', lambda *a: json.dumps({'streams': [
        {'codec_type': 'video', 'start_time': '0', 'side_data_list': [{'rotation': 45}]}]}))
    with pytest.raises(visual.UnfixableTimeline):
        visual.validate_timeline('x.mp4', 30)
    monkeypatch.setattr(visual, 'run', lambda *a: json.dumps({'streams': [
        {'codec_type': 'video', 'start_time': '0', 'side_data_list': [{'rotation': -90}]}]}))
    with pytest.raises(ValueError) as error:
        visual.validate_timeline('x.mp4', 30)
    assert not isinstance(error.value, visual.UnfixableTimeline)


@pytest.mark.parametrize('stderr,expected', [
    (TRACEBACK_MISSING, 'Object detection is not installed in this backend. Run: pip install -e ./vision (see README).'),
    ("ModuleNotFoundError: No module named 'ultralytics'\n", 'Object detection is not installed in this backend. Run: pip install -e ./vision (see README).'),
    (TRACEBACK_OFFLINE, 'Could not download detector weights (offline?); run `python -m soundcanvas_cv --model yolo26n-seg.pt ...` once with network or set VISION_MODEL to a local path'),
    ('loading...\nCV processing failed: Input does not exist: /x.mp4\n\n', 'CV processing failed: Input does not exist: /x.mp4'),
    ('', 'Object detection exited with code 3'),
])
def test_failed_detection_reports_actionable_last_line(api, monkeypatch, clips, stderr, expected):
    import app.detector as detector
    import app.worker as worker
    installed(monkeypatch, True)
    monkeypatch.setattr(detector, '_import_error', None)
    monkeypatch.setattr(detector, '_spawn', Detector(3, stderr))
    pid, h = upload(api, clips['cfr'])
    jid = api.post(f'/api/projects/{pid}/visual-analyze', headers=h).json()['job_id']
    assert worker.work_once()
    failed = job(api, jid, h)
    assert failed['status'] == 'failed' and failed['error'] == expected
    assert failed['stage'] == 'detecting'  # Where it stopped.
    if expected == detector.NOT_INSTALLED:
        assert detector._import_error is ...  # The next capabilities call re-checks.


def test_detector_timeout_is_reported(api, monkeypatch, clips):
    import app.detector as detector
    import app.worker as worker
    monkeypatch.setattr(detector, '_import_error', None)
    def slow(args, timeout, **kwargs): raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(detector, '_spawn', slow)
    monkeypatch.setenv('VISION_TIMEOUT_SECONDS', '7')
    pid, h = upload(api, clips['cfr'])
    jid = api.post(f'/api/projects/{pid}/visual-analyze', headers=h).json()['job_id']
    assert worker.work_once()
    assert job(api, jid, h)['error'].startswith('Object detection timed out after 7 s')


def test_local_model_path_is_passed_absolute(api, monkeypatch, tmp_path):
    import app.detector as detector
    monkeypatch.chdir(tmp_path)
    (tmp_path/'custom.pt').write_bytes(b'w')
    monkeypatch.setenv('VISION_MODEL', 'custom.pt')
    assert detector.model_argument() == str((tmp_path/'custom.pt').resolve())
    assert detector.weights_cached() is True
    monkeypatch.setenv('VISION_MODEL', 'yolo26n-seg.pt')
    assert detector.model_argument() == 'yolo26n-seg.pt'
