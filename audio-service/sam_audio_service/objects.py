"""Optional SAM object selection. Masks never condition AudioSep inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def configuration():
    return (os.getenv('AUDIO_OBJECT_PYTHON', ''), os.getenv('AUDIO_OBJECT_CHECKPOINT', ''),
            os.getenv('AUDIO_OBJECT_SOURCE_ROOT', ''))


def capabilities():
    python, checkpoint, root = configuration()
    prompted = all(Path(p).is_file() for p in (python, checkpoint)) and (Path(root)/'vision/soundcanvas_cv/sam3_adapter.py').is_file() if all((python, checkpoint, root)) else False
    discovery = os.getenv('AUDIO_OBJECT_DISCOVERY_MODEL', '')
    automatic = bool(python and root and Path(python).is_file() and discovery and Path(discovery).is_file() and (Path(root)/'vision/soundcanvas_cv/yolo_adapter.py').is_file())
    available = bool(prompted or automatic)
    return {'available': available, 'reason': None if available else 'Object selection runtime is not configured.',
            'model': 'sam3.1' if prompted else (Path(discovery).stem if automatic else 'sam3.1'), 'device': 'cuda', 'weights_cached': bool(automatic or (checkpoint and Path(checkpoint).is_file())),
            'selection_only': True, 'max_seconds': 30, 'automatic': automatic, 'prompted':bool(prompted),
            'discovery_model': Path(discovery).name if automatic else None}


def analyze(source, concept, video_id, lock):
    python, checkpoint, root = configuration()
    automatic = concept == 'auto'
    caps=capabilities()
    if automatic:
        if not caps['automatic']:
            raise ValueError('Automatic object discovery runtime is not configured.')
        checkpoint = os.environ['AUDIO_OBJECT_DISCOVERY_MODEL']
    elif not caps['prompted']:
        raise ValueError('SAM object refinement is not configured.')
    content_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    checkpoint_stat = Path(checkpoint).stat()
    implementation = b''.join((Path(root)/'vision/soundcanvas_cv'/name).read_bytes() for name in
                              ('yolo_adapter.py' if automatic else 'sam3_adapter.py', 'pipeline.py'))
    key = hashlib.sha256(implementation + json.dumps([content_hash, concept, checkpoint, checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, 30]).encode()).hexdigest()
    cache = Path(os.getenv('AUDIO_OBJECT_CACHE', '/workspace/cache/soundcanvas-objects'))
    cache.mkdir(parents=True, exist_ok=True)
    result_file = cache/(key+'.json')
    # The same lock used by Engine.separate serializes actual SAM and audio compute.
    # SAM runs in its own prepared interpreter and exits before audio can proceed.
    with lock:
        cached = result_file.is_file()
        if not cached:
            with tempfile.TemporaryDirectory(prefix='sam-', dir=cache) as temporary:
                env = dict(os.environ)
                env['PYTHONPATH'] = str(Path(root)/'vision')
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('YOLO_CONFIG_DIR', str(cache/'ultralytics'))
                command = [python, '-m', 'soundcanvas_cv', '--input', str(source), '--output-dir', temporary,
                           '--adapter', 'yolo' if automatic else 'sam3.1', '--model', checkpoint, '--concept', concept,
                           '--sample-fps', '1000', '--max-seconds', '30', '--device', 'cuda']
                result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=900)
                if result.returncode:
                    # Keep full diagnostic output server-side; never expose paths/tokens.
                    import logging
                    logging.getLogger(__name__).error('SAM object analysis failed: %s', result.stderr[-5000:])
                    raise RuntimeError('SAM object analysis failed; check GPU service logs and retry.')
                data = json.loads((Path(temporary)/'detections.json').read_text())
                # Do not expose implementation paths or artifacts in the public document.
                data.pop('annotated_media', None)
                data['source'] = 'analysis-video.mp4'
                target = cache/(key+'.tmp')
                target.write_text(json.dumps(data))
                target.replace(result_file)
        data = json.loads(result_file.read_text())
    if automatic:
        # YOLO may reuse a model track ID for another class across a cut. Keep
        # person/dog rows distinct; these IDs still aren't speaker identities.
        track_ids = {}
        for frame in data['frames']:
            for obj in frame['objects']:
                if obj.get('track_id') is not None:
                    key = (obj['track_id'], obj['class_id'])
                    obj['track_id'] = track_ids.setdefault(key, len(track_ids)+1)
                    obj['object_id'] = f"track:{obj['track_id']}"
    data.update(video_id=video_id, cache_hit=cached, selection_only=True,
                discovery_mode='automatic' if automatic else 'prompted')
    return data


def register(app, engine):
    from fastapi import File, Form, HTTPException, UploadFile
    from starlette.concurrency import run_in_threadpool

    @app.get('/objects/capabilities')
    def object_capabilities():
        return capabilities()

    @app.post('/objects/analyze')
    async def object_analysis(video: UploadFile = File(...), concept: str = Form('auto'), video_id: str = Form('video')):
        concept = concept.strip()
        if not concept or len(concept) > 80 or len(video_id) > 100:
            raise HTTPException(422, 'Choose a short object description.')
        if not capabilities()['available']:
            raise HTTPException(503, 'SAM object selection is unavailable.')
        with tempfile.TemporaryDirectory(prefix='object-upload-') as temporary:
            source = Path(temporary)/'input.mp4'
            size = 0
            with source.open('wb') as stream:
                while chunk := await video.read(1024*1024):
                    size += len(chunk)
                    if size > 100*1024*1024:
                        raise HTTPException(413, 'Analysis video exceeds 100 MiB.')
                    stream.write(chunk)
            try:
                return await run_in_threadpool(analyze, source, concept, video_id, engine._lock)
            except subprocess.TimeoutExpired:
                raise HTTPException(504, 'SAM analysis timed out. Try a shorter video.')
            except (ValueError, RuntimeError):
                raise HTTPException(503, 'SAM analysis failed; retry or check the GPU service.')
