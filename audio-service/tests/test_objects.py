import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from sam_audio_service import objects


def test_sam_cache_and_inference_lock_prevent_duplicate_gpu_work(tmp_path,monkeypatch):
    python=tmp_path/'python';python.touch();checkpoint=tmp_path/'sam.pt';checkpoint.touch()
    root=tmp_path/'project';code=root/'vision/soundcanvas_cv';code.mkdir(parents=True)
    for name in ['sam3_adapter.py','pipeline.py']:(code/name).write_text('tested')
    for name,value in {'AUDIO_OBJECT_PYTHON':python,'AUDIO_OBJECT_CHECKPOINT':checkpoint,'AUDIO_OBJECT_SOURCE_ROOT':root,'AUDIO_OBJECT_CACHE':tmp_path/'cache'}.items():monkeypatch.setenv(name,str(value))
    assert objects.capabilities()['available']
    source=tmp_path/'input.mp4';source.write_bytes(b'video')
    calls=[];lock=threading.Lock()
    def run(command,**kwargs):
        assert lock.locked()
        assert command[0]==str(python) and command[command.index('--concept')+1]=='dog'
        calls.append(command)
        out=Path(command[command.index('--output-dir')+1])
        (out/'detections.json').write_text(json.dumps({'frames':[],'annotated_media':'private.mp4','source':'private.mp4'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(objects.subprocess,'run',run)
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda pid:objects.analyze(source,'dog',pid,lock),['project-a','project-b']))
    assert len(calls)==1 and sorted(r['cache_hit'] for r in results)==[False,True]
    assert [r['video_id'] for r in results]==['project-a','project-b']
    assert all(r['selection_only'] and 'annotated_media' not in r for r in results)


def test_unconfigured_sam_is_reported_unavailable(monkeypatch):
    for key in ['AUDIO_OBJECT_PYTHON','AUDIO_OBJECT_CHECKPOINT','AUDIO_OBJECT_SOURCE_ROOT']:monkeypatch.delenv(key,raising=False)
    assert objects.capabilities()['available'] is False
