import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from sam_audio_service import objects


@pytest.mark.parametrize('concept,adapter', [('dog','sam3.1'),('auto','yolo')])
def test_sam_cache_and_inference_lock_prevent_duplicate_gpu_work(tmp_path,monkeypatch,concept,adapter):
    python=tmp_path/'python';python.touch();checkpoint=tmp_path/'sam.pt';checkpoint.touch()
    root=tmp_path/'project';code=root/'vision/soundcanvas_cv';code.mkdir(parents=True)
    for name in ['sam3_adapter.py','yolo_adapter.py','pipeline.py']:(code/name).write_text('tested')
    discovery=tmp_path/'yolo.pt';discovery.write_bytes(b'cached discovery checkpoint')
    for name,value in {'AUDIO_OBJECT_PYTHON':python,'AUDIO_OBJECT_CHECKPOINT':checkpoint,'AUDIO_OBJECT_DISCOVERY_MODEL':discovery,'AUDIO_OBJECT_SOURCE_ROOT':root,'AUDIO_OBJECT_CACHE':tmp_path/'cache'}.items():monkeypatch.setenv(name,str(value))
    assert objects.capabilities()['available']
    assert objects.capabilities()['automatic']
    source=tmp_path/'input.mp4';source.write_bytes(b'video')
    calls=[];lock=threading.Lock()
    def run(command,**kwargs):
        assert lock.locked()
        assert command[0]==str(python) and command[command.index('--concept')+1]==concept
        assert command[command.index('--adapter')+1]==adapter
        assert command[command.index('--model')+1]==str(discovery if concept=='auto' else checkpoint)
        calls.append(command)
        out=Path(command[command.index('--output-dir')+1])
        frames=[{'objects':[{'track_id':5,'class_id':c}]} for c in [0,16,0]]
        (out/'detections.json').write_text(json.dumps({'frames':frames,'annotated_media':'private.mp4','source':'private.mp4'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(objects.subprocess,'run',run)
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda pid:objects.analyze(source,concept,pid,lock),['project-a','project-b']))
    assert len(calls)==1 and sorted(r['cache_hit'] for r in results)==[False,True]
    assert [r['video_id'] for r in results]==['project-a','project-b']
    assert all(r['selection_only'] and 'annotated_media' not in r for r in results)
    assert all(r['discovery_mode']==('automatic' if concept=='auto' else 'prompted') for r in results)
    expected=[1,2,1] if concept=='auto' else [5,5,5]
    assert all([f['objects'][0]['track_id'] for f in r['frames']]==expected for r in results)
    # An automatic scan and a specific SAM refinement must never share cached masks.
    other='dog' if concept=='auto' else 'auto'
    def other_run(command,**kwargs):
        out=Path(command[command.index('--output-dir')+1])
        (out/'detections.json').write_text('{"frames":[]}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(objects.subprocess,'run',other_run)
    assert objects.analyze(source,other,'project-c',lock)['cache_hit'] is False


def test_unconfigured_sam_is_reported_unavailable(monkeypatch):
    for key in ['AUDIO_OBJECT_PYTHON','AUDIO_OBJECT_CHECKPOINT','AUDIO_OBJECT_SOURCE_ROOT']:monkeypatch.delenv(key,raising=False)
    assert objects.capabilities()['available'] is False
