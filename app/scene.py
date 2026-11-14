"""Bounded visual context; never use visual presence as evidence of an audible source."""
import json
import os
import subprocess
import sys
from .candidates import safe_path


def describe(project):
    metadata=json.loads(project.get('metadata') or '{}')
    if not metadata.get('has_video'): return {'status':'not_video'}
    model=os.getenv('SOUNDCANVAS_VISION_LANGUAGE_MODEL','').strip()
    if not model: return {'status':'unavailable'}
    video=safe_path(project['id'],project['original'])
    payload={'video':str(video),'duration':metadata['duration'],'model':model}
    try:
        result=subprocess.run([sys.executable,'-m','app.scene_worker'],input=json.dumps(payload),capture_output=True,text=True,timeout=180,check=True,env=os.environ.copy())
        parsed=json.loads(result.stdout.strip().splitlines()[-1])
        description=parsed.get('description')
        if not isinstance(description,str) or not description.strip(): raise ValueError('Missing visual description')
        return {'status':'ok','description':description[:1200], 'frame_times':parsed.get('frame_times',[]),'model':parsed.get('model','visual model'),'elapsed_seconds':parsed.get('elapsed_seconds'),'peak_gpu_bytes':parsed.get('peak_gpu_bytes')}
    except (OSError,ValueError,KeyError,subprocess.SubprocessError):
        return {'status':'unavailable'}
