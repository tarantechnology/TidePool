"""Visual selection as text-edit context, independent of audio visual conditioning."""
import json
import os
from pathlib import Path
import httpx
from .core import run
from .visual import file_sha256


def service_url():
    return os.getenv('VISION_SERVICE_URL', '').rstrip('/')


def status():
    try:
        response = httpx.get(service_url()+'/objects/capabilities', timeout=5)
        response.raise_for_status()
        data = response.json()
        return {'available': data.get('available') is True, 'reason': data.get('reason'),
                'model': data.get('model'), 'device': data.get('device'),
                'automatic': data.get('automatic') is True, 'prompted': data.get('prompted',data.get('available')) is True,
                'discovery_model': data.get('discovery_model'),
                'weights_cached': data.get('weights_cached') is True, 'selection_only': True}
    except (httpx.HTTPError, ValueError, AttributeError):
        return {'available': False, 'retryable': True, 'reason': 'Object selection service is unreachable; check the GPU tunnel and retry.',
                'model': 'sam3.1', 'device': 'cuda', 'weights_cached': None, 'selection_only': True}


def analysis_fps():
    # Selection needs responsive outlines; audio/export retain their native cadence.
    fps = float(os.getenv('OBJECT_ANALYSIS_FPS', '10'))
    if not 1 <= fps <= 30:
        raise ValueError('OBJECT_ANALYSIS_FPS must be between 1 and 30')
    return fps


def detect(source, output_dir, video_id, concept):
    fps = analysis_fps()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source)
    # A video-only analysis proxy leaves extracted audio and export media untouched.
    info = json.loads(run('ffprobe','-v','error','-show_streams','-of','json',str(source)))
    streams = info['streams']
    video = next(s for s in streams if s['codec_type']=='video')
    audio = next(s for s in streams if s['codec_type']=='audio')
    if abs(float(video.get('start_time',0))-float(audio.get('start_time',0))) > .05:
        raise ValueError('Object selection needs aligned audio/video start times; the original was preserved.')
    proxy = output_dir/'analysis.mp4'
    run('ffmpeg','-v','error','-y','-i',str(source),'-map','0:v:0','-an',
        '-vf',f'setpts=PTS-STARTPTS,fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-t','30','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(proxy))
    try:
        with proxy.open('rb') as stream:
            response = httpx.post(service_url()+'/objects/analyze',
                                  files={'video':('analysis.mp4',stream,'video/mp4')},
                                  data={'concept':concept,'video_id':video_id},timeout=1000)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError('SAM object selection failed; check the GPU connection and retry.') from exc
    if (data.get('video_id') != video_id or data.get('selection_only') is not True
            or data.get('source_sha256') != file_sha256(proxy) or not isinstance(data.get('frames'),list)):
        raise ValueError('Object detector returned mismatched media; run detection again.')
    data['original_source_sha256'] = source_hash
    (output_dir/'detections.json').write_text(json.dumps(data))
    return data


def validate_context(project, selection):
    stored = json.loads(project['visual']) if project['visual'] else {}
    if (stored.get('selection_only') is not True or stored.get('video_id') != project['id'] or stored.get('analysis_id') != selection['analysis_id']
            or stored.get('original_source_sha256') != file_sha256(project['original'])):
        raise ValueError('Object selection is stale; find objects again.')
    frame = next((f for f in stored.get('frames',[]) if f['frame_index']==selection['frame_index']),None)
    obj = next((o for o in (frame or {}).get('objects',[]) if o['detection_id']==selection['detection_id'] and o['track_id']==selection['track_id']),None)
    if not obj or not obj.get('mask'):
        raise ValueError('Selected object is unavailable in this analysis.')
    return {**selection,'label':obj['label'],'model':(stored.get('detector') or {}).get('model'),
            'audio_conditioning':'text_only','notice':'Object selection supplies a sound description; masks do not condition AudioSep.'}


def appearances(project, track_id, label):
    """Find starts of contiguous observations for one explicitly named track."""
    import math
    from collections import Counter
    stored=json.loads(project['visual']) if project['visual'] else {}
    if (stored.get('selection_only') is not True or stored.get('video_id')!=project['id']
            or not stored.get('analysis_id') or stored.get('original_source_sha256')!=file_sha256(project['original'])):
        raise ValueError('Find objects in this video first, then use the displayed object number.')
    fps=float(stored.get('fps') or 0); processed=float(stored.get('processed_duration_seconds') or 0)
    if not math.isfinite(fps) or not 1<=fps<=120 or not math.isfinite(processed) or processed<=0:
        raise ValueError('Object analysis timing is unavailable; find objects again.')
    observations=[];labels=Counter()
    for frame in stored.get('frames',[]):
        t=frame.get('timestamp_seconds')
        if not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<processed: continue
        obj=next((o for o in frame.get('objects',[]) if o.get('track_id')==track_id and o.get('mask')),None)
        if obj:
            observations.append(float(t));labels[str(obj.get('label','')).casefold()]+=1
    minimum=3 if stored.get('discovery_mode')=='automatic' else 1
    if len(observations)<minimum: raise ValueError(f'Object #{track_id} was not found in the current analysis. Find objects and choose a displayed number.')
    best=labels.most_common(1)[0][0]
    if best!=label.casefold(): raise ValueError(f'Object #{track_id} is labeled {best}, not {label}. Choose the number shown in Find objects.')
    times=sorted(set(observations)); groups=[[times[0]]]
    for previous,current in zip(times,times[1:]):
        if current-previous>max(.3,2.1/fps)+1e-6: groups.append([])
        groups[-1].append(current)
    starts=[group[0] for group in groups if len(group)>=minimum]
    if not starts: raise ValueError(f'Object #{track_id} has no stable detected appearances. Find objects again or choose another track.')
    if len(starts)>100: raise ValueError('More than 100 appearances were found. Try a shorter video.')
    return {'analysis_id':stored['analysis_id'],'source_sha256':stored['original_source_sha256'],
            'label':best,'track_id':track_id,'starts':starts,'processed_duration':processed}
