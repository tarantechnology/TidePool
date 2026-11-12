import base64
import json
import os
from pathlib import Path
import httpx
from .core import run, probe, uid

def separate(audio, directory, payload, report=None):
    directory.mkdir(parents=True, exist_ok=True)
    mode = os.getenv('INFERENCE_ADAPTER', 'fake')
    if mode == 'fake':
        if report is not None:
            report.update(model={'backend':'fake','is_mock':True}, warnings=['Fake adapter copies audio; no source separation performed'])
        # Deterministic fixtures, not source separation: each stem is a copy.
        results = []
        names=payload.get('prompts') or ['Audio']
        if payload.get('request_id') and payload.get('profile'): names=[*names,'residual']
        for name in names:
            if isinstance(name,dict): name=name.get('name') or name['text']
            path = directory / (uid() + '.wav')
            extra=['-af','volume=0'] if name=='residual' and payload.get('profile') else []
            run('ffmpeg','-v','error','-y','-i',str(audio),*extra,'-c:a','pcm_f32le',str(path))
            results.append((name, path))
        return results
    if mode != 'http': raise RuntimeError('Unknown inference adapter')
    url = os.getenv('SAM_AUDIO_SERVICE_URL')
    if not url: raise RuntimeError('SAM_AUDIO_SERVICE_URL is required')
    try:
        with open(audio, 'rb') as f:
            response = httpx.post(url, files={'audio': ('audio.wav', f, 'audio/wav')}, data={'metadata': json.dumps(payload)}, timeout=float(os.getenv('INFERENCE_TIMEOUT_SECONDS','3600')))
    except httpx.TimeoutException as exc:
        raise RuntimeError('Inference timed out; retry the job or increase INFERENCE_TIMEOUT_SECONDS') from exc
    if response.is_error:
        try:
            error = response.json()['error']
            message = f"{error['code']}: {error['message']}"
        except (ValueError, KeyError, TypeError):
            try:
                detail = response.json().get('detail')
                message = f'Audio service HTTP {response.status_code}: {detail}' if isinstance(detail, str) else f'Audio service HTTP {response.status_code}'
            except (ValueError, AttributeError):
                message = f'Audio service HTTP {response.status_code}'
        raise RuntimeError(message)
    result = response.json()
    if report is not None:
        report.update({key:result[key] for key in ('model','warnings','source','request_id','contract_version') if key in result})
    if not isinstance(result.get('stems'), list) or not result['stems']: raise ValueError('Inference returned no stems')
    source_info = probe(audio) if payload.get('source') else None
    stems = []
    for item in result['stems']:
        name = item['name']
        raw = base64.b64decode(item['audio_base64'], validate=True)
        if len(raw) > int(os.getenv('MAX_STEM_BYTES',str(500*1024*1024))): raise ValueError('Inference stem too large')
        path = directory / (uid()+'.wav')
        path.write_bytes(raw)
        info = probe(path)
        if source_info:
            # Visual results must already be reconstructed on the complete audio timeline.
            if (item.get('start_time', 0) != 0 or abs(info['duration'] - source_info['duration']) > 1e-6
                    or (info['sample_rate'], info['channels']) != (source_info['sample_rate'], source_info['channels'])):
                path.unlink(missing_ok=True)
                raise ValueError('Visual stem must start at zero and match the full extracted audio duration, sample rate and channels')
        stems.append((name,path))
    return stems

def export(stems, original, destination, fmt, include_video):
    active = [s for s in stems if not s['muted'] and (not any(x['solo'] for x in stems) or s['solo'])]
    # A silent source maintains duration for an all-muted mix.
    duration = probe(original)['duration']
    args = ['ffmpeg','-v','error','-y']
    if include_video: args += ['-i',str(original)]
    for s in active: args += ['-i',s['path']]
    args += ['-f','lavfi','-t',str(duration),'-i','anullsrc=r=48000:cl=stereo']
    filters=[]; labels=[]; base = 1 if include_video else 0
    for i,s in enumerate(active):
        points=json.loads(s['automation']); ranges=json.loads(s['mute_ranges'])
        expr=str(float(s['volume']))
        # Keyframes are linear, in project timeline seconds.
        if points:
            expr=str(float(points[-1]['value']))
            for j in range(len(points)-2,-1,-1):
                a,b=points[j],points[j+1]; t1=float(a['time']); t2=float(b['time']); v1=float(a['value']); v2=float(b['value'])
                expr=f'if(lt(t,{t2}),{v1}+({v2}-{v1})*(t-{t1})/({t2}-{t1}),{expr})'
            expr=f"({s['volume']})*if(lt(t,{float(points[0]['time'])}),{float(points[0]['value'])},{expr})"
        for rg in ranges: expr=f'({expr})*if(between(t,{float(rg[0])},{float(rg[1])}),0,1)'
        delay=round(float(s['start_offset'])*1000)
        filters.append(f"[{base+i}:a]adelay={delay}:all=1,volume='{expr}':eval=frame[a{i}]")
        labels.append(f'[a{i}]')
    silent=base+len(active)
    labels.append(f'[{silent}:a]')
    filters.append(''.join(labels)+f'amix=inputs={len(labels)}:duration=longest:normalize=0[mix]')
    args += ['-filter_complex',';'.join(filters),'-map','[mix]']
    if include_video: args += ['-map','0:v:0','-c:v','copy','-c:a','aac','-t',str(duration),'-movflags','+faststart']
    elif fmt=='mp3': args += ['-c:a','libmp3lame']
    else: args += ['-c:a','pcm_f32le']
    run(*args,str(destination))


def capabilities():
    """Whitelist public capability fields; never expose URLs, paths or load errors."""
    if os.getenv('INFERENCE_ADAPTER','fake') != 'http':
        return {'preview':True,'quality':False,'visual':False,'experimental_visual':True,'reason':'Fake adapter: no AI separation or judge ranking.'}
    from urllib.parse import urlsplit, urlunsplit
    url=urlsplit(os.getenv('SAM_AUDIO_SERVICE_URL',''))
    try:
        response=httpx.get(urlunsplit((url.scheme,url.netloc,'/capabilities','','')),timeout=5)
        response.raise_for_status(); data=response.json(); model=data.get('model',{})
        quality=bool(model.get('loaded') and model.get('text_ranker_loaded') and not data.get('is_mock'))
        return {'preview':True,'quality':quality,'visual':bool(data.get('prompting_modes',{}).get('visual')),
                'experimental_visual':True,'reason':'' if quality else 'Quality retry requires the pretrained text ranker to be loaded.',
                'profiles':{'preview':{'candidates':1,'chunk_seconds':10,'chunk_batch':2},'quality':{'candidates':2,'chunk_seconds':10,'chunk_batch':2}},
                'long_window_promoted':False,'automatic_spans':False}
    except (httpx.HTTPError,ValueError,TypeError,AttributeError):
        return {'preview':True,'quality':False,'visual':False,'experimental_visual':True,'reason':'Service capabilities unavailable; quality retry is disabled.'}


def inspect_audio(path):
    import re
    import subprocess
    result=subprocess.run(['ffmpeg','-v','info','-i',str(path),'-af','astats=metadata=0:reset=0','-f','null','-'],capture_output=True,text=True,timeout=600,check=True)
    text=result.stderr.split('Overall')[-1]
    def level(name):
        match=re.search(name+r':\s*([-+\w.]+)',text)
        if not match: return None
        value=float(match[1])
        return value if __import__('math').isfinite(value) else None
    peak,rms=level('Peak level dB'),level('RMS level dB')
    findings=[]
    if rms is None or rms < -70: findings.append('Audio is silent or nearly silent. Choose an audible recording.')
    if peak is not None and peak >= -.01: findings.append('Audio reaches full scale and may be clipped. Separation cannot restore missing waveform detail.')
    return {'peak_dbfs':peak,'rms_dbfs':rms,'findings':findings}
