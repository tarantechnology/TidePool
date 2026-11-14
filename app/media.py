import base64
import json
import math
import os
from pathlib import Path
import httpx
from . import guidance
from .core import max_passes, run, probe, uid


class InferenceError(RuntimeError):
    """Audio-service failure that carries the service's structured error fields."""
    def __init__(self, message, code=None, retryable=None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def separate(audio, directory, payload, report=None):
    directory.mkdir(parents=True, exist_ok=True)
    mode = os.getenv('INFERENCE_ADAPTER', 'fake')
    if mode == 'fake':
        passes=int((payload.get('options') or {}).get('passes') or 1)
        if report is not None:
            report.update(model={'backend':'fake','is_mock':True,'passes':passes}, warnings=['Fake adapter copies audio; no source separation performed'])
        # Deterministic fixtures, not source separation: each stem is a copy.
        results = []
        names=payload.get('prompts') or ['Audio']
        if payload.get('request_id') and payload.get('profile'): names=[*names,'residual']
        for name in names:
            if isinstance(name,dict): name=name.get('name') or name['text']
            path = directory / (uid() + '.wav')
            extra=['-af','volume=0'] if name=='residual' and payload.get('profile') else []
            run('ffmpeg','-v','error','-y','-i',str(audio),*extra,'-c:a','pcm_f32le',str(path))
            role='residual' if name.lower()=='residual' else 'target'
            # Mirrors the service's per-target refinement block; the fake measures nothing.
            refinement={'passes_requested':passes,'passes_run':passes,'per_pass_energy_db':[0.0]*passes,'stopped_early':False} if role=='target' else None
            results.append({'name':name,'path':path,'role':role,'prompt':name,'label':name,'refinement':refinement})
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
        code = retryable = None
        try:
            error = response.json()['error']
            code = error['code']
            message = f"{code}: {error['message']}"
            if isinstance(error.get('retryable'), bool): retryable = error['retryable']
        except (ValueError, KeyError, TypeError):
            try:
                detail = response.json().get('detail')
                message = f'Audio service HTTP {response.status_code}: {detail}' if isinstance(detail, str) else f'Audio service HTTP {response.status_code}'
            except (ValueError, AttributeError):
                message = f'Audio service HTTP {response.status_code}'
        raise InferenceError(message,code,retryable)
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
        stems.append({'name': name, 'path': path, 'role': item.get('role'), 'source_stem_id': item.get('stem_id'),
                      'prompt': item.get('prompt'), 'label': item.get('label'), 'sample_rate': item.get('sample_rate'),
                      'channels': item.get('channels'), 'format': item.get('format'), 'spans': item.get('spans'),
                      'refinement': item.get('refinement') if isinstance(item.get('refinement'), dict) else None})
    return stems

def read_wav(path):
    """Decode a PCM/float WAV into (float64 samples [frames, channels], sample_rate)."""
    import struct
    import numpy as np
    data = Path(path).read_bytes()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE': raise ValueError(f'{Path(path).name} is not a WAV file')
    pos, fmt, body = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos+4], struct.unpack('<I', data[pos+4:pos+8])[0]
        chunk = data[pos+8:pos+8+size]
        if cid == b'fmt ': fmt = chunk
        elif cid == b'data': body = chunk
        pos += 8 + size + (size & 1)
    if fmt is None or body is None: raise ValueError('WAV is missing fmt or data chunk')
    tag, channels, rate = struct.unpack('<HHI', fmt[:8]); bits = struct.unpack('<H', fmt[14:16])[0]
    if tag == 0xFFFE and len(fmt) >= 26: tag = struct.unpack('<H', fmt[24:26])[0]
    width = bits // 8
    body = body[:len(body) - len(body) % (width * channels)]
    if tag == 3 and bits in (32, 64): samples = np.frombuffer(body, '<f4' if bits == 32 else '<f8').astype(np.float64)
    elif tag == 1 and bits == 16: samples = np.frombuffer(body, '<i2') / 32768.0
    elif tag == 1 and bits == 32: samples = np.frombuffer(body, '<i4') / 2147483648.0
    elif tag == 1 and bits == 24:
        raw = np.frombuffer(body, np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = np.where(value >= 1 << 23, value - (1 << 24), value) / 8388608.0
    else: raise ValueError(f'Unsupported WAV encoding (format {tag}, {bits} bit)')
    return samples.reshape(-1, channels), rate


def write_wav(path, samples, rate):
    """Write float32 IEEE WAV, the format every stem uses."""
    Path(path).write_bytes(encode_wav(samples, rate))


def encode_wav(samples, rate):
    import struct
    import numpy as np
    samples = np.ascontiguousarray(samples, dtype='<f4')
    frames, channels = samples.shape
    body = samples.tobytes()
    header = (b'RIFF' + struct.pack('<I', 4 + 26 + 12 + 8 + len(body)) + b'WAVE'
              + b'fmt ' + struct.pack('<IHHIIHHH', 18, 3, channels, rate, rate * channels * 4, channels * 4, 32, 0)
              + b'fact' + struct.pack('<II', 4, frames)
              + b'data' + struct.pack('<I', len(body)))
    return header + body


def energy_db(samples):
    """Mean-square level in dBFS, floored at -120 so silence stays JSON-safe."""
    import numpy as np
    power = float(np.mean(np.square(samples))) if samples.size else 0.0
    return round(max(-120.0, 10 * math.log10(power)) if power > 0 else -120.0, 3)


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
        return {'preview':True,'quality':False,'visual':False,'experimental_visual':True,'reason':'Fake adapter: no AI separation or judge ranking.',
                'max_passes':max_passes(),'refine':True}
    from urllib.parse import urlsplit, urlunsplit
    url=urlsplit(os.getenv('SAM_AUDIO_SERVICE_URL',''))
    try:
        response=httpx.get(urlunsplit((url.scheme,url.netloc,'/capabilities','','')),timeout=5)
        response.raise_for_status(); data=response.json(); model=data.get('model',{})
        defaults=data.get('defaults') or {}
        ranked=bool(model.get('text_ranker_loaded')); consistent='candidate_select' in defaults and data.get('independent_candidates', True) is not False
        quality=bool(model.get('loaded') and not data.get('is_mock') and (ranked or consistent))
        reason='' if quality else ('Quality retry needs a real model (the mock backend cannot rank or refine).' if data.get('is_mock') or not model.get('loaded')
                                   else 'Quality retry requires the pretrained text ranker or a service with per-window candidate selection.')
        return {'preview':True,'quality':quality,'visual':bool(data.get('prompting_modes',{}).get('visual')),
                'experimental_visual':True,**({'span_anchors': data['prompting_modes']['span_anchors']} if isinstance((data.get('prompting_modes') or {}).get('span_anchors'), bool) else {}),'reason':reason,'quality_mode':('ranker' if ranked else 'consistency') if quality else None,
                'profiles':{'preview':{'candidates':1,'chunk_seconds':10,'chunk_batch':2},'quality':{'candidates':2,'chunk_seconds':10,'chunk_batch':2,**guidance.profile_extras('quality')}},
                'long_window_promoted':False,'automatic_spans':False,**passes_capability(data)}
    except (httpx.HTTPError,ValueError,TypeError,AttributeError):
        return {'preview':True,'quality':False,'visual':False,'experimental_visual':True,'reason':'Service capabilities unavailable; quality retry is disabled.',
                'max_passes':max_passes(),'refine':True}


def passes_capability(data):
    """Backend MAX_PASSES, narrowed by the service's advertised passes limit when it reports one."""
    def number(value):
        if isinstance(value, dict): value = value.get('max', value.get('default'))
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None
    limit = number((data.get('limits') or {}).get('passes'))
    default = number((data.get('defaults') or {}).get('passes'))
    result = {'max_passes': min(max_passes(), limit) if limit else max_passes(), 'refine': True}
    if default: result['default_passes'] = min(default, result['max_passes'])
    return result


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
