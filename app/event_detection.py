"""Lightweight onset proposals and example matching, without source-identity claims."""
from __future__ import annotations
import subprocess
import numpy as np

RATE=16000
HOP=160
FFT=512


def decode_audio(path, duration):
    if duration>600: raise ValueError('The clip is longer than 10 minutes. Please choose a shorter range for example matching.')
    result=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(RATE),'-f','f32le','pipe:1'],capture_output=True,check=True,timeout=180)
    return np.frombuffer(result.stdout,dtype='<f4').copy()


def onset_features(samples, rate=RATE):
    """Spectral-flux attacks and local spectral fingerprints for candidate ranking."""
    if rate!=RATE: raise ValueError('Expected 16 kHz audio.')
    samples=np.asarray(samples,dtype=np.float32)
    if len(samples)<FFT:return np.array([]),np.empty((0,32),np.float32)
    windows=np.lib.stride_tricks.sliding_window_view(samples,FFT)[::HOP]
    # Batch to bound peak memory on long recordings.
    spectra=[]
    window=np.hanning(FFT).astype(np.float32)
    for begin in range(0,len(windows),4096):
        spectra.append(np.abs(np.fft.rfft(windows[begin:begin+4096]*window,axis=1)).astype(np.float32))
    spec=np.concatenate(spectra);freq=np.fft.rfftfreq(FFT,1/rate)
    spec[:,freq<120]=0
    log=np.log1p(spec*10);flux=np.mean(np.maximum(0,np.diff(log,axis=0,prepend=log[:1])),axis=1)
    flux=np.convolve(flux,[.25,.5,.25],mode='same')
    # Local adaptive threshold, with explicit silence gate and peak-spacing control.
    candidates=[]
    for i in range(1,len(flux)-1):
        if flux[i]<flux[i-1] or flux[i]<=flux[i+1]: continue
        local=flux[max(0,i-25):min(len(flux),i+26)]
        median=float(np.median(local));mad=float(np.median(np.abs(local-median)))
        if flux[i]<max(.008,median+max(.008,1.8*mad)):continue
        candidates.append(i)
    chosen=[];blocked=np.zeros(len(flux),dtype=bool)
    for i in sorted(candidates,key=lambda n:float(flux[n]),reverse=True):
        if not blocked[i]:
            chosen.append(i);blocked[max(0,i-8):min(len(flux),i+9)]=True
    chosen.sort()
    # Fixed dimensional band representation; duplicates near low frequencies are okay.
    raw_edges=np.geomspace(4,spec.shape[1]-1,33).astype(int)
    vectors=[]
    for i in chosen:
        after=np.mean(spec[i:min(len(spec),i+10)],axis=0)
        before=np.mean(spec[max(0,i-14):max(1,i-4)],axis=0)
        contrast=np.maximum(after-before,0)
        bins=np.array([np.sum(contrast[a:max(a+1,b)]) for a,b in zip(raw_edges[:-1],raw_edges[1:])],dtype=np.float32)
        bins=np.sqrt(bins);norm=np.linalg.norm(bins)
        vectors.append(bins/norm if norm>1e-8 else bins)
    # Offset by half a frame; this is approximate, with final alignment refined later.
    return (np.asarray(chosen,dtype=np.float64)*HOP+FFT/2)/rate, np.asarray(vectors,dtype=np.float32).reshape(-1,32)


def find_similar_events(path, duration, example_time, threshold=.84):
    samples=decode_audio(path,duration)
    times,vectors=onset_features(samples)
    if not len(times):raise ValueError('I could not find a clear sound near that time. Try a different example timestamp.')
    idx=int(np.argmin(abs(times-example_time)))
    if abs(float(times[idx])-example_time)>.4:raise ValueError('I could not find a clear sound near that time. Try a different example timestamp.')
    if np.linalg.norm(vectors[idx])<.01:raise ValueError('The example was too quiet to match. Try a clearer example timestamp.')
    similarity=vectors@vectors[idx]; selected=np.flatnonzero(similarity>=threshold)
    if len(selected)>100:raise ValueError('The example matched too many sounds. Try a more distinctive example or a shorter clip.')
    # Merge nearby candidate selections to reduce repeated inference, with a 10-second cap.
    ranges=[]
    for i in selected:
        t=float(times[i]);start=max(0,t-.08);end=min(duration,t+.28)
        if ranges and start<=ranges[-1]['end']+.25 and end-ranges[-1]['start']<=10:
            ranges[-1]['end']=end
        else:ranges.append({'start':start,'end':end})
    if len(ranges)>20:raise ValueError('The example matched too many separate sections. Try a shorter clip or a more distinctive example.')
    return {'intervals':ranges,'times':[round(float(times[i]),3) for i in selected],
            'example_time':round(float(times[idx]),3),'method':'spectral-example-matching',
            'notice':'Similar-sound matches are approximate; review the proposed placements.'}


def flexsed_intervals(path, duration, query):
    """Return coarse text-conditioned regions, or None when not configured."""
    import json,os
    from pathlib import Path
    root=os.getenv('SOUNDCANVAS_FLEXSED_ROOT','').strip()
    checkpoint=os.getenv('SOUNDCANVAS_FLEXSED_CHECKPOINT','').strip()
    python=os.getenv('SOUNDCANVAS_FLEXSED_PYTHON','').strip()
    if not root or not checkpoint or not python:return None
    if duration>600:raise ValueError('The clip is longer than 10 minutes. Please choose a shorter range.')
    if not all(Path(p).exists() for p in (root,checkpoint,python)):raise RuntimeError('The optional sound detector is unavailable.')
    env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
    env['PYTHONPATH']=root+os.pathsep+env.get('PYTHONPATH','')
    payload={'audio':str(path),'checkpoint':checkpoint,'query':query[:200]}
    try:
        process=subprocess.run([python,str(Path(__file__).with_name('flexsed_worker.py'))],input=json.dumps(payload),capture_output=True,text=True,timeout=300,check=True,cwd=root,env=env)
        lines=[line for line in process.stdout.splitlines() if line.startswith('SOUNDCANVAS_FLEXSED=')]
        result=json.loads(lines[-1].split('=',1)[1]);scores=result['scores'];fps=float(result['fps'])
        if fps<=0 or len(scores)>round(duration*fps)+100:raise ValueError('Invalid detector output.')
    except (OSError,subprocess.SubprocessError,KeyError,IndexError,ValueError,TypeError) as exc:raise RuntimeError('The optional sound detector failed.') from exc
    intervals=[];start=None
    for i,score in enumerate(scores):
        on=type(score) in (int,float) and np.isfinite(score) and score>=.5
        if on and start is None:start=i/fps
        if not on and start is not None:
            intervals.append({'start':max(0,start-.08),'end':min(duration,i/fps+.08)});start=None
    if start is not None:intervals.append({'start':max(0,start-.08),'end':duration})
    return {'intervals':intervals,'method':'flexsed-experimental','seconds':result.get('seconds'),'peak_gpu_bytes':result.get('peak_gpu_bytes')}


def candidate_times_in_regions(path, duration, regions):
    times,_=onset_features(decode_audio(path,duration))
    return [round(float(t),3) for t in times if any(r['start']-.1<=t<=r['end']+.1 for r in regions)]


def merge_candidate_ranges(times,duration):
    ranges=[]
    for t in sorted(times):
        start=max(0,t-.08);end=min(duration,t+.28)
        if end<=start:continue
        if ranges and start<=ranges[-1]['end']+.25 and end-ranges[-1]['start']<=10:ranges[-1]['end']=end
        else:ranges.append({'start':start,'end':end})
    if len(ranges)>20 or len(times)>100:raise ValueError('Too many possible sounds. Please narrow the selection or give a more distinctive example.')
    return ranges
