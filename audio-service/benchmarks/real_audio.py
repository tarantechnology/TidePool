"""Benchmark speech-over-music separation using existing recordings, without labels.

Inputs are local: a LibriSpeech parquet file containing FLAC bytes and a music recording.
The two are mixed here, so the clean voice is already ground truth. Nothing is uploaded
except the constructed mixture to the explicitly configured inference endpoint.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from sam_audio_service.metrics import si_sdr_improvement, reconstruction_error

RATE = 48000


def mono(audio, source_rate):
    data=np.asarray(audio,dtype=np.float32)
    if data.ndim==2: data=data.mean(axis=1)
    if source_rate!=RATE:
        divisor=gcd(int(source_rate),RATE)
        data=resample_poly(data,RATE//divisor,int(source_rate)//divisor).astype(np.float32)
    return data


def build_mixture(voice, background, duration):
    count=int(duration*RATE)
    if len(voice)<count or len(background)<count: raise ValueError('recordings are shorter than requested clip')
    voice=voice[:count].astype(np.float32); background=background[:count].astype(np.float32)
    # Fixed equal RMS gives reproducible 0 dB speech/music ratio, without clipping.
    def unit(x): return x/max(float(np.sqrt(np.mean(x.astype(np.float64)**2))),1e-7)
    voice=unit(voice);background=unit(background)
    scale=.9/max(float(np.max(np.abs(voice+background))),1e-7)
    return voice*scale,background*scale


def score(target, reference, mixture):
    n=min(len(target),len(reference),len(mixture))
    if n < len(reference): raise ValueError('output shorter than the input')
    return float(si_sdr_improvement(target[:n][None,:],reference[:n][None,:],mixture[:n][None,:]))


def sam(url, mixture, prompt, seed, candidates=1):
    buffer=io.BytesIO();sf.write(buffer,mixture,RATE,format='WAV',subtype='PCM_16')
    meta={'prompts':[prompt],'options':{'seed':seed,'reranking_candidates':candidates}}
    started=time.monotonic()
    response=httpx.post(url,files={'audio':('mix.wav',buffer.getvalue(),'audio/wav')},data={'metadata':json.dumps(meta)},timeout=600)
    elapsed=time.monotonic()-started;response.raise_for_status();body=response.json()
    target=next(x for x in body['stems'] if x['role']=='target')
    residual=next(x for x in body['stems'] if x['role']=='residual')
    t,sr=sf.read(io.BytesIO(base64.b64decode(target['audio_base64'])),dtype='float32');r,_=sf.read(io.BytesIO(base64.b64decode(residual['audio_base64'])),dtype='float32')
    return mono(t,sr),mono(r,sr),elapsed,body


def demucs(python, mixture, directory):
    source=directory/'mix.wav';sf.write(source,mixture,RATE,subtype='PCM_16')
    started=time.monotonic()
    result=subprocess.run([python,'-m','demucs','-n','htdemucs','--two-stems','vocals','--float32','--device','cuda','--shifts','0','-o',str(directory/'out'),str(source)],capture_output=True,text=True,timeout=600)
    elapsed=time.monotonic()-started
    if result.returncode: raise RuntimeError('Demucs failed: '+result.stderr[-1200:])
    path=directory/'out'/'htdemucs'/'mix'/'vocals.wav';target,sr=sf.read(path,dtype='float32')
    return mono(target,sr),elapsed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speech-parquet',required=True);p.add_argument('--music',required=True)
    p.add_argument('--url',default='http://127.0.0.1:18001/separate');p.add_argument('--demucs-python')
    p.add_argument('--seconds',type=float,default=6);p.add_argument('--clips',type=int,default=3)
    p.add_argument('--repeat',type=int,default=2);p.add_argument('--candidates',type=int,default=1)
    p.add_argument('--out',required=True);args=p.parse_args()
    if args.seconds<=0 or args.clips<1 or args.repeat<1: p.error('seconds, clips and repeat must be positive')
    table=pq.read_table(args.speech_parquet,columns=['audio','id'])
    background,bg_rate=sf.read(args.music,dtype='float32');background=mono(background,bg_rate)
    rows=[];clips=0
    for row in table.to_pylist():
        speech,sr=sf.read(io.BytesIO(row['audio']['bytes']),dtype='float32');speech=mono(speech,sr)
        if len(speech)<args.seconds*RATE: continue
        voice,music=build_mixture(speech,background,args.seconds);mixture=voice+music
        for prompt in ('person speaking','spoken narration'):
            for attempt in range(args.repeat):
                try:
                    target,residual,elapsed,body=sam(args.url,mixture,prompt,attempt+1,args.candidates)
                    rows.append({'clip':row['id'],'method':'sam','prompt':prompt,'seed':attempt+1,'candidates':args.candidates,
                        'si_sdri_db':score(target,voice,mixture),'latency_seconds':round(elapsed,3),
                        'reconstruction_error':float(reconstruction_error([target[None,:],residual[None,:]],mixture[None,:])),
                        'model':body.get('model'),'warnings':body.get('warnings',[])})
                except (httpx.HTTPError,ValueError,KeyError,StopIteration) as exc:
                    rows.append({'clip':row['id'],'method':'sam','prompt':prompt,'seed':attempt+1,'error':str(exc)[:500]})
        if args.demucs_python:
            try:
                with tempfile.TemporaryDirectory(prefix='sc-demucs-') as tmp:
                    target,elapsed=demucs(args.demucs_python,mixture,Path(tmp))
                rows.append({'clip':row['id'],'method':'demucs_vocals','si_sdri_db':score(target,voice,mixture),'latency_seconds':round(elapsed,3),'note':'Music-vocal model evaluated on speech; not a narration specialist.'})
            except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
                rows.append({'clip':row['id'],'method':'demucs_vocals','error':str(exc)[-800:]})
        clips+=1
        print(f'finished {clips}/{args.clips}',flush=True)
        if clips>=args.clips: break
    if clips<args.clips: print(f'Only {clips} recordings long enough',flush=True)
    report={'method':'real clean speech + real music mixed at equal RMS','speech_dataset':'hf-internal-testing/librispeech_asr_dummy validation',
            'music_source':str(Path(args.music).name),'duration_seconds':args.seconds,'clips':clips,'results':rows,
            'caveat':'Small constructed mixtures, one LibriSpeech subset and one music recording; not a substitute for diverse real-video listening tests.'}
    destination=Path(args.out);destination.mkdir(parents=True,exist_ok=True)
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('report',destination/'report.json',flush=True)
    return 0 if clips and any('si_sdri_db' in row for row in rows) else 1


if __name__=='__main__': raise SystemExit(main())
