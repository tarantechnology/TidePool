"""Label-free, manifest-driven evaluation on licensed existing recordings.

A manifest supplies clean target/background files, attribution/license, category, split,
seed and optional placement intervals. Clean sources are references, never new labels.
No profile is promoted by this script; it emits a numerical gate plus listening tasks.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path
import httpx
import numpy as np
from sam_audio_service.audio_io import decode, encode_wav_float32
from sam_audio_service.metrics import si_sdr_improvement

CATEGORIES={'speech_music','speech_noise','intermittent','competing_voices','stereo'}
PROFILES={
    'text':{'chunk_seconds':10,'chunk_batch':2,'reranking_candidates':1},
    'anchor':{'chunk_seconds':10,'chunk_batch':2,'reranking_candidates':1},
    'judge':{'chunk_seconds':10,'chunk_batch':2,'reranking_candidates':2},
    'long':{'chunk_seconds':20,'chunk_batch':1,'reranking_candidates':2},
    'stereo':{'chunk_seconds':10,'chunk_batch':2,'reranking_candidates':1,'stereo_mode':'per_channel'},
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest,root):
    seen={}; ids=set()
    for item in manifest['examples']:
        if item['id'] in ids: raise ValueError('Example IDs must be unique')
        ids.add(item['id'])
        if item['category'] not in CATEGORIES or item['split'] not in ('tuning','heldout'): raise ValueError('Unknown category or split')
        if not isinstance(item['seed'],int): raise ValueError('Fixed integer seed required')
        for key in ('target','background'):
            source=item[key]
            if not all(source.get(k) for k in ('file','license','attribution','url','sha256')): raise ValueError('Source provenance is required')
            path=root/source['file']
            if sha(path)!=source['sha256']: raise ValueError('Source hash differs from manifest')
            if source['sha256'] in seen and seen[source['sha256']]!=item['split']: raise ValueError('Tuning and heldout recordings must be disjoint')
            seen[source['sha256']]=item['split']


def mixture(item,root):
    rate=48000; count=round(item.get('seconds',12)*rate); rng=np.random.default_rng(item['seed'])
    tracks=[]
    for key in ('target','background'):
        raw=decode(path=root/item[key]['file']).samples
        if raw.shape[1]<count: raw=np.tile(raw,(1,math.ceil(count/raw.shape[1])))
        offset=int(rng.integers(0,raw.shape[1]-count+1));raw=raw[:,offset:offset+count]
        if item['category']=='stereo':
            mono=raw.mean(axis=0); gains=[1,.25] if key=='target' else [.2,1]
            raw=np.stack([mono*g for g in gains])
        else: raw=raw.mean(axis=0,keepdims=True)
        if key=='target' and item.get('placements'):
            envelope=np.zeros(count,np.float32)
            for start,end in item['placements']:
                lo,hi=round(start*rate),round(end*rate)
                if not 0<=lo<hi<=count: raise ValueError('Invalid placement')
                envelope[lo:hi]=1
                ramp=min(240,(hi-lo)//2);envelope[lo:lo+ramp]=np.linspace(0,1,ramp);envelope[hi-ramp:hi]=np.linspace(1,0,ramp)
            raw=raw*envelope
        rms=float(np.sqrt(np.mean(raw.astype(np.float64)**2)))
        if rms<1e-8: raise ValueError('Silent reference')
        tracks.append(raw/rms)
    target,background=tracks;scale=.85/float(np.max(np.abs(target+background)))
    return (target*scale).astype(np.float32),(background*scale).astype(np.float32)


def metrics(target,residual,reference,background,mix,chunk):
    if target.shape!=mix.shape or residual.shape!=mix.shape: raise ValueError('Output shape mismatch')
    # Per-channel metrics prevent mono averaging from hiding stereo damage.
    target_score=np.mean([si_sdr_improvement(a,b,c) for a,b,c in zip(target,reference,mix)])
    background_score=np.mean([si_sdr_improvement(a,b,c) for a,b,c in zip(residual,background,mix)])
    error=(target-reference).astype(np.float64); windows=[]
    for boundary in np.arange(chunk-2,mix.shape[1]/48000,chunk-2):
        i=round(boundary*48000);windows.append(error[:,max(0,i-480):min(mix.shape[1],i+480)])
    return {'target_si_sdri_db':float(target_score),'background_si_sdri_db':float(background_score),
            'reconstruction_max_error':float(np.max(np.abs(target+residual-mix))),
            'boundary_error_rms':float(np.sqrt(np.mean(np.concatenate(windows,axis=1)**2))) if windows else None}


def gate(rows):
    decisions={}
    for profile in PROFILES:
        if profile=='text':continue
        categories={}
        for category in sorted(CATEGORIES):
            baseline={r['id']:r for r in rows if r['profile']=='text' and r['category']==category and r['split']=='heldout'}
            candidate={r['id']:r for r in rows if r['profile']==profile and r['category']==category and r['split']=='heldout'}
            complete=bool(baseline) and baseline.keys()==candidate.keys() and all('error' not in r for r in [*baseline.values(),*candidate.values()])
            deltas={}
            if complete:
                for key in ('target_si_sdri_db','background_si_sdri_db'):
                    values=[candidate[i][key]-baseline[i][key] for i in baseline]
                    deltas[key]=float(np.median(values)) if all(np.isfinite(values)) else None
            categories[category]={'paired_examples':len(candidate),'median_delta_db':deltas,
                                  'passes':complete and all(deltas.get(k) is not None and deltas[k]>=-.5 for k in ('target_si_sdri_db','background_si_sdri_db'))}
        decisions[profile]={'numerical_gate_passes':all(c['passes'] for c in categories.values()),'categories':categories,
                            'promoted':False,'pending':'Listening review, memory/runtime checks and adequate held-out coverage are required.'}
    return decisions


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--url',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES));args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());root=args.manifest.parent;validate_manifest(manifest,root);args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in manifest['examples']:
        reference,background=mixture(item,root);mix=reference+background
        directory=args.out/item['id'];directory.mkdir(exist_ok=True)
        for name,samples in [('mixture',mix),('reference',reference),('background',background)]: (directory/(name+'.wav')).write_bytes(encode_wav_float32(samples,48000))
        for profile in args.profiles:
            row={k:item[k] for k in ('id','category','split','seed')};row['profile']=profile
            options={**PROFILES[profile],'seed':item['seed'],'predict_spans':False,'output_format':'pcm_f32le','residual':'mixture_minus_targets'}
            metadata={'prompts':[item['prompt']],'options':options}
            if profile=='anchor': metadata['anchors']=item.get('anchors',[])
            started=time.monotonic()
            try:
                response=httpx.post(args.url,files={'audio':('mixture.wav',encode_wav_float32(mix,48000),'audio/wav')},data={'metadata':json.dumps(metadata)},timeout=600);response.raise_for_status();body=response.json()
                stems={s['role']:base64.b64decode(s['audio_base64']) for s in body['stems']}
                target=decode(data=stems['target']).samples;residual=decode(data=stems['residual']).samples
                row.update(metrics(target,residual,reference,background,mix,options['chunk_seconds']))
                row.update(model=body['model'],warnings=body.get('warnings',[]),peak_memory_bytes=body['model'].get('peak_memory_bytes'))
                for role,raw in stems.items(): (directory/(profile+'-'+role+'.wav')).write_bytes(raw)
            except Exception as exc: row['error']=str(exc)[:500]
            row['seconds']=round(time.monotonic()-started,3);row['requested_options']=options;rows.append(row)
            # Checkpoint every request, including errors; no failed samples disappear.
            report={'manifest':manifest,'results':rows,'promotion_gate':gate(rows),'listening_review_completed':False}
            def clean(value):
                if isinstance(value,float) and not math.isfinite(value):return None
                if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
                if isinstance(value,list):return [clean(v) for v in value]
                return value
            (args.out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:row.get(k) for k in ('id','profile','target_si_sdri_db','seconds','error')}),flush=True)
    return int(any('error' in r for r in rows))

if __name__=='__main__':raise SystemExit(main())
