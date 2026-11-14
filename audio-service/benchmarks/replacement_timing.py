"""Small, reproducible repeated-event smoke benchmark using existing editor stems.

Run from the repository root with PYTHONPATH=.:audio-service. This creates short
synthetic mixtures; insertion times are known, but the cropped source may not be a
complete natural event. Results do not replace listening or real-clip annotation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from app.media import read_wav,write_wav
from app.event_detection import onset_features,find_similar_events,decode_audio
from app.replacement import transient_peaks


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def score(reference, estimated, tolerance=.15):
    unused=list(estimated);matched=[]
    for t in reference:
        if not unused:break
        i=min(range(len(unused)),key=lambda j:abs(unused[j]-t))
        if abs(unused[i]-t)<=tolerance:matched.append(abs(unused.pop(i)-t))
    tp=len(matched);return {'matched':tp,'expected':len(reference),'false_positives':len(unused),
                            'median_error_ms':round(float(np.median(matched))*1000,1) if matched else None}


def run(root, output, flex_root=None, checkpoint=None):
    manifest=json.loads((root/'render-manifest.json').read_text());reports=[];model=None
    if flex_root:
        import os,torch
        sys.path.insert(0,str(flex_root));os.chdir(flex_root)
        from api import FlexSED
        model=FlexSED(ckpt_path=str(checkpoint),device='cuda' if torch.cuda.is_available() else 'cpu')
    np.random.default_rng(20261114)  # Document the fixed evaluation seed.
    with tempfile.TemporaryDirectory(prefix='sc-timing-bench-') as tmp:
        for item in manifest['cases']:
            directory=root/item['id'];target=directory/'target.wav';background=directory/'background.wav'
            if not target.exists() or not background.exists():continue
            a,sr=read_wav(target);bg,br=read_wav(background)
            if sr!=br or len(bg)<8*sr:continue
            # Use the highest-energy 250 ms passage; this is a synthetic proxy for an event.
            mono=np.mean(a*a,axis=1);win=round(.25*sr);cumulative=np.concatenate(([0.0],np.cumsum(mono)));energy=((cumulative[win:]-cumulative[:-win])/win)[::round(.05*sr)]
            start=int(np.argmax(energy))*round(.05*sr);sample=a[start:start+win].copy()
            if len(sample)!=win or np.max(np.abs(sample))<1e-5:continue
            sample*=.5/max(.5,float(np.max(np.abs(sample))))
            mixed=bg[:8*sr].copy()*.5;target_mix=np.zeros_like(mixed);positions=[1.0,3.5,6.0]
            for position in positions:
                at=round(position*sr);target_mix[at:at+win]+=sample
            mixed+=target_mix
            wav=Path(tmp)/(item['id']+'.wav');write_wav(wav,mixed,sr)
            expected=[p+int(np.argmax(np.max(np.abs(sample),axis=1)))/sr for p in positions]
            start_time=time.monotonic();energy_peaks,_=transient_peaks(mixed,sr,0,8)
            baseline=score(expected,[p/sr for p in energy_peaks]);baseline['seconds']=round(time.monotonic()-start_time,3)
            start_time=time.monotonic();times,_=onset_features(decode_audio(wav,8))
            onset=score(expected,times.tolist());onset['seconds']=round(time.monotonic()-start_time,3)
            try:
                start_time=time.monotonic();similar=find_similar_events(wav,8,expected[0]);example=score(expected,similar['times']);example['seconds']=round(time.monotonic()-start_time,3)
            except ValueError as exc:example={'error':str(exc)}
            result={'id':item['id'],'query':item['prompt'],'split':item.get('split'),'source_hashes':{'target':sha(target),'background':sha(background)},'reference_times':expected,'energy_baseline':baseline,'spectral_onsets':onset,'example_matching':example}
            if model:
                import torch
                start_time=time.monotonic()
                with torch.inference_mode():pred=model.run_inference(str(wav),[item['prompt']])
                probs=pred.squeeze().cpu().numpy().reshape(-1);indices=np.flatnonzero(probs>=.5);segments=[]
                for idx in indices:
                    t=idx/25
                    if not segments or t-segments[-1][-1]>.05:segments.append([t])
                    else:segments[-1].append(t)
                # Regions are compared to each known insertion; durations are coarse.
                hits=sum(any(seg[0]-.15<=t<=seg[-1]+.15 for seg in segments) for t in expected)
                result['flexsed']={'expected':len(expected),'covered':hits,'regions':len(segments),'max_score':round(float(probs.max()),3),'seconds':round(time.monotonic()-start_time,3)}
            reports.append(result);print(json.dumps(result),flush=True)
    output.write_text(json.dumps({'seed':20261114,'caveat':'Synthetic cropped bursts; no human labels, no claim of natural-event detection quality.','cases':reports},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--flexsed-root',type=Path);p.add_argument('--checkpoint',type=Path)
    args=p.parse_args();run(args.root,args.output,args.flexsed_root,args.checkpoint)
