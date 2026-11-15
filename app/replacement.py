"""Sound effect assets and bounded, revision-checked replacement previews."""
import json
import os
import shutil
import subprocess
from pathlib import Path
from . import candidates
from .core import probe, project_dir, run, uid
from .guidance import options
from .media import export
from .visual import file_sha256


class NoUsableSeparation(ValueError):
    def __init__(self,message,recovery):
        super().__init__(message);self.recovery=recovery


def asset_path(pid, aid):
    if len(aid)!=32 or any(c not in '0123456789abcdef' for c in aid): raise ValueError('Invalid sound effect ID.')
    return candidates.safe_path(pid,project_dir(pid)/'effects'/(aid+'.wav'))


def generate(project, payload):
    executable=os.getenv('SOUNDCANVAS_SFX_CLI','').strip()
    if not executable or not Path(executable).is_file(): raise ValueError('Sound generation is not configured. Upload an effect or configure Stable Audio.')
    directory=project_dir(project['id'])/'effects';directory.mkdir(parents=True,exist_ok=True)
    aid=uid();out=directory/(aid+'.wav')
    try:
        env=os.environ.copy()
        # The generator has its own environment and may use a different Python version.
        # Inheriting the editor's PYTHONPATH can load incompatible binary modules.
        env.pop('PYTHONPATH',None)
        subprocess.run([executable,'--model','small-sfx','-p',payload['prompt'],'--duration','3','-o',str(out)],check=True,capture_output=True,text=True,timeout=900,env=env)
        info=probe(out)
        if not 0<info['duration']<=15: raise ValueError('Generated sound duration is invalid.')
        return {'asset_id':aid,'duration':info['duration'],'source':'generated','model':'stable-audio-3-small-sfx'}
    except subprocess.TimeoutExpired as e:
        out.unlink(missing_ok=True);raise ValueError('Sound generation timed out. Try again or upload an effect.') from e
    except (OSError,subprocess.CalledProcessError,ValueError) as e:
        out.unlink(missing_ok=True);raise ValueError('Sound generation failed. Check Stable Audio model access and configuration, or upload an effect.') from e



def transient_peaks(samples, rate, start, end):
    """Find separated attacks using short stereo energy windows, not raw waveform cycles.

    Return sample positions and whether the single-peak fallback was necessary. This
    cannot determine semantic identity; the separator and chosen interval supply that.
    """
    import numpy as np
    if samples.ndim==1: samples=samples[:,None]
    left=max(0,round(start*rate));right=min(len(samples),round(end*rate))
    if right<=left: return [],False
    segment=samples[left:right];frame=max(1,round(rate*.01));n=(len(segment)+frame-1)//frame
    padded=np.pad(segment,((0,n*frame-len(segment)),(0,0)))
    envelope=np.sqrt(np.mean(np.square(padded.reshape(n,frame,-1)),axis=(1,2)))
    smooth=np.convolve(envelope,[.25,.5,.25],mode='same') if n>1 else envelope
    maximum=float(np.max(smooth)) if len(smooth) else 0
    if maximum<1e-5: return [],False
    floor=float(np.percentile(smooth,30));threshold=max(1e-5,maximum*.03,floor*1.5)
    radius=10; candidates_at=[]
    for i in range(n):
        prev=smooth[i-1] if i>0 else 0;following=smooth[i+1] if i+1<n else 0
        if smooth[i]<threshold or smooth[i]<prev or smooth[i]<=following: continue
        lo=max(0,i-radius);hi=min(n,i+radius+1)
        before=float(np.min(smooth[lo:i+1]));after=float(np.min(smooth[i:hi]))
        prominence=float(smooth[i])-max(before,after)
        if prominence>=max(maximum*.035,floor*.12,1e-5): candidates_at.append(i)
    # Keep the strongest peak within 90 ms to avoid counting one sound's oscillations.
    chosen=[];min_gap=max(1,round(.09*rate/frame))
    for i in sorted(candidates_at,key=lambda k:float(smooth[k]),reverse=True):
        if all(abs(i-j)>=min_gap for j in chosen):chosen.append(i)
    fallback=not chosen
    if fallback: chosen=[int(np.argmax(smooth))]
    positions=[]
    for i in sorted(chosen):
        a=max(0,(i-1)*frame);b=min(len(segment),(i+2)*frame)
        # Align to the real waveform peak around the smoothed energy maximum.
        positions.append(left+a+int(np.argmax(np.max(np.abs(segment[a:b]),axis=1))))
    return positions,fallback


def effect_peak(path, rate):
    """Find the actual strongest sample after final resampling and fades."""
    import numpy as np
    from .media import read_wav
    samples,actual=read_wav(path)
    if actual!=rate or not len(samples):raise ValueError('Sound effect audio is unavailable.')
    amplitude=np.max(np.abs(samples),axis=1)
    if float(np.max(amplitude))<1e-5:raise ValueError('The replacement sound is silent. Choose or generate another sound.')
    return int(np.argmax(amplitude))


def onset_before_peak(samples, rate, peak):
    """Estimate the attack preceding a detected peak, bounded to 200 ms."""
    import numpy as np
    if samples.ndim==1: samples=samples[:,None]
    frame=max(1,round(rate*.005));peak=max(0,min(int(peak),len(samples)-1))
    left=max(0,peak-round(rate*.2)); segment=samples[left:peak+1]
    n=(len(segment)+frame-1)//frame
    padded=np.pad(segment,((n*frame-len(segment),0),(0,0)))
    env=np.sqrt(np.mean(np.square(padded.reshape(n,frame,-1)),axis=(1,2)))
    threshold=max(1e-5,float(np.max(env))*.12)
    active=np.flatnonzero(env>=threshold)
    if not len(active):return peak
    # Walk back from the peak, stopping at an extended quiet gap.
    last=int(active[-1]);first=last
    for i in active[-2::-1]:
        if first-int(i)>4:break
        first=int(i)
    return max(left,left+first*frame-(n*frame-len(segment)))


def duck_events(raw_path, output_path, events, group, rate, source_offset=0):
    """Attenuate only approved events, with short fades; never modify the raw stem."""
    import numpy as np
    from .media import read_wav,write_wav
    samples,sr=read_wav(raw_path)
    if sr!=rate: raise ValueError('Separated sound timing changed.')
    for event in events:
        if event.get('group')!=group or not event.get('enabled',True):continue
        center=round((event['source_peak']-source_offset)*rate);a=max(0,center-round(.035*rate));b=min(len(samples),center+round(.18*rate))
        if b<=a:continue
        fade=max(1,round(.01*rate)); length=b-a
        gain=np.zeros(length,dtype=np.float64);edge=min(fade,length//2)
        if edge:gain[:edge]=np.linspace(1,0,edge,endpoint=False);gain[-edge:]=np.linspace(0,1,edge,endpoint=True)
        samples[a:b]*=gain[:,None]
    write_wav(output_path,samples,rate)


def render_effect(normalized, output, events, rate, channels, count, alignment):
    """Render event copies at their independent anchor and offset."""
    peak_offset=effect_peak(normalized,rate)
    from .media import read_wav
    samples,sr=read_wav(normalized)
    onset_offset=onset_before_peak(samples,rate,peak_offset)
    effect_length=len(samples);offset=onset_offset if alignment=='onset' else peak_offset
    active=[e for e in events if e.get('enabled',True)]; points=[]
    for event in active:
        anchor=event['source_onset'] if alignment=='onset' else event['source_peak']
        point=round((anchor+event.get('offset',0))*rate)
        if not 0<=point<count:raise ValueError('A placement falls outside the audio timeline.')
        points.append(point)
    warnings=[]
    if not points:
        run('ffmpeg','-v','error','-y','-f','lavfi','-i',f'anullsrc=r={rate}:cl='+('mono' if channels==1 else 'stereo'),'-t',str(count/rate),'-c:a','pcm_f32le',str(output))
        return warnings,peak_offset,onset_offset
    ordered=sorted(points)
    if any(b-a<effect_length for a,b in zip(ordered,ordered[1:])): warnings.append('Some effects overlap and may sound louder; adjust the effect layer if needed.')
    if any(point-offset+effect_length>count for point in points): warnings.append('An effect extends past the end of the clip and its tail was trimmed.')
    filters=[f'[0:a]asplit={len(points)}'+''.join(f'[s{i}]' for i in range(len(points)))];labels=[];clipped=False
    for i,point in enumerate(points):
        shift=point-offset;trim=max(0,-shift);delay=max(0,shift)
        clipped=clipped or bool(trim)
        filters.append(f'[s{i}]atrim=start_sample={trim},asetpts=PTS-STARTPTS,adelay={delay}S:all=1[a{i}]');labels.append(f'[a{i}]')
    filters.append(''.join(labels)+f'amix=inputs={len(labels)}:duration=longest:normalize=0,apad,atrim=end_sample={count}[mix]')
    run('ffmpeg','-v','error','-y','-i',str(normalized),'-filter_complex',';'.join(filters),'-map','[mix]','-ac',str(channels),'-ar',str(rate),'-c:a','pcm_f32le',str(output))
    if clipped:warnings.append('An effect started before the clip and its beginning was trimmed to align it.')
    return warnings,peak_offset,onset_offset


def effect_layers(pid, label, normalized, events, rate, count, alignment, solo=False, group_id=None, asset_id=None):
    """Persist one short clip per event so placements stay editable after apply."""
    from .media import read_wav,write_wav
    samples,sr=read_wav(normalized)
    if sr!=rate:raise ValueError('Sound effect timing changed.')
    peak=effect_peak(normalized,rate);anchor_offset=onset_before_peak(samples,rate,peak) if alignment=='onset' else peak
    directory=project_dir(pid)/'stems';rows=[];created=[]
    try:
        for index,event in enumerate(events):
            if not event.get('enabled',True):continue
            anchor=event['source_onset'] if alignment=='onset' else event['source_peak']
            point=round((anchor+event.get('offset',0))*rate);shift=point-anchor_offset
            trim=max(0,-shift);position=max(0,shift);end=min(len(samples),trim+count-position)
            if not 0<=position<count or end<=trim:raise ValueError('A placement falls outside the audio timeline.')
            path=directory/(uid()+'-effect-clip.wav');created.append(path);write_wav(path,samples[trim:end],rate)
            name=label[:100] if index==0 else f'{label[:90]} {index+1}'
            rows.append({'id':event['id'],'project_id':pid,'name':name,'label':name,'path':str(path),'duration':(end-trim)/rate,'start_offset':position/rate,'volume':1,'muted':0,'solo':int(solo),'automation':'[]','mute_ranges':'[]','revision':0,'role':'effect','effect_group_id':group_id,'effect_asset_id':asset_id,'effect_label':label})
        return rows
    except Exception:
        for path in created:path.unlink(missing_ok=True)
        raise

def build(project,payload,cid):
    pid=project['id'];snapshot=payload['_candidate'];source=candidates.safe_path(pid,snapshot['source']);rows=[dict(r) for r in snapshot['rows']]
    if file_sha256(source)!=snapshot['source_sha256']: raise ValueError('Source changed; generate another preview.')
    for row in rows:
        if file_sha256(candidates.safe_path(pid,row['path']))!=snapshot['stem_hashes'][row['id']]: raise ValueError('A layer changed; generate another preview.')
    effect=asset_path(pid,payload['asset_id'])
    if file_sha256(effect)!=payload['asset_sha256']: raise ValueError('Sound effect changed; upload or generate it again.')
    directory=project_dir(pid)/'stems';directory.mkdir(parents=True,exist_ok=True)
    original=directory/(uid()+'.wav');mix=directory/(uid()+'.wav');placed=directory/(uid()+'.wav')
    if rows: export(rows,source,original,'wav',False)
    else: shutil.copyfile(source,original)
    warnings=[];is_mock=False;peak_samples=[];events=[];groups=[];rate=probe(source)['sample_rate'];sound_group_id=uid()
    if payload['mode']=='replace':
        for span in payload['intervals']:
            selected=next((r for r in rows if r['id']==payload.get('source_stem_id')),None)
            residuals=[r for r in rows if r['name'].lower()=='residual']
            if payload.get('source_stem_id') and not selected:raise ValueError('The selected source layer is unavailable.')
            if selected:
                event_source=candidates.safe_path(pid,selected['path']);rid=selected['id'];current=rows;source_offset=float(selected['start_offset'])
            elif rows and len(residuals)==1 and residuals[0]['start_offset']==0:
                residual=residuals[0];event_source=candidates.safe_path(pid,residual['path']);rid=residual['id'];current=rows
                source_offset=0.0
            elif not rows:
                event_source=source;rid=None;current=[];source_offset=0.0
            else: raise ValueError('Replacement needs one unshifted residual layer. Choose add over the original or separate a fresh project.')
            local_span={'start':span['start']-source_offset,'end':span['end']-source_offset}
            if local_span['start']<0 or local_span['end']>probe(event_source)['duration']+1/48000:raise ValueError('Select times within the source layer.')
            entry={'_candidate':{'rows':current,'source':str(event_source),'source_sha256':file_sha256(event_source),'project_audio':project['audio'],'target_id':None,'residual_id':rid},
                   '_processing_interval':local_span,'_processing_time':None,'prompts':[payload['source_prompt']],
                   'anchors':[],'visual_prompt':{},'edit_scope':{'action':'keep'},'profile':'preview','options':options('preview')}
            if selected:entry['_candidate'].update(source_layer_id=selected['id'],source_offset=source_offset)
            if selected:entry['options']['mode']='prompted'
            result=candidates.build(project,entry,cid);next_rows=result['proposed']
            inference=result.get('inference') or {}
            mock=(inference.get('model') or {}).get('is_mock') is True
            if not mock:
                from .director import measured_ratio
                ratio,correlation=measured_ratio(event_source,result['target']['path'],local_span)
                if ratio<.02 or (ratio>.8 and abs(correlation)>.95):
                    warnings.append('Skipped a sound: its separation was too quiet or too similar to the whole mix.')
                    continue
            from .media import read_wav
            target,target_rate=read_wav(result['target']['path'])
            peaks,fallback=transient_peaks(target,target_rate,local_span['start'],local_span['end'])
            if not peaks:
                warnings.append('Skipped a sound because the separated target was silent.')
                continue
            if fallback: warnings.append('A selected range had no distinct transient; one strongest point was used. Check the timing in the preview.')
            hints=[t for t in (payload.get('event_hints') or []) if span['start']<=t<=span['end']]
            if hints:
                matched=[]
                for hint in hints:
                    available=[p for p in peaks if p not in matched and abs(p/target_rate+source_offset-hint)<=.16]
                    if available:matched.append(min(available,key=lambda p:abs(p/target_rate+source_offset-hint)))
                if len(matched)<len(hints):warnings.append('Some suggested sounds could not be verified in the separated target and were skipped.')
                peaks=sorted(matched)
            if not peaks:
                warnings.append('Skipped a range because the suggested sound was not clear in the separated audio.')
                continue
            peak_samples.extend(round((p/target_rate+source_offset)*rate) for p in peaks)
            if len(peak_samples)>100: raise ValueError('More than 100 peaks were detected. Try a shorter selection.')
            group=len(groups);raw_path=result['target']['path'];target_id=result['target']['id']
            groups.append({'target_id':target_id,'raw_path':raw_path,'interval':span,'source_offset':source_offset,'source_rate':target_rate})
            for point in peaks:
                events.append({'id':uid(),'group':group,'source_peak':point/target_rate+source_offset,'source_onset':onset_before_peak(target,target_rate,point)/target_rate+source_offset,'offset':0,'enabled':True})
            processed=directory/(uid()+'-ducked.wav');duck_events(raw_path,processed,events,group,target_rate,source_offset)
            next(row for row in next_rows if row['id']==target_id).update(path=str(processed),sound_group_id=sound_group_id,sound_group_label=payload['source_prompt'],sound_start=local_span['start'],sound_end=local_span['end'])
            rows=next_rows
            is_mock=is_mock or mock
            warnings.extend(inference.get('warnings') or [])
        if not events:
            hints=payload.get('event_hints') or [span['start'] for span in payload['intervals']]
            end=probe(source)['duration']
            recovery={'asset_id':payload['asset_id'],'effect_label':payload['effect_label'],'source_prompt':payload['source_prompt'],'mode':'add','intervals':[{'start':t,'end':min(end,t+.01)} for t in hints if 0<=t<end]}
            raise NoUsableSeparation('No usable separated sound was found; it may be silent or contain too much background. You can preview adding the effect without removing the original.',recovery)
        warnings.append('Original sound reduction and event detection are approximate. Listen for missed sounds, extra effects, or damage to other audio.')
    if not rows:
        rows=[{'id':uid(),'project_id':pid,'name':'Original audio','path':str(source),'duration':probe(source)['duration'],'start_offset':0,'volume':1,'muted':0,'solo':0,'automation':'[]','mute_ranges':'[]','revision':0}]
    duration=probe(source)['duration'];channels=probe(source)['channels'];count=round(duration*rate)
    if payload['mode']=='add':
        peak_samples=[round(span['start']*rate) for span in payload['intervals']]
        events=[{'id':uid(),'group':None,'source_peak':p/rate,'source_onset':p/rate,'offset':0,'enabled':True} for p in peak_samples]
    normalized=directory/(uid()+'-effect.wav')
    run('ffmpeg','-v','error','-y','-i',str(effect),'-ar',str(rate),'-ac',str(channels),'-af','afade=t=in:st=0:d=0.005','-c:a','pcm_f32le',str(normalized))
    alignment=payload.get('alignment','peak')
    if alignment not in ('peak','onset'): raise ValueError('Unsupported alignment.')
    notices,peak_offset,onset_offset=render_effect(normalized,placed,events,rate,channels,count,alignment)
    warnings.extend(notices)
    effect_group_id=uid()
    rows.extend(effect_layers(pid,payload['effect_label'],normalized,events,rate,count,alignment,any(row['solo'] for row in rows),effect_group_id,payload['asset_id']))
    export(rows,source,mix,'wav',False)
    return {'_replacement_edit':{'effect_path':str(normalized),'effect_ids':[e['id'] for e in events],'effect_label':payload['effect_label'],'effect_group_id':effect_group_id,'effect_asset_id':payload['asset_id'],'groups':groups,'rate':rate,'channels':channels,'count':count},'target':{'path':str(placed)},'mix':{'path':str(mix)},'original':{'path':str(original)},'snapshot':snapshot,'proposed':rows,
            'inference':{'model':{'backend':'sound-replacement','is_mock':is_mock},'warnings':warnings},
            'provenance':{'kind':'replacement','mode':payload['mode'],'intervals':payload['intervals'],'effect_label':payload['effect_label'],'source_prompt':payload['source_prompt'],'source_layer_id':payload.get('source_stem_id'),'peak_times':[round(x/rate,6) for x in peak_samples], 'events':events,'alignment':alignment,'effect_peak_offset':round(peak_offset/rate,6),'effect_onset_offset':round(onset_offset/rate,6)}}


def effect_request(text):
    """Keep count explicit; add only grounded details to the sound description."""
    import re
    text=' '.join(text.strip().split())
    if re.search(r'\b(?:five|six|seven|eight|nine|ten|[5-9]|\d{2,})\s+quacks?\b',text,re.I): raise ValueError('Please request one to four quacks per replacement.')
    number=re.search(r'\b(one|single|two|three|four|1|2|3|4)\b',text,re.I)
    counts={'one':1,'single':1,'two':2,'three':3,'four':4,'1':1,'2':2,'3':3,'4':4}
    count=counts[number.group(1).lower()] if number else 1
    name=re.sub(r'\b(?:exactly|one|single|two|three|four|1|2|3|4)\b','',text,flags=re.I).strip(' ,')
    if re.search(r'\bquacks?\b',name,re.I):
        single_name=re.sub(r'\bquacks?(?:\s+sounds?)?\b','quack',name,flags=re.I)
        description=f'One single short sound: {single_name}. A clearly audible duck quack, with silence before and after. No speech, no music, no repeated quacks.'
        return {'prompt':description,'count':count,'kind':'quack'}
    if count==1:
        name=re.sub(r'\b([a-z]{3,}?)(?<!s)s\b$',r'\1',name,flags=re.I)
    return {'prompt':f'Exactly {count} distinct occurrence(s) of this isolated sound effect: {name}. Silence before and after. No speech or music.', 'count':count,'kind':'generic'}


def isolate_one_quack(path):
    """Keep the first audible burst; thresholding is a heuristic, not semantic proof."""
    import numpy as np
    from .media import read_wav
    samples,rate=read_wav(path); mono=np.max(np.abs(samples),axis=1) if samples.ndim==2 else np.abs(samples)
    window=max(1,round(rate*.01)); n=len(mono)//window
    if not n: raise ValueError('Generated effect was empty; try again.')
    peaks=np.max(mono[:n*window].reshape(n,window),axis=1);maximum=float(np.max(peaks))
    if maximum<1e-4: raise ValueError('Generated effect was silent; try again.')
    active=np.flatnonzero(peaks>max(maximum*.08,1e-4));start=max(0,int(active[0])-2)
    last=int(active[0]);end=min(n,last+150)
    for i in active[1:]:
        if i-last>18 or i>=start+150: end=min(n,last+5);break
        last=int(i)
    else: end=min(n,last+5)
    return start*.01,max((start+1)*.01,end*.01)


def prepare_effect(project,payload):
    """Generate one sound and make requested repeated quacks by explicit copies."""
    spec=effect_request(payload['prompt']);result=generate(project,{'prompt':spec['prompt']})
    if spec['kind']!='quack':
        result.update({'requested_count':spec['count'],'count_verified':False,'generation_prompt':spec['prompt']})
        return result
    path=asset_path(project['id'],result['asset_id']);directory=path.parent
    single=directory/(uid()+'.wav');out=directory/(uid()+'.wav')
    try:
        start,end=isolate_one_quack(path)
        run('ffmpeg','-v','error','-y','-ss',str(start),'-i',str(path),'-t',str(end-start),'-af',f'afade=t=in:st=0:d=0.005,afade=t=out:st={max(0,end-start-.01)}:d=0.01','-c:a','pcm_f32le',str(single))
        duration=probe(single)['duration']; inputs=[];filters=[]
        for i in range(spec['count']):
            inputs.extend(['-i',str(single)]);filters.append(f'[{i}:a]adelay={round(i*(duration+.25)*1000)}:all=1[a{i}]')
        filters.append(''.join(f'[a{i}]' for i in range(spec['count']))+f'amix=inputs={spec["count"]}:duration=longest:normalize=0[mix]')
        run('ffmpeg','-v','error','-y',*inputs,'-filter_complex',';'.join(filters),'-map','[mix]','-c:a','pcm_f32le',str(out))
        out.replace(path);result['duration']=probe(path)['duration']
        result.update({'requested_count':spec['count'],'count_verified':False,'generation_prompt':spec['prompt']})
        return result
    except Exception:
        path.unlink(missing_ok=True);raise
    finally:
        single.unlink(missing_ok=True);out.unlink(missing_ok=True)


def parse_replacement(text,duration):
    import re
    text=' '.join(text.strip().split())
    prefix=re.match(r"^(?:i\s+(?:would\s+like|want|need)|i['’]d\s+like)\s+(?:you\s+)?to\s+",text,re.I)
    if prefix: text=text[prefix.end():]
    # A trailing example clause describes the source, not the generated effect.
    # Normalize only this explicit form so additional commands still require clarification.
    example=re.search(r'\s*[;,]\s*(?:use|using)\s+(?:the|that)\s+sound\s+((?:at|around|near)\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|s))\s+as\s+(?:an?\s+)?example[.!?]?\s*$',text,re.I)
    if example:
        text=text[:example.start()].rstrip()+' '+example.group(1)
    spoken=re.fullmatch(r'''(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?add\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(?:after|when|whenever)\s+(?:(?:the|every|each)\s+)?(?:word\s+)?["“']?([\w]+(?:['’][\w]+)*)["”']?(?:\s+(?:is|was|has\s+been)\s+(?:said|spoken|mentioned))?[.!?]?''',text,re.I)
    if spoken:
        effect,word=spoken.group(1).strip(),spoken.group(2).strip()
        if not effect or len(effect)>100 or len(word)>60: raise ValueError('What sound and spoken word should I use?')
        effect_request(effect)
        return {'source':'','effect':effect,'time':None,'mode':'add','spoken_word':word}
    addition=re.fullmatch(r'(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?add\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+at\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)[.!?]?',text,re.I)
    if addition:
        effect=addition.group(1).strip();point=float(addition.group(2))
        if point>=duration: raise ValueError('That timestamp is outside the video.')
        if not effect or len(effect)>100: raise ValueError('What sound should I add?')
        effect_request(effect)
        return {'source':'','effect':effect,'time':point,'mode':'add'}
    match=re.fullmatch(r'(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:replace|swap|substitute)\s+(?:(?:(?:every|all)(?:\s+of)?(?:\s+the)?|the)\s+)?(.+?)\s+(?:with|for)\s+(?:a\s+|an\s+|the\s+)?(.+?)[.!?]?',text,re.I)
    if not match:
        match=re.fullmatch(r'(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?make\s+(?:(?:every|all(?:\s+the)?|the)\s+)?(.+?)\s+sound\s+like\s+(?:a\s+|an\s+|the\s+)?(.+?)[.!?]?',text,re.I)
    if not match: raise ValueError('What sound should I replace, and what should I replace it with?')
    source,effect=match.group(1).strip(),match.group(2).strip()
    times=re.findall(r'\b(?:at|around|near)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b',text,re.I)
    if times:
        if len(times)!=1: raise ValueError('Please give one timestamp or ask to replace the sound throughout the video.')
        effect=re.sub(r'\s+\b(?:at|around|near)\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b\s*$','',effect,flags=re.I).strip()
        source=re.sub(r'\s+\b(?:at|around|near)\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b\s*$','',source,flags=re.I).strip()
        if re.search(r'\b(?:at|around|near)\s+\d',source+' '+effect,re.I): raise ValueError('Please describe one replacement and one timestamp at a time.')
        point=float(times[0])
        if point>=duration: raise ValueError('That timestamp is outside the video.')
    else: point=None
    if not source or not effect or len(source)>200 or len(effect)>100: raise ValueError('Please name the original sound and the replacement sound.')
    if re.search(r'\b(?:some|something|anything|sound|sounds|noise)\b',effect,re.I) and len(effect.split())==1: raise ValueError('What sound would you like me to use as the replacement?')
    if re.search(r'\b(?:between|except|without|only|after|before|but)\b',source+' '+effect,re.I): raise ValueError('Please give a specific sound and one timestamp, or ask to replace it throughout the clip.')
    effect_request(effect)
    result={'source':source,'effect':effect,'time':point}
    if point is not None and re.search(r'\b(?:all|every|throughout)\b',text,re.I): result['all_occurrences']=True
    return result


def auto_build(project,payload,cid):
    from . import discovery
    from .worker import set_stage
    pid=project['id'];snapshot=payload['_candidate'];source=candidates.safe_path(pid,snapshot['source']);duration=probe(source)['duration']
    plan=parse_replacement(payload['request'],duration)
    if file_sha256(source)!=snapshot['source_sha256']: raise ValueError('Source changed; please retry.')
    for row in snapshot['rows']:
        if file_sha256(candidates.safe_path(pid,row['path']))!=snapshot['stem_hashes'][row['id']]: raise ValueError('A layer changed; please retry.')
    selected_layer=next((r for r in snapshot['rows'] if r['id']==payload.get('source_stem_id')),None)
    if payload.get('source_stem_id') and not selected_layer:raise ValueError('The selected source layer is unavailable.')
    from .effect_followup import matching_effects, replace_known_effects, resolve_effect_reference
    resolver_unavailable=False
    if plan.get('mode')!='add':
        known=matching_effects([selected_layer] if selected_layer else snapshot['rows'],plan['source'])
        if not selected_layer and not known and any(row.get('role')=='effect' for row in snapshot['rows']):
            known=resolve_effect_reference(plan['source'],snapshot['rows'])
            if known is None:
                import re
                resolver_unavailable=True
                if re.search(r'\b(?:it|them|that|those|added|generated|placed)\b',plan['source'],re.I):
                    raise ValueError('The sound reference assistant is unavailable. Use the exact name of the effect layer.')
        if known:
            return replace_known_effects(project,payload,plan,snapshot['rows'],source,cid,known)
    event_hints=None;detection_meta={};detector_unavailable=False
    residuals=[r for r in snapshot['rows'] if r['name'].lower()=='residual']
    detection_source=candidates.safe_path(pid,selected_layer['path']) if selected_layer else (candidates.safe_path(pid,residuals[0]['path']) if residuals else source)
    source_offset=float(selected_layer['start_offset']) if selected_layer else 0.0
    search_duration=probe(detection_source)['duration'] if selected_layer else duration
    local_time=plan['time']-source_offset if plan['time'] is not None else None
    if local_time is not None and not 0<=local_time<search_duration:raise ValueError('The timestamp is outside the selected source layer.')
    if plan.get('spoken_word'):
        from .censorship import transcribe, MAX_DURATION
        import re
        if duration>MAX_DURATION: raise ValueError('Word timing is limited to 10 minutes per preview.')
        set_stage(cid,'transcribing',15)
        transcript_audio=project_dir(pid)/'stems'/(uid()+'-transcription.wav')
        try:
            if snapshot['rows']: export(snapshot['rows'],source,transcript_audio,'wav',False)
            else: shutil.copyfile(source,transcript_audio)
            words,language=transcribe(transcript_audio,duration)
        finally:
            transcript_audio.unlink(missing_ok=True)
        normalize=lambda word: re.sub(r"[^\w']",'',word.casefold())
        matches=[w for w in words if normalize(w['word'])==normalize(plan['spoken_word']) and w['end']<duration]
        if not matches: raise ValueError(f"I couldn't find the spoken word '{plan['spoken_word']}' in the current mix. Check the spelling or try a timestamp; transcription can miss words.")
        if len(matches)>100: raise ValueError('More than 100 matching words were found. Try a shorter clip.')
        intervals=[{'start':w['end'],'end':min(duration,w['end']+.01)} for w in matches]
        detection_meta={'method':'local-word-transcription','word':plan['spoken_word'],'language':language}
        note=f"Found {len(matches)} possible occurrence(s) of '{plan['spoken_word']}'. Word recognition and timing are approximate; review the placements."
    elif plan.get('all_occurrences'):
        from .event_detection import find_similar_events,flexsed_intervals,merge_candidate_ranges
        set_stage(cid,'finding',15)
        match=find_similar_events(detection_source,search_duration,local_time)
        event_hints=match['times'];detection_meta={'method':match['method'],'example_time':match['example_time']}
        try: flex=flexsed_intervals(detection_source,search_duration,plan['source'])
        except RuntimeError:flex=None;detector_unavailable=True
        if flex and flex['intervals']:
            reference=match['example_time']
            supports_reference=any(r['start']-.12<=reference<=r['end']+.12 for r in flex['intervals'])
            selected=[t for t in event_hints if abs(t-reference)<.04 or any(r['start']-.12<=t<=r['end']+.12 for r in flex['intervals'])]
            if supports_reference and len(selected)>1:event_hints=selected;detection_meta['filter']='flexsed-experimental'
        intervals=merge_candidate_ranges(event_hints,search_duration)
        note=f"Found {len(event_hints)} possible similar sounds using your example; review placements because matches are approximate."
    elif plan['time'] is not None:
        t=local_time; intervals=[{'start':t if plan.get('mode')=='add' else max(0,t-.05),'end':min(search_duration,t+.6)}]
        note='Used a short range around the requested timestamp. Sounds outside that range were not checked.'
        detection_meta={'method':'supplied-time'}
    else:
        from .event_detection import flexsed_intervals,candidate_times_in_regions,merge_candidate_ranges
        set_stage(cid,'finding',15)
        try: flex=flexsed_intervals(detection_source,search_duration,plan['source'])
        except RuntimeError:flex=None;detector_unavailable=True
        if flex is not None and flex['intervals']:
            event_hints=candidate_times_in_regions(detection_source,search_duration,flex['intervals'])
            intervals=merge_candidate_ranges(event_hints,search_duration)
            detection_meta={'method':'flexsed-experimental','seconds':flex['seconds'],'peak_gpu_bytes':flex['peak_gpu_bytes']}
        else:
            detected=discovery.discover(detection_source,search_duration)
            q=plan['source'].lower()
            mapped='impact' if any(w in q for w in ('punch','hit','impact','slap','thud')) else next((v for k,v in discovery.QUERY_LABELS.items() if k in q),q)
            matching=[s for s in detected['sources'] if s['label']==mapped]
            intervals=[{'start':x['start'],'end':min(x['end'],x['start']+10)} for item in matching for x in item['intervals']]
            intervals.sort(key=lambda x:x['start']);detection_meta={'method':'yamnet'}
            if len(intervals)>20: raise ValueError('I found more than 20 possible sounds. Please give an example timestamp to narrow it down.')
        if not intervals: raise ValueError(f"I couldn't confidently find {plan['source']}. Give me an example timestamp, such as 'at 4 seconds', and I can look for similar sounds.")
        note=f'Found {len(event_hints) if event_hints is not None else len(intervals)} possible {plan["source"]} occurrence(s); detection can miss or misidentify sounds.'
        if detection_meta['method']=='flexsed-experimental': note+=' The text-conditioned detector is experimental.'
    if source_offset:
        intervals=[{'start':i['start']+source_offset,'end':i['end']+source_offset} for i in intervals]
        if event_hints is not None:event_hints=[t+source_offset for t in event_hints]
        if 'example_time' in detection_meta:detection_meta['example_time']+=source_offset
    set_stage(cid,'generating',35)
    asset=prepare_effect(project,{'prompt':plan['effect']})
    set_stage(cid,'separating',60)
    output=build(project,{'_candidate':snapshot,'asset_id':asset['asset_id'],'asset_sha256':file_sha256(asset_path(pid,asset['asset_id'])),
                          'source_prompt':plan['source'] or 'sound effect','effect_label':plan['effect'],'mode':plan.get('mode','replace'),'intervals':intervals,'event_hints':event_hints,'source_stem_id':payload.get('source_stem_id'),'alignment':'onset' if plan.get('spoken_word') else 'peak'},cid)
    output['inference']['warnings'].insert(0,note)
    if resolver_unavailable:
        output['inference']['warnings'].append('Could not check for a similarly named effect layer; searched the residual audio instead.')
    if detector_unavailable:
        output['inference']['warnings'].append('The experimental text-conditioned detector was unavailable; the other detection method was used. Review the suggested placements.')
    output['inference']['warnings'].append('Generated sound count and content are approximate; listen to the effect before applying.')
    output['provenance'].update({'request':payload.get('original_request',payload['request']),'generation_prompt':asset['generation_prompt'],'requested_count':asset['requested_count'],'automatic':True,'detection':detection_meta})
    return output


def adjust(project,payload,candidate_id):
    """Revise one candidate without rerunning either model; source edits remain reversible."""
    from .core import connect
    pid=project['id']
    with connect() as db:
        job=db.execute("SELECT * FROM jobs WHERE id=? AND project_id=? AND kind='candidate'",(payload['parent_id'],pid)).fetchone()
    if not job or job['status']!='completed' or not job['output']:raise ValueError('The previous preview is unavailable.')
    output=json.loads(job['output']);provenance=output.get('provenance') or {};meta=output.get('_replacement_edit')
    if provenance.get('kind')!='replacement' or not meta:raise ValueError('This preview cannot be adjusted; generate it again.')
    events=provenance.get('events') or [];duration=meta['count']/meta['rate'];operation=payload['operation'];dirty_groups=set()
    if operation=='alignment':
        alignment=payload['alignment']
        if alignment not in ('peak','onset'):raise ValueError('Unsupported alignment.')
        provenance['alignment']=alignment
    elif operation=='add':
        time=payload['time']
        if not 0<=time<duration or len(events)>=100:raise ValueError('The new placement is outside the audio or the limit was reached.')
        group=None
        if provenance['mode']=='replace':
            group=next((i for i,g in enumerate(meta['groups']) if g['interval']['start']<=time<g['interval']['end']),None)
            if group is None:raise ValueError('Add a point within a processed range, or generate a new preview for that time.')
        events.append({'id':uid(),'group':group,'source_peak':time,'source_onset':time,'offset':0,'enabled':True})
        if group is not None:dirty_groups.add(group)
    else:
        event=next((e for e in events if e['id']==payload.get('event_id')),None)
        if not event:raise ValueError('This placement is unavailable.')
        if operation=='toggle':
            event['enabled']=payload['enabled']
            if event['group'] is not None:dirty_groups.add(event['group'])
        elif operation=='move':
            anchor=event['source_onset'] if provenance['alignment']=='onset' else event['source_peak']
            if not 0<=payload['time']<duration or abs(payload['time']-anchor)>.5:raise ValueError('Move a placement by at most half a second within the audio.')
            event['offset']=payload['time']-anchor
        else:raise ValueError('Unsupported placement adjustment.')
    effect=candidates.safe_path(pid,meta['effect_path']);source=candidates.safe_path(pid,output['snapshot']['source'])
    directory=project_dir(pid)/'stems';created=[]
    try:
        for group_id,group in enumerate(meta['groups']):
            if group_id not in dirty_groups:continue
            raw=candidates.safe_path(pid,group['raw_path']);row=next((r for r in output['proposed'] if r['id']==group['target_id']),None)
            if not row:raise ValueError('The separated sound layer is unavailable.')
            path=directory/(uid()+'-ducked.wav');created.append(path)
            duck_events(raw,path,events,group_id,group.get('source_rate',meta['rate']),group.get('source_offset',0));row['path']=str(path)
        placed=directory/(uid()+'.wav');mix=directory/(uid()+'.wav');created.extend([placed,mix])
        notices,_,_=render_effect(effect,placed,events,meta['rate'],meta['channels'],meta['count'],provenance['alignment'])
        prior_ids=set(meta['effect_ids']);base=[r for r in output['proposed'] if r['id'] not in prior_ids]
        clips=effect_layers(pid,meta['effect_label'],effect,events,meta['rate'],meta['count'],provenance['alignment'],any(r['solo'] for r in base),meta.get('effect_group_id'),meta.get('effect_asset_id'))
        created.extend(Path(r['path']) for r in clips)
        output['proposed']=base+clips;meta['effect_ids']=[e['id'] for e in events]
        export(output['proposed'],source,mix,'wav',False)
        output['target']={'path':str(placed)};output['mix']={'path':str(mix)}
        provenance['peak_times']=[round((e['source_onset'] if provenance['alignment']=='onset' else e['source_peak'])+e.get('offset',0),6) for e in events if e['enabled']]
        dynamic=('Some effects overlap','An effect extends','An effect started')
        output['inference']['warnings']=[w for w in output['inference'].get('warnings',[]) if not w.startswith(dynamic)]+notices
        return output
    except Exception:
        for path in created:path.unlink(missing_ok=True)
        raise
