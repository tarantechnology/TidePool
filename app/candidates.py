"""Candidate snapshots, rendered mix previews and atomic, revision-checked history."""
import json
import shutil
from pathlib import Path
from fastapi import HTTPException
from .core import connect, discard, probe, project_dir, run, stem, uid
from .guidance import edit_target
from .media import separate, export
from .visual import file_sha256


def revisions(rows):
    return {r['id']:r['revision'] for r in rows}


def safe_path(pid, path):
    path = Path(path).resolve()
    if not path.is_relative_to(project_dir(pid).resolve()) or not path.is_file():
        raise ValueError('Audio reference is unavailable or outside this project')
    return path


def build(project, payload, cid):
    pid=project['id']; snapshot=payload.pop('_candidate')
    object_context=payload.pop('_object_context',None)
    if 'rows' not in snapshot: raise ValueError('Queued preview uses an older format; generate a new preview.')
    source=safe_path(pid,snapshot['source'])
    mix_source=safe_path(pid,snapshot.get('project_audio') or snapshot['source'])
    source_layer_id=snapshot.get('source_layer_id');source_offset=float(snapshot.get('source_offset',0))
    if file_sha256(source)!=snapshot['source_sha256']: raise ValueError('Source changed; generate another preview')
    reuse=payload.pop('_reuse_target_id',None)
    if reuse:
        return reuse_target(project,payload,cid,snapshot,object_context,reuse)
    payload['request_id']=cid
    inference={}; directory=project_dir(pid)/'stems'
    interval=payload.pop('_processing_interval',None)
    point=payload.pop('_processing_time',None)
    source_anchors=payload.get('anchors',[])
    localization=None
    if point is not None:
        from .discovery import locate_near
        localization=locate_near(source, probe(source)['duration'], payload['prompts'][0], point)
        interval=localization['interval']
    service_source=source
    if interval:
        start,end=interval['start'],interval['end']
        duration=probe(source)['duration']
        # Context is only for model inference. The output remains inside the selected range.
        padding=min(1.0,max(0.0,(10.0-(end-start))/2))
        context_start=max(0.0,start-padding);context_end=min(duration,end+padding)
        service_source=directory/(uid()+'-segment.wav')
        run('ffmpeg','-v','error','-y','-ss',str(context_start),'-i',str(source),'-t',str(context_end-context_start),'-c:a','pcm_f32le',str(service_source))
        payload['anchors']=[{**a,'start':max(0,a['start']-context_start),'end':min(context_end-context_start,a['end']-context_start)} for a in source_anchors if a['start']>=context_start and a['end']<=context_end]
    results=separate(service_source,directory,payload,report=inference)
    if interval:
        selected=[r for r in results if r['name'].lower()!='residual']
        if len(selected)!=1: raise ValueError('Preview requires exactly one target; existing layers were preserved.')
        info=probe(source); rate=info['sample_rate']; count=round(info['duration']*rate)
        offset=round((start-context_start)*rate);delay=round(start*rate);length=round((end-start)*rate)
        fade=min(.02,(end-start)/4)
        full_target=directory/(uid()+'-target.wav'); full_residual=directory/(uid()+'-residual.wav')
        run('ffmpeg','-v','error','-y','-i',str(selected[0]['path']),'-af',f'atrim=start_sample={offset}:end_sample={offset+length},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={fade},afade=t=out:st={max(0,end-start-fade)}:d={fade},adelay={delay}S:all=1,apad,atrim=end_sample={count}','-c:a','pcm_f32le',str(full_target))
        run('ffmpeg','-v','error','-y','-i',str(source),'-i',str(full_target),'-filter_complex','[1:a]volume=-1[neg];[0:a][neg]amix=inputs=2:duration=first:normalize=0','-c:a','pcm_f32le',str(full_residual))
        results=[{**selected[0],'path':full_target},{'name':'residual','path':full_residual}]
        if localization and localization.get('notice'):
            inference.setdefault('warnings',[]).append(localization['notice'])
    elif source_layer_id:
        # Some backends additionally suppress their residual. A split of a chosen
        # editable layer must reconstruct that layer before any user edits.
        found=[r if isinstance(r,dict) else {'name':r[0],'path':r[1]} for r in results]
        isolated=[r for r in found if r['name'].lower()!='residual']
        if len(isolated)!=1:raise ValueError('Preview requires exactly one target; existing layers were preserved.')
        remainder=directory/(uid()+'-layer-remainder.wav')
        run('ffmpeg','-v','error','-y','-i',str(source),'-i',str(isolated[0]['path']),'-filter_complex','[1:a]volume=-1[neg];[0:a][neg]amix=inputs=2:duration=first:normalize=0','-c:a','pcm_f32le',str(remainder))
        results=[isolated[0],{'name':'residual','path':remainder}]
    results=[r if isinstance(r,dict) else {'name':r[0],'path':r[1]} for r in results]
    refinement=next((r.get('refinement') for r in results if r['name'].lower()!='residual'),None)
    targets=[(r['name'],r['path']) for r in results if r['name'].lower()!='residual']
    residuals=[(r['name'],r['path']) for r in results if r['name'].lower()=='residual']
    if len(targets)!=1 or len(residuals)!=1: raise ValueError('Preview requires one target and one residual; current layers were preserved.')
    source_info=probe(source)
    for result in results:
        info=probe(result['path'])
        if abs(info['duration']-source_info['duration'])>1/48000 or (info['sample_rate'],info['channels'])!=(source_info['sample_rate'],source_info['channels']):
            raise ValueError('Preview stems must match the source timeline and channels; existing layers were preserved.')
    rows=snapshot['rows']; by_id={r['id']:r for r in rows}
    def new_row(name,path,sid):
        # New splits inherit the residual edits so the unedited split preserves the mix.
        original=by_id.get(sid) or by_id.get(snapshot['residual_id'])
        row=dict(original) if original else dict(project_id=pid,start_offset=0,volume=1,muted=0,solo=0,automation='[]',mute_ranges='[]',revision=0)
        row.update(id=sid or uid(),name=original['name'] if sid else name,path=str(path),duration=source_info['duration'])
        if not sid: row['revision']=0
        return row
    target=edit_target(new_row(*targets[0],snapshot['target_id']),payload.get('edit_scope',{'action':'keep'}))
    residual=new_row(*residuals[0],snapshot['residual_id'])
    target['role']='target'
    if source_layer_id:
        # The selected layer becomes its own remainder, not the project residual.
        parent=by_id[source_layer_id]
        residual['role']=parent.get('role');residual['label']=parent.get('label')
        if snapshot.get('promote_original'):
            residual.update(name='residual',role='residual',label='everything else')
        for key in ('effect_group_id','effect_asset_id','effect_label','sound_group_id','sound_group_label','sound_start','sound_end'):
            target[key]=None
    else:residual['role']='residual'
    target['refinement']=json.dumps(refinement) if refinement is not None else None
    if not snapshot['target_id']:
        target['prompt']=payload['prompts'][0];target['label']=target['name'];target['source_stem_id']=source_layer_id
    if not source_layer_id:residual['label']='everything else'
    replaced={snapshot['target_id'],snapshot['residual_id']}
    proposed=[dict(r) for r in rows if r['id'] not in replaced]+[target,residual]
    if not snapshot['target_id']:
        # The same explicitly requested description in one project represents the
        # same logical sound. Do not combine different existing group identities.
        prompt=' '.join(payload['prompts'][0].casefold().split())
        matching=[r for r in proposed if r['id'] not in (target['id'],source_layer_id) and r.get('role')=='target' and r.get('source_stem_id')==source_layer_id and r.get('prompt') and ' '.join(r['prompt'].casefold().split())==prompt]
        existing={r.get('sound_group_id') for r in matching if r.get('sound_group_id')}
        group=next(iter(existing)) if len(existing)==1 else uid()
        target.update(sound_group_id=group,sound_group_label=payload['prompts'][0],sound_start=interval['start'] if interval else None,sound_end=interval['end'] if interval else None)
        if len(existing)<=1:
            for item in matching:item.update(sound_group_id=group,sound_group_label=payload['prompts'][0])
        elif matching:
            inference.setdefault('warnings',[]).append('Multiple existing sound groups use this description; the new layer was kept separate.')
    mix=directory/(uid()+'.wav'); original=directory/(uid()+'.wav')
    export(proposed,mix_source,mix,'wav',False)
    if rows: export(rows,mix_source,original,'wav',False)
    else: shutil.copyfile(source,original)
    def timeline_range(value):
        return {'start':value['start']+source_offset,'end':value['end']+source_offset} if value else None
    public_anchors=[{**a,'start':a['start']+source_offset,'end':a['end']+source_offset} for a in source_anchors]
    if localization and source_offset:
        localization={**localization,'interval':timeline_range(localization.get('interval'))}
    return {'target':target,'residual':residual,'mix':{'path':str(mix)},'original':{'path':str(original)},
            'snapshot':snapshot,'proposed':proposed,'inference':inference,
            'provenance':{'source_sha256':snapshot['source_sha256'],'source_kind':'selected-layer' if source_layer_id else ('original' if snapshot['target_id'] or not rows else 'residual'),'source_layer_id':source_layer_id,'source_offset':source_offset,
                          'stem_revisions':revisions(rows),'prompt':payload['prompts'][0],'anchors':public_anchors,'object_selection':object_context,
                          'visual_guidance':bool(payload.get('visual_prompt')),'edit_scope':payload.get('edit_scope'),
                          'profile':payload.get('profile','preview'),'processing_interval':timeline_range(interval),'processing_time':point+source_offset if point is not None else None,'localization':localization,'inference_interval':timeline_range({'start':context_start,'end':context_end}) if interval else None,'requested_options':payload['options'],
                          'passes':payload['options'].get('passes',1),'refinement':refinement}}


def reuse_target(project,payload,cid,snapshot,context,target_id):
    """Preview a cached layer edit through the same reversible history contract."""
    rows=snapshot['rows'];pid=project['id'];source=safe_path(pid,snapshot['source'])
    for row in rows:
        if file_sha256(safe_path(pid,row['path']))!=snapshot['stem_hashes'][row['id']]:
            raise ValueError('A source layer changed; preview the edit again.')
    proposed=[edit_target(row,payload['edit_scope']) if row['id']==target_id else dict(row) for row in rows]
    target=next(row for row in proposed if row['id']==target_id)
    directory=project_dir(pid)/'stems'
    mix=directory/(uid()+'.wav');original=directory/(uid()+'.wav')
    export(proposed,source,mix,'wav',False);export(rows,source,original,'wav',False)
    residual=next((row for row in rows if row['role']=='residual'),None)
    return {'target':target,'mix':{'path':str(mix)},'original':{'path':str(original)},
            **({'residual':residual} if residual else {}),'snapshot':snapshot,'proposed':proposed,
            'inference':{'model':{'backend':'existing-stem'},'warnings':['Reused the existing sound layer; no new separation was run.']},
            'provenance':{'kind':'separation','source_sha256':snapshot['source_sha256'],
                          'stem_revisions':revisions(rows),'prompt':payload['prompts'][0],
                          'object_selection':context,'visual_guidance':False,'source_kind':'existing-layer',
                          'reused_target_id':target_id,'edit_scope':payload['edit_scope']}}


def completed_job(db,pid):
    jid=uid(); rows=[stem(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
    db.execute("INSERT INTO jobs(id,project_id,kind,status,progress,output) VALUES(?,?,'process','completed',100,?)",(jid,pid,json.dumps({'stems':rows})))
    db.execute('UPDATE projects SET status=? WHERE id=?',('completed' if rows else 'uploaded',pid))
    return jid


def apply(job):
    data=json.loads(job['output']); pid=job['project_id']; snapshot=data['snapshot']
    if 'rows' not in snapshot: raise HTTPException(409,'This older preview must be regenerated before applying.')
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
        if db.execute("SELECT 1 FROM jobs WHERE project_id=? AND kind IN ('process','normalize','refine') AND status IN ('queued','processing')",(pid,)).fetchone():
            raise HTTPException(409,'Wait for active media processing before applying a preview.')
        if db.execute('SELECT 1 FROM candidate_history WHERE candidate_id=?',(job['id'],)).fetchone() or revisions(current)!=revisions(snapshot['rows']):
            raise HTTPException(409,'Layers changed since preview. Generate a new preview.')
        try:
            if file_sha256(safe_path(pid,snapshot['source']))!=snapshot['source_sha256']: raise ValueError('Source changed')
            for item in data['proposed']: safe_path(pid,item['path'])
            for item in snapshot['rows']:
                expected = snapshot.get('stem_hashes', {}).get(item['id'])
                if expected and file_sha256(safe_path(pid, item['path'])) != expected: raise ValueError('A source layer changed')
        except (OSError,ValueError) as exc: raise HTTPException(409,'Preview media changed or is unavailable; regenerate it.') from exc
        project=db.execute('SELECT audio FROM projects WHERE id=?',(pid,)).fetchone()
        if project['audio']!=snapshot.get('project_audio',project['audio']): raise HTTPException(409,'Project media changed since preview; regenerate it.')
        old=revisions(current)
        remove_ids=data.get('remove_ids',[])
        if remove_ids:
            proposed_ids={r['id'] for r in data['proposed']}
            removable={r['id'] for r in current if r.get('role')=='effect'}
            if (data.get('provenance') or {}).get('kind')!='effect-clip-edit' or len(remove_ids)!=len(set(remove_ids)) or not set(remove_ids)<=removable or set(remove_ids)&proposed_ids:
                raise HTTPException(409,'The requested clip removal is no longer valid.')
            db.executemany('DELETE FROM stems WHERE id=? AND project_id=?',[(sid,pid) for sid in remove_ids])
        for item in data['proposed']:
            if item['id'] in old:
                # The snapshot is unchanged, including every existing edit.
                db.execute('UPDATE stems SET path=?,name=?,label=?,duration=?,start_offset=?,volume=?,muted=?,mute_ranges=?,revision=revision+1 WHERE id=? AND project_id=?',
                           (item['path'],item['name'],item.get('label'),item['duration'],item['start_offset'],item['volume'],item['muted'],item['mute_ranges'],item['id'],pid))
                if snapshot.get('promote_original') and item['id']==snapshot.get('source_layer_id'):
                    db.execute("UPDATE stems SET role='residual' WHERE id=? AND project_id=?",(item['id'],pid))
                for key in ('effect_group_id','effect_asset_id','effect_label','sound_group_id','sound_group_label','sound_start','sound_end'):
                    if key in item:
                        value='COALESCE(?, '+key+')' if key in ('effect_group_id','effect_asset_id','effect_label','sound_group_id','sound_group_label') else '?'
                        db.execute(f'UPDATE stems SET {key}={value} WHERE id=? AND project_id=?',(item[key],item['id'],pid))
                if 'refinement' in item:  # Previews built before refinement existed leave it alone.
                    db.execute('UPDATE stems SET refinement=? WHERE id=? AND project_id=?',(item['refinement'],item['id'],pid))
            else:
                keys=list(item)
                db.execute('INSERT INTO stems('+','.join(keys)+') VALUES('+','.join('?' for _ in keys)+')',list(item.values()))
        after=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
        db.execute('INSERT INTO candidate_history(candidate_id,previous_rows,applied_revisions) VALUES(?,?,?)',(job['id'],json.dumps(current),json.dumps(revisions(after))))
        jid=completed_job(db,pid)
    # Undo can restore the replaced files right now, so discard keeps them until it no longer can.
    discard(pid,[r['path'] for r in current])
    return {'status':'applied','job_id':jid}


def undo(job):
    pid=job['project_id']
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        history=db.execute('SELECT * FROM candidate_history WHERE candidate_id=?',(job['id'],)).fetchone()
        current=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
        if not history or history['undone'] or revisions(current)!=json.loads(history['applied_revisions']):
            raise HTTPException(409,'Cannot undo: layers changed after apply or this preview was already undone.')
        previous=json.loads(history['previous_rows']); revs=revisions(current)
        for row in previous:
            try: safe_path(pid,row['path'])
            except (ValueError,OSError) as exc: raise HTTPException(409,'Previous audio is unavailable') from exc
        # Restore complete prior rows, but revisions always advance to invalidate stale clients.
        replaced=[r['path'] for r in current]
        db.execute('DELETE FROM stems WHERE project_id=?',(pid,))
        for row in previous:
            row['revision']=revs.get(row['id'],row['revision'])+1
            keys=list(row)
            db.execute('INSERT INTO stems('+','.join(keys)+') VALUES('+','.join('?' for _ in keys)+')',list(row.values()))
        db.execute('UPDATE candidate_history SET undone=1 WHERE candidate_id=?',(job['id'],))
        jid=completed_job(db,pid)
    discard(pid,replaced)
    return {'status':'undone','job_id':jid}
