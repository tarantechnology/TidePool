"""Candidate snapshots, rendered mix previews and atomic, revision-checked history."""
import json
import shutil
from pathlib import Path
from fastapi import HTTPException
from .core import connect, probe, project_dir, stem, uid
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
    if 'rows' not in snapshot: raise ValueError('Queued preview uses an older format; generate a new preview.')
    source=safe_path(pid,snapshot['source'])
    if file_sha256(source)!=snapshot['source_sha256']: raise ValueError('Source changed; generate another preview')
    payload['request_id']=cid
    inference={}; directory=project_dir(pid)/'stems'
    results=separate(source,directory,payload,report=inference)
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
    target['role']='target'; residual['role']='residual'
    target['refinement']=json.dumps(refinement) if refinement is not None else None
    if not snapshot['target_id']:
        target['prompt']=payload['prompts'][0];target['label']=target['name'];target['source_stem_id']=None
    residual['label']='everything else'
    replaced={snapshot['target_id'],snapshot['residual_id']}
    proposed=[dict(r) for r in rows if r['id'] not in replaced]+[target,residual]
    mix=directory/(uid()+'.wav'); original=directory/(uid()+'.wav')
    export(proposed,source,mix,'wav',False)
    if rows: export(rows,source,original,'wav',False)
    else: shutil.copyfile(source,original)
    return {'target':target,'residual':residual,'mix':{'path':str(mix)},'original':{'path':str(original)},
            'snapshot':snapshot,'proposed':proposed,'inference':inference,
            'provenance':{'source_sha256':snapshot['source_sha256'],'source_kind':'original' if snapshot['target_id'] or not rows else 'residual',
                          'stem_revisions':revisions(rows),'prompt':payload['prompts'][0],'anchors':payload.get('anchors',[]),
                          'visual_guidance':bool(payload.get('visual_prompt')),'edit_scope':payload.get('edit_scope'),
                          'profile':payload.get('profile','preview'),'requested_options':payload['options'],
                          'passes':payload['options'].get('passes',1),'refinement':refinement}}


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
        except (OSError,ValueError) as exc: raise HTTPException(409,'Preview media changed or is unavailable; regenerate it.') from exc
        project=db.execute('SELECT audio FROM projects WHERE id=?',(pid,)).fetchone()
        if project['audio']!=snapshot.get('project_audio',project['audio']): raise HTTPException(409,'Project media changed since preview; regenerate it.')
        old=revisions(current)
        for item in data['proposed']:
            if item['id'] in old:
                # The snapshot is unchanged, including every existing edit.
                db.execute('UPDATE stems SET path=?,duration=?,volume=?,muted=?,mute_ranges=?,revision=revision+1 WHERE id=? AND project_id=?',
                           (item['path'],item['duration'],item['volume'],item['muted'],item['mute_ranges'],item['id'],pid))
                if 'refinement' in item:  # Previews built before refinement existed leave it alone.
                    db.execute('UPDATE stems SET refinement=? WHERE id=? AND project_id=?',(item['refinement'],item['id'],pid))
            else:
                keys=list(item)
                db.execute('INSERT INTO stems('+','.join(keys)+') VALUES('+','.join('?' for _ in keys)+')',list(item.values()))
        after=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
        db.execute('INSERT INTO candidate_history(candidate_id,previous_rows,applied_revisions) VALUES(?,?,?)',(job['id'],json.dumps(current),json.dumps(revisions(after))))
        jid=completed_job(db,pid)
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
        db.execute('DELETE FROM stems WHERE project_id=?',(pid,))
        for row in previous:
            row['revision']=revs[row['id']]+1
            keys=list(row)
            db.execute('INSERT INTO stems('+','.join(keys)+') VALUES('+','.join('?' for _ in keys)+')',list(row.values()))
        db.execute('UPDATE candidate_history SET undone=1 WHERE candidate_id=?',(job['id'],))
        jid=completed_job(db,pid)
    return {'status':'undone','job_id':jid}
