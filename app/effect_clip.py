"""Revision-checked moves and removals of individual effect clips."""
from . import candidates
from .core import probe, project_dir, uid
from .media import export
from .visual import file_sha256


def build(project,payload,cid):
    pid=project['id'];snapshot=payload['_candidate'];source=candidates.safe_path(pid,snapshot['source']);rows=snapshot['rows']
    if file_sha256(source)!=snapshot['source_sha256']:raise ValueError('Source changed; try again.')
    for row in rows:
        if file_sha256(candidates.safe_path(pid,row['path']))!=snapshot['stem_hashes'][row['id']]:raise ValueError('A layer changed; try again.')
    selected=next((r for r in rows if r['id']==payload['stem_id'] and r.get('role')=='effect'),None)
    if not selected:raise ValueError('The effect clip no longer exists.')
    proposed=[dict(r) for r in rows if payload['operation']!='delete' or r['id']!=selected['id']]
    removed=[selected['id']] if payload['operation']=='delete' else []
    if not removed:
        offset=payload['start_offset']
        if not 0<=offset or offset+selected['duration']>probe(source)['duration']+1/48000:raise ValueError('Move the entire clip within the project.')
        next(r for r in proposed if r['id']==selected['id'])['start_offset']=offset
    directory=project_dir(pid)/'stems';original=directory/(uid()+'.wav');mix=directory/(uid()+'.wav')
    export(rows,source,original,'wav',False);export(proposed,source,mix,'wav',False)
    return {'target':{'path':selected['path']},'original':{'path':str(original)},'mix':{'path':str(mix)},'snapshot':snapshot,'proposed':proposed,'remove_ids':removed,
            'inference':{'model':{'backend':'effect-clip-edit'},'warnings':[]},'provenance':{'kind':'effect-clip-edit','operation':payload['operation']}}
