"""Bounded, previewable per-layer EQ and compression."""
import shutil
from .candidates import safe_path, revisions
from .core import probe, project_dir, run, uid
from .media import export
from .visual import file_sha256


def build(project, payload, candidate_id):
    pid=project['id']; snapshot=payload['_candidate']; source=safe_path(pid,snapshot['source'])
    if file_sha256(source)!=snapshot['source_sha256']: raise ValueError('Source changed; generate another preview.')
    rows=snapshot['rows']
    for row in rows:
        if file_sha256(safe_path(pid,row['path']))!=snapshot['stem_hashes'][row['id']]: raise ValueError('A layer changed; generate another preview.')
    selected=payload['stem_id'];group=payload.get('effect_group_id')
    if rows:
        if group:
            selected_ids={r['id'] for r in rows if r.get('role')=='effect' and r.get('effect_group_id')==group}
            if not selected_ids:raise ValueError('Effect group no longer exists.')
        else:
            if not any(r['id']==selected for r in rows): raise ValueError('Layer no longer exists.')
            selected_ids={selected}
        proposed=[dict(r) for r in rows]
    else:
        if selected is not None: raise ValueError('Layer no longer exists.')
        proposed=[{'id':uid(),'project_id':pid,'name':'Original audio','path':str(source),'duration':probe(source)['duration'],
                   'start_offset':0,'volume':1,'muted':0,'solo':0,'automation':'[]','mute_ranges':'[]','revision':0}]
        selected=proposed[0]['id']
        selected_ids={selected}
    params=payload['effects'];directory=project_dir(pid)/'stems';directory.mkdir(parents=True,exist_ok=True)
    targets=[row for row in proposed if row['id'] in selected_ids];target=targets[0]
    original=directory/(uid()+'.wav');mix=directory/(uid()+'.wav');rendered=[]
    try:
        if rows: export(rows,source,original,'wav',False)
        else: shutil.copyfile(source,original)
        filters=[]
        if params['highpass']>0: filters.append(f"highpass=f={params['highpass']}")
        if params['bass']!=0: filters.append(f"bass=g={params['bass']}:f=120:w=0.6")
        if params['treble']!=0: filters.append(f"treble=g={params['treble']}:f=4000:w=0.7")
        if params['compression']>1: filters.append(f"acompressor=threshold=0.25:ratio={params['compression']}:attack=20:release=250:makeup=1")
        for item in targets:
            path=directory/(uid()+'.wav');rendered.append(path)
            run('ffmpeg','-v','error','-y','-i',str(safe_path(pid,item['path'])),'-af',','.join(filters) or 'anull','-c:a','pcm_f32le',str(path))
            before=probe(item['path']);after=probe(path)
            if (before['sample_rate'],before['channels'])!=(after['sample_rate'],after['channels']) or abs(before['duration']-after['duration'])>1/before['sample_rate']: raise ValueError('Effects changed the audio timeline.')
            item['path']=str(path)
        export(proposed,source,mix,'wav',False)
        return {'target':{'path':str(rendered[0])},'mix':{'path':str(mix)},'original':{'path':str(original)},'snapshot':snapshot,'proposed':proposed,
                'inference':{'model':{'backend':'manual-effects','is_mock':False},'warnings':['Effects are rendered into this layer when applied. Use Undo to restore the previous audio. Listen for clipping.']},
                'provenance':{'kind':'effects','layer':target['name'],'effects':params,'stem_revisions':revisions(rows),'source_sha256':snapshot['source_sha256']}}
    except Exception:
        for path in (original,*rendered,mix): path.unlink(missing_ok=True)
        raise
