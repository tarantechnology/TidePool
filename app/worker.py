import json
import threading
import time
from .core import connect, event, project_dir, probe, run, stem, uid
from .media import InferenceError, separate, export
from .messaging import provider
from .visual import project_source, file_sha256, validate_timeline

stop = threading.Event()

def enqueue(pid, kind, payload, stage=None):
    job=uid()
    with connect() as db:
        db.execute('INSERT INTO jobs(id,project_id,kind,status,payload,stage,progress) VALUES(?,?,?,?,?,?,?)',(job,pid,kind,'queued',json.dumps(payload),stage,5 if stage else 0))
        if kind=='process': db.execute("UPDATE projects SET status='queued' WHERE id=?",(pid,))
    return job

def set_stage(job, stage, progress):
    with connect() as db: db.execute('UPDATE jobs SET stage=?,progress=? WHERE id=?',(stage,progress,job))

def normalize(pid, p, finish=None):
    """Re-encode a CFR, zero-start, unrotated copy and switch the project to it.

    The uploaded original and old audio stay untouched. ``finish(db)`` runs in the same
    transaction as the project update. Returns the updated project row as a dict."""
    from fractions import Fraction
    info=json.loads(p['metadata']); fps=Fraction(info['frame_rate']).limit_denominator(1001)
    video=project_dir(pid)/'original'/(uid()+'.mp4')
    audio=project_dir(pid)/'extracted'/(uid()+'.wav')
    try:
        run('ffmpeg','-v','error','-y','-i',p['original'],'-map','0:v:0','-map','0:a:0',
            '-vf',f'fps={fps.numerator}/{fps.denominator},setpts=PTS-STARTPTS',
            '-af','asetpts=PTS-STARTPTS','-c:v','libx264','-preset','veryfast','-crf','18',
            '-pix_fmt','yuv420p','-c:a','alac','-movflags','+faststart',str(video))
        run('ffmpeg','-v','error','-y','-i',str(video),'-vn','-ac','2','-ar','48000','-c:a','pcm_f32le',str(audio))
        # The copy was forced to exactly ``fps``; ffprobe's avg_frame_rate re-derives it from
        # nb_frames/duration and some muxer versions drop the last frame's duration (10
        # frames over 0.9 s reads as 11.1 fps), which would fail validate_timeline below.
        info=probe(video); info.update(video_id=pid,source_sha256=file_sha256(video),frame_rate=float(fps))
        validate_timeline(video,float(fps))
        (project_dir(pid)/'extracted'/'preview.mp4').unlink(missing_ok=True)
        with connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT original FROM projects WHERE id=?',(pid,)).fetchone()
            if current['original'] != p['original'] or db.execute('SELECT 1 FROM stems WHERE project_id=? LIMIT 1',(pid,)).fetchone():
                raise ValueError('Project changed while preparing the video; existing media and layers were preserved.')
            db.execute("UPDATE projects SET original=?,audio=?,metadata=?,visual=NULL,status='uploaded' WHERE id=?",(str(video),str(audio),json.dumps(info),pid))
            if finish: finish(db)
            return dict(db.execute('SELECT * FROM projects WHERE id=?',(pid,)).fetchone())
    except Exception:
        video.unlink(missing_ok=True); audio.unlink(missing_ok=True); raise

def work_once():
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
        if not row: return False
        db.execute("UPDATE jobs SET status='processing',progress=5 WHERE id=?",(row['id'],))
        if row['kind']=='process': db.execute("UPDATE projects SET status='processing' WHERE id=?",(row['project_id'],))
    job=dict(row); pid=job['project_id']; payload=json.loads(job['payload'])
    event('job_started',job_id=job['id'],kind=job['kind'])
    try:
        with connect() as db: p=dict(db.execute('SELECT * FROM projects WHERE id=?',(pid,)).fetchone())
        if job['kind']=='process':
            if payload.get('source'):
                source=project_source(p)
                if source['sha256'] != payload['source']['sha256']:
                    raise ValueError('Uploaded media hash changed after the visual job was queued')
                payload['source']=source
            payload['request_id']=job['id']
            baseline=payload.pop('_baseline',None)
            append=payload.pop('_append',None)
            residual=None
            if append:
                with connect() as db: residual=db.execute('SELECT * FROM stems WHERE project_id=? AND id=?',(pid,append['id'])).fetchone()
                if not residual or residual['revision'] != append['revision']: raise ValueError('Residual stem changed while separation was queued; retry after reloading.')
            inference={}
            results=separate(residual['path'] if residual else p['audio'],project_dir(pid)/'stems',payload,report=inference)
            results=[r if isinstance(r,dict) else {'name':r[0],'path':r[1]} for r in results]
            if append and sum(r['name'].lower()=='residual' for r in results)!=1: raise ValueError('Inference must return exactly one residual to add a sound; existing stems were preserved.')
            items=[]
            for result in results:
                info=probe(result['path'])
                # Stem metadata as the audio service reported it; NULL where it said nothing.
                spans=result.get('spans')
                items.append((uid(),pid,result['name'],str(result['path']),info['duration'],
                              result.get('role'),result.get('source_stem_id'),result.get('prompt'),result.get('label'),
                              result.get('sample_rate'),result.get('channels'),result.get('format'),
                              json.dumps(spans) if spans is not None else None,
                              json.dumps(result['refinement']) if result.get('refinement') is not None else None))
            with connect() as db:
                db.execute('BEGIN IMMEDIATE')
                if baseline is not None and {r['id']:r['revision'] for r in db.execute('SELECT id,revision FROM stems WHERE project_id=?',(pid,))} != baseline:
                    raise ValueError('Layers changed during processing; existing edits were preserved. Generate another preview.')
                if append:
                    current=db.execute('SELECT * FROM stems WHERE project_id=? AND id=?',(pid,append['id'])).fetchone()
                    if not current or current['revision'] != append['revision']: raise ValueError('Residual stem changed during separation; existing stems were preserved. Reload and retry.')
                    db.execute('DELETE FROM stems WHERE project_id=? AND id=?',(pid,append['id']))
                    db.executemany('INSERT INTO stems(id,project_id,name,path,duration,role,source_stem_id,prompt,label,sample_rate,channels,format,spans,refinement,start_offset,volume,muted,solo,automation,mute_ranges) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', [(*item,current['start_offset'],current['volume'],current['muted'],current['solo'],current['automation'],current['mute_ranges']) for item in items])
                else:
                    db.execute('DELETE FROM stems WHERE project_id=?',(pid,))
                    db.executemany('INSERT INTO stems(id,project_id,name,path,duration,role,source_stem_id,prompt,label,sample_rate,channels,format,spans,refinement) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',items)
                if payload.get('edit_scope',{}).get('action','keep') != 'keep':
                    from .guidance import edit_target
                    for item in items:
                        if item[2].lower() == 'residual': continue
                        row=db.execute('SELECT * FROM stems WHERE id=?',(item[0],)).fetchone()
                        edited=edit_target(row,payload['edit_scope'])
                        db.execute('UPDATE stems SET volume=?,muted=?,mute_ranges=? WHERE id=?',(edited['volume'],edited['muted'],edited['mute_ranges'],item[0]))
                output=[stem(s) for s in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
                db.execute("UPDATE projects SET status='completed' WHERE id=?",(pid,))
                db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?",(json.dumps({'stems':output,'inference':inference}),job['id']))
        elif job['kind']=='discovery':
            from .discovery import discover
            if file_sha256(p['audio']) != payload['audio_sha256']: raise ValueError('Project audio changed; run discovery again.')
            output=discover(p['audio'], json.loads(p['metadata'])['duration'])
            with connect() as db:
                current=db.execute('SELECT audio FROM projects WHERE id=?',(pid,)).fetchone()
                if not current or current['audio']!=p['audio'] or file_sha256(p['audio'])!=payload['audio_sha256']:
                    raise ValueError('Project audio changed; run discovery again.')
                db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?",(json.dumps(output),job['id']))
        elif job['kind']=='candidate':
            if payload.get('_effects'):
                from .effects import build
            elif payload.get('_director_adjust'):
                from .director import adjust as build
            elif payload.get('_director'):
                from .director import build
            elif payload.get('_cinematic'):
                from .cinematic import build
            else:
                from .candidates import build
            output=build(p,payload,job['id'])
            with connect() as db:
                db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?",(json.dumps(output),job['id']))
        elif job['kind']=='refine':
            from .refine import refine
            refine(p,payload,job['id'])
        elif job['kind']=='normalize':
            def finish(db): db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?",(json.dumps({'prepared':True}),job['id']))
            normalize(pid,p,finish)
        elif job['kind']=='vision':
            prepared=False
            selection_only=payload.get('object_selection') is True
            if selection_only and file_sha256(p['original'])!=payload['source_sha256']:
                raise ValueError('Project video changed; find objects again.')
            if payload.get('prepare'):
                try: validate_timeline(p['original'],json.loads(p['metadata'])['frame_rate'])
                except ValueError:
                    set_stage(job['id'],'preparing',15)
                    p=normalize(pid,p); prepared=True
            set_stage(job['id'],'detecting',30)
            directory=project_dir(pid)/'extracted'/('vision-'+job['id'])
            from .detector import detect
            if selection_only:
                from .object_selection import detect as select_objects
                metadata=select_objects(p['original'],directory,pid,payload['concept'])
            else:
                metadata=detect(p['original'],directory,pid)
            with connect() as db:
                db.execute('BEGIN IMMEDIATE')
                if db.execute('SELECT original FROM projects WHERE id=?',(pid,)).fetchone()['original'] != p['original']:
                    raise ValueError('Project media changed during object detection; run detection again.')
                if selection_only and file_sha256(p['original'])!=payload['source_sha256']:
                    raise ValueError('Project video changed during detection; find objects again.')
                db.execute('UPDATE projects SET visual=? WHERE id=?',(json.dumps(metadata),pid))
                db.execute("UPDATE jobs SET status='completed',stage='completed',progress=100,output=? WHERE id=?",(json.dumps({'analyzed_frames':metadata['analyzed_frames'],'processed_duration_seconds':metadata['processed_duration_seconds'],'prepared':prepared}),job['id']))
        else:
            with connect() as db: stems=[dict(s) for s in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
            if not stems: raise ValueError('No stems available')
            path=project_dir(pid)/'exports'/(job['id']+'.'+payload['format'])
            path.parent.mkdir(parents=True,exist_ok=True)
            export(stems,p['original'],path,payload['format'],payload['include_video'])
            with connect() as db: db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?",(json.dumps({'path':str(path)}),job['id']))
        if job['kind']=='process':
            try: provider.send(pid, 'Your SoundCanvas video is ready to edit.')
            except Exception as exc: event('notification_failed', job_id=job['id'], error=str(exc))
        event('job_completed',job_id=job['id'])
    except Exception as e:
        # Only the audio service reports structured error codes.
        code=e.code if isinstance(e,InferenceError) else None
        retryable=e.retryable if isinstance(e,InferenceError) else None
        event('job_failed',job_id=job['id'],error=str(e),error_code=code,retryable=retryable)
        with connect() as db:
            db.execute("UPDATE jobs SET status='failed',error=?,error_code=?,retryable=? WHERE id=?",(str(e)[:1000],code,retryable,job['id']))
            if job['kind']=='process': db.execute("UPDATE projects SET status='failed' WHERE id=?",(pid,))
    return True

def loop():
    while not stop.is_set():
        if not work_once(): stop.wait(.25)
