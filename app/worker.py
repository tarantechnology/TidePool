import json
import threading
import time
from pathlib import Path
from .core import connect, event, project_dir, probe, stem, uid
from .media import separate, export
from .messaging import provider
from .visual import project_source

stop = threading.Event()

def enqueue(pid, kind, payload):
    job=uid()
    with connect() as db:
        db.execute('INSERT INTO jobs(id,project_id,kind,status,payload) VALUES(?,?,?,?,?)',(job,pid,kind,'queued',json.dumps(payload)))
        if kind=='process': db.execute("UPDATE projects SET status='queued' WHERE id=?",(pid,))
    return job

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
            inference={}
            results=separate(p['audio'],project_dir(pid)/'stems',payload,report=inference)
            items=[]
            for name,path in results:
                info=probe(path); items.append((uid(),pid,name,str(path),info['duration']))
            with connect() as db:
                db.execute('DELETE FROM stems WHERE project_id=?',(pid,))
                db.executemany('INSERT INTO stems(id,project_id,name,path,duration) VALUES(?,?,?,?,?)',items)
                output=[stem(s) for s in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
                db.execute("UPDATE projects SET status='completed' WHERE id=?",(pid,))
                db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?",(json.dumps({'stems':output,'inference':inference}),job['id']))
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
        event('job_failed',job_id=job['id'],error=str(e))
        with connect() as db:
            db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",(str(e)[:1000],job['id']))
            if job['kind']=='process': db.execute("UPDATE projects SET status='failed' WHERE id=?",(pid,))
    return True

def loop():
    while not stop.is_set():
        if not work_once(): stop.wait(.25)
