import hmac
import json
import os
import secrets
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from .core import MAX_UPLOAD, connect, digest, init, probe, project_dir, run, stem, uid
from .commands import parse
from .worker import enqueue, loop, stop

@asynccontextmanager
async def lifespan(app):
    init(); stop.clear(); thread=threading.Thread(target=loop,daemon=True); thread.start()
    yield
    stop.set(); thread.join(timeout=2)
app=FastAPI(title='SoundCanvas API',lifespan=lifespan)

def auth(pid, authorization):
    with connect() as db: p=db.execute('SELECT * FROM projects WHERE id=?',(pid,)).fetchone()
    if not p: raise HTTPException(404,'Project not found')
    token=authorization.removeprefix('Bearer ') if authorization and authorization.startswith('Bearer ') else ''
    if not token or not hmac.compare_digest(digest(token),p['token_hash']): raise HTTPException(401,'Invalid project token')
    return p

def get_job(jid, kind, authorization):
    with connect() as db: row=db.execute('SELECT * FROM jobs WHERE id=? AND kind=?',(jid,kind)).fetchone()
    if not row: raise HTTPException(404,'Job not found')
    auth(row['project_id'],authorization)
    d=dict(row); d.pop('payload'); output=json.loads(d.pop('output')) if d['output'] else None
    if kind=='process': d['stems']=output or []
    else: d['download_url']=f'/api/exports/{jid}/download' if output else None
    return d

class Keyframe(BaseModel):
    time: float=Field(ge=0,allow_inf_nan=False)
    value: float=Field(ge=0,le=2,allow_inf_nan=False)
class StemUpdate(BaseModel):
    expected_revision: int | None=Field(default=None,ge=0)
    name: str | None=Field(default=None,min_length=1,max_length=200)
    volume: float | None=Field(default=None,ge=0,le=2,allow_inf_nan=False)
    muted: bool | None=None
    solo: bool | None=None
    start_offset: float | None=Field(default=None,ge=0,le=86400,allow_inf_nan=False)
    automation: list[Keyframe] | None=Field(default=None,max_length=64)
    @model_validator(mode='after')
    def ordered(self):
        if self.automation and any(a.time>=b.time for a,b in zip(self.automation,self.automation[1:])): raise ValueError('Keyframe times must strictly increase')
        return self
class Process(BaseModel):
    prompts: list[str]=Field(default_factory=list,max_length=20)
    visual_prompt: dict | None=None
class Export(BaseModel):
    format: Literal['mp4','wav','mp3']='mp4'
    include_video: bool=True
    @model_validator(mode='after')
    def valid_format(self):
        if self.include_video != (self.format=='mp4'): raise ValueError('Use mp4 with video, or wav/mp3 without video')
        return self
class Message(BaseModel):
    project_id: str
    token: str
    text: str=Field(min_length=1,max_length=2000)

@app.post('/api/projects',status_code=201)
def create():
    pid=uid(); token=secrets.token_urlsafe(32)
    for directory in ['original','extracted','stems','exports']: (project_dir(pid)/directory).mkdir(parents=True,exist_ok=True)
    with connect() as db: db.execute('INSERT INTO projects(id,token_hash,status) VALUES(?,?,?)',(pid,digest(token),'created'))
    return {'project_id':pid,'access_token':token,'status':'created'}

@app.post('/api/projects/{pid}/upload')
async def upload(pid: str, file: UploadFile=File(...), authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if p['original']: raise HTTPException(409,'Project already has media')
    ext=Path(file.filename or '').suffix.lower()
    if ext not in ('.mp4','.mov','.wav','.mp3'): raise HTTPException(415,'Supported extensions: mp4, mov, wav, mp3')
    path=project_dir(pid)/'original'/(uid()+ext); audio=project_dir(pid)/'extracted'/'audio.wav'
    size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_UPLOAD: raise HTTPException(413,'Upload too large')
                out.write(chunk)
        info=probe(path)
        if ext in ('.mp4','.mov') and not info['has_video']: raise ValueError('Expected a video stream')
        run('ffmpeg','-v','error','-y','-i',str(path),'-vn','-ac','2','-ar','48000','-c:a','pcm_s16le',str(audio))
        with connect() as db: db.execute("UPDATE projects SET filename=?,original=?,audio=?,metadata=?,status='uploaded' WHERE id=?",(Path(file.filename or '').name,str(path),str(audio),json.dumps(info),pid))
    except HTTPException:
        path.unlink(missing_ok=True); audio.unlink(missing_ok=True); raise
    except (ValueError,RuntimeError) as e:
        path.unlink(missing_ok=True); audio.unlink(missing_ok=True); raise HTTPException(422,str(e))
    finally: await file.close()
    return {'project_id':pid,'filename':Path(file.filename or '').name,'metadata':info,'status':'uploaded'}

@app.get('/api/projects/{pid}')
def project(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    return {'project_id':pid,'status':p['status'],'filename':p['filename'],'metadata':json.loads(p['metadata']) if p['metadata'] else None,'visual_metadata':json.loads(p['visual']) if p['visual'] else None,'stems':list_stems(pid,authorization)}

@app.put('/api/projects/{pid}/visual-metadata')
def visual(pid: str, body: dict, authorization: str | None=Header(None)):
    auth(pid,authorization)
    with connect() as db: db.execute('UPDATE projects SET visual=? WHERE id=?',(json.dumps(body),pid))
    return {'status':'saved'}

def start(pid,kind,payload):
    try: return enqueue(pid,kind,payload)
    except sqlite3.IntegrityError: raise HTTPException(409,'A job of this type is already active')

@app.post('/api/projects/{pid}/process',status_code=202)
def process(pid: str, body: Process=Process(), authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    payload=body.model_dump()
    if payload['visual_prompt'] is None and p['visual']: payload['visual_prompt']=json.loads(p['visual'])
    return {'job_id':start(pid,'process',payload),'status':'queued'}

@app.get('/api/jobs/{jid}')
def job(jid: str, authorization: str | None=Header(None)): return get_job(jid,'process',authorization)

@app.get('/api/projects/{pid}/stems')
def list_stems(pid: str, authorization: str | None=Header(None)):
    auth(pid,authorization)
    with connect() as db: return [stem(s) for s in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]

def find_stem(pid,sid):
    with connect() as db: row=db.execute('SELECT * FROM stems WHERE project_id=? AND id=?',(pid,sid)).fetchone()
    if not row: raise HTTPException(404,'Stem not found')
    return row

@app.patch('/api/projects/{pid}/stems/{sid}')
def update_stem(pid: str,sid: str,body: StemUpdate,authorization: str | None=Header(None)):
    auth(pid,authorization)
    values=body.model_dump(exclude_unset=True,exclude_none=True)
    expected=values.pop('expected_revision',None)
    if 'automation' in values: values['automation']=json.dumps(values['automation'])
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM stems WHERE project_id=? AND id=?',(pid,sid)).fetchone()
        if not row: raise HTTPException(404,'Stem not found')
        if expected is not None and row['revision'] != expected:
            raise HTTPException(409,'Stem changed since it was loaded; reload to resolve the conflict')
        if values:
            db.execute('UPDATE stems SET '+','.join(k+'=?' for k in values)+',revision=revision+1 WHERE project_id=? AND id=?',(*values.values(),pid,sid))
        row=db.execute('SELECT * FROM stems WHERE project_id=? AND id=?',(pid,sid)).fetchone()
    return stem(row)

@app.get('/api/projects/{pid}/stems/{sid}/audio')
def audio(pid: str,sid: str,authorization: str | None=Header(None)):
    auth(pid,authorization); return FileResponse(find_stem(pid,sid)['path'],media_type='audio/wav')

@app.post('/api/projects/{pid}/export',status_code=202)
def create_export(pid: str,body: Export,authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['original'] or not list_stems(pid,authorization): raise HTTPException(409,'Processed stems required')
    if body.include_video and not json.loads(p['metadata'])['has_video']: raise HTTPException(422,'Original has no video')
    return {'export_id':start(pid,'export',body.model_dump()),'status':'queued'}

@app.get('/api/exports/{eid}')
def export_status(eid: str,authorization: str | None=Header(None)): return get_job(eid,'export',authorization)
@app.get('/api/exports/{eid}/download')
def download(eid: str,authorization: str | None=Header(None)):
    info=get_job(eid,'export',authorization)
    if info['status']!='completed': raise HTTPException(409,'Export is not complete')
    with connect() as db: row=db.execute('SELECT output FROM jobs WHERE id=?',(eid,)).fetchone()
    path=Path(json.loads(row['output'])['path']); return FileResponse(path,filename=path.name)

@app.post('/api/webhooks/local')
def webhook(body: Message):
    p=auth(body.project_id,'Bearer '+body.token)
    op=parse(body.text)
    if not op: return {'status':'clarification','message':'Please specify a supported command and a precise time range.'}
    if op['action']=='export':
        if not p['metadata'] or not json.loads(p['metadata'])['has_video']: return {'status':'clarification','message':'This project has no video to export.'}
        if not list_stems(body.project_id,'Bearer '+body.token): return {'status':'clarification','message':'Process the project first.'}
        eid=start(body.project_id,'export',{'format':'mp4','include_video':True})
        return {'status':'accepted','operation':op,'export_id':eid,'message':'Export queued.'}
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        rows=db.execute('SELECT * FROM stems WHERE project_id=?',(body.project_id,)).fetchall()
        matches=[r for r in rows if op['stem_name'].casefold() in r['name'].casefold()]
        if len(matches)!=1: return {'status':'clarification','message':'Please identify exactly one matching stem.'}
        s=matches[0]; action=op['action']
        if action=='mute_range':
            if op['end']>s['duration']+s['start_offset']: return {'status':'clarification','message':'The range extends beyond the stem.'}
            ranges=json.loads(s['mute_ranges']); ranges.append([op['start'],op['end']])
            db.execute('UPDATE stems SET mute_ranges=?,revision=revision+1 WHERE id=?',(json.dumps(ranges),s['id']))
        elif action in ('mute','unmute'): db.execute('UPDATE stems SET muted=?,revision=revision+1 WHERE id=?',(action=='mute',s['id']))
        else:
            volume=op['volume'] if action=='set_volume' else min(2,s['volume']+0.2)
            db.execute('UPDATE stems SET volume=?,revision=revision+1 WHERE id=?',(volume,s['id']))
        db.execute('INSERT INTO messages(id,project_id,text) VALUES(?,?,?)',(uid(),body.project_id,'Edit applied to '+s['name']+'.'))
    return {'status':'applied','operation':op,'stem_id':s['id'],'message':'Edit applied to '+s['name']+'.'}

@app.get('/api/projects/{pid}/messages')
def messages(pid: str,authorization: str | None=Header(None)):
    auth(pid,authorization)
    with connect() as db: return [dict(r) for r in db.execute('SELECT id,text,created_at FROM messages WHERE project_id=? ORDER BY rowid',(pid,))]
