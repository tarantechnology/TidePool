import hmac
import json
import math
import os
import secrets
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, StrictBool, model_validator
from .core import MAX_UPLOAD, connect, digest, event, init, max_passes, probe, project_dir, run, stem, uid
from .guidance import Guidance, SoundPrompt, EditScope, Interval, validate_timing, options, description
from .commands import parse
from .agent import AgentRequest, AgentError, plan as agent_plan
from .worker import enqueue, loop, stop
from .refine import is_residual
from .visual import UnfixableTimeline, file_sha256, prepare_payload, reject_media_refs, validate_timeline

@asynccontextmanager
async def lifespan(app):
    init(); stop.clear(); thread=threading.Thread(target=loop,daemon=True); thread.start()
    yield
    stop.set(); thread.join(timeout=2)
app=FastAPI(title='SoundCanvas API',lifespan=lifespan)
_preview_lock = threading.Lock()

def build_preview(pid, original):
    """Browser-compatible H.264 preview. yuv420p needs even dimensions, so round the
    odd ones down while keeping the aspect ratio."""
    path=project_dir(pid)/'extracted'/'preview.mp4'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(uid()+'.mp4')
    try:
        run('ffmpeg','-v','error','-y','-i',str(original),'-map','0:v:0','-an',
            '-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(temp))
        temp.replace(path)
    except RuntimeError:
        temp.unlink(missing_ok=True); raise
    return path

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
    d=dict(row); d.pop('payload')
    # Process jobs report 'splitting' or 'separating' while the audio service runs.
    if kind not in ('vision','process'): d.pop('stage',None)
    output=json.loads(d.pop('output')) if d['output'] else None
    d['retryable']=None if d.get('retryable') is None else bool(d['retryable'])
    if kind=='process':
        d['stems'] = output.get('stems', []) if isinstance(output, dict) else output or []
        d['inference'] = output.get('inference') if isinstance(output, dict) else None
        d['routing'] = (d['inference'] or {}).get('routing')
    elif kind=='export': d['download_url']=f'/api/exports/{jid}/download' if output else None
    else: d['result']=output
    if kind=='vision':
        d['stage']=d.get('stage') or ('completed' if d['status']=='completed' else 'queued')
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
    mute_ranges: list[tuple[float, float]] | None=Field(default=None,max_length=100)
    @model_validator(mode='after')
    def ordered(self):
        if self.automation and any(a.time>=b.time for a,b in zip(self.automation,self.automation[1:])): raise ValueError('Keyframe times must strictly increase')
        if self.mute_ranges and any(not math.isfinite(a) or not math.isfinite(b) or a<0 or b<=a for a,b in self.mute_ranges): raise ValueError('Mute ranges must have finite, ordered nonnegative times')
        return self
def max_prompts():
    """Mirrors the audio service's AUDIO_MAX_PROMPTS so over-limit jobs fail fast."""
    try: return max(1,int(os.getenv('MAX_PROMPTS','8')))
    except ValueError: return 8
class Process(BaseModel):
    prompts: list[str | SoundPrompt]=Field(default_factory=list,max_length=1000)
    visual_prompt: dict | None=None
    guidance: Guidance=Field(default_factory=Guidance)
    edit_scope: EditScope=Field(default_factory=EditScope)
    profile: Literal["preview", "quality"]="preview"
    append: bool=False
    reranking_candidates: int=Field(default=1,ge=1,le=4)
    passes: int | None=None
    # auto lets the audio service route (song -> music stems, otherwise the prompted model).
    mode: Literal['auto','prompted','music_stems']='auto'
    stems: list[str] | None=Field(default=None,min_length=1,max_length=16)
    @model_validator(mode='after')
    def within_prompt_cap(self):
        if len(self.prompts)>max_prompts(): raise ValueError(f'At most {max_prompts()} prompts per job')
        check_passes(self.passes)
        if self.stems is not None:
            if self.mode!='music_stems': raise ValueError('stems requires mode music_stems')
            self.stems=[s.strip().lower() for s in self.stems]
            if any(not s or len(s)>40 for s in self.stems) or len(set(self.stems))!=len(self.stems): raise ValueError('stems must be distinct nonempty names')
        if self.mode=='music_stems' and self.prompts: raise ValueError('Use stems, not prompts, with mode music_stems')
        return self

def check_passes(passes):
    if passes is not None and not 1<=passes<=max_passes(): raise ValueError(f'passes must be between 1 and {max_passes()}')

def with_passes(opts, passes):
    """Forward passes only when asked; the audio service defaults to one."""
    return opts if passes is None else {**opts,'passes':passes}

class ObjectSelection(BaseModel):
    analysis_id: str = Field(min_length=1,max_length=100)
    detection_id: str = Field(min_length=1,max_length=100)
    track_id: int = Field(ge=0)
    frame_index: int = Field(ge=0)

class CandidateRequest(BaseModel):
    object_selection: ObjectSelection | None = None
    processing_interval: Interval | None = None
    processing_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    stem_id: str | None=None
    source_stem_id: str | None=Field(default=None,max_length=100)
    guidance: Guidance=Field(default_factory=Guidance)
    edit_scope: EditScope=Field(default_factory=EditScope)
    profile: Literal["preview", "quality"]="preview"
    visual_prompt: dict | None=None
    prompt: str=Field(min_length=1,max_length=200)
    reranking_candidates: int=Field(default=1,ge=1,le=4)
    passes: int | None=None
    @model_validator(mode='after')
    def passes_in_range(self):
        check_passes(self.passes)
        return self

class RefineRequest(BaseModel):
    passes: int | None=None
    @model_validator(mode='after')
    def passes_in_range(self):
        check_passes(self.passes)
        return self

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
    if ext not in ('.mp4','.mov','.wav','.mp3','.flac','.m4a'): raise HTTPException(415,'Supported extensions: mp4, mov, wav, mp3, flac, m4a')
    path=project_dir(pid)/'original'/(uid()+ext); audio=project_dir(pid)/'extracted'/'audio.wav'
    size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_UPLOAD: raise HTTPException(413,'Upload too large')
                out.write(chunk)
        info=probe(path)
        info.update(video_id=pid, source_sha256=file_sha256(path))
        if ext in ('.mp4','.mov') and not info['has_video']: raise ValueError('Expected a video stream')
        run('ffmpeg','-v','error','-y','-i',str(path),'-vn','-ac','2','-ar','48000','-c:a','pcm_f32le',str(audio))
        if info['has_video']:
            try: build_preview(pid,path)
            except RuntimeError as exc: event('preview_failed',project_id=pid,error=str(exc))
        from .media import inspect_audio
        info['audio_quality']=inspect_audio(audio)
        if info['channels']>2: info['audio_quality']['findings'].append('Processing downmixes this multichannel source to stereo; the original upload is preserved.')
        if info['has_video']:
            try: validate_timeline(path,info['frame_rate'])
            except (ValueError,RuntimeError,TypeError) as exc: info['audio_quality']['findings'].append(str(exc))
        with connect() as db: db.execute("UPDATE projects SET filename=?,original=?,uploaded_original=?,audio=?,metadata=?,status='uploaded' WHERE id=?",(Path(file.filename or '').name,str(path),str(path),str(audio),json.dumps(info),pid))
    except HTTPException:
        path.unlink(missing_ok=True); audio.unlink(missing_ok=True); raise
    except (ValueError,RuntimeError) as e:
        path.unlink(missing_ok=True); audio.unlink(missing_ok=True); raise HTTPException(422,str(e))
    finally: await file.close()
    return {'project_id':pid,'filename':Path(file.filename or '').name,'metadata':info,'status':'uploaded'}

@app.get('/api/projects/{pid}')
def project(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    with connect() as db:
        latest=db.execute("SELECT id,status FROM jobs WHERE project_id=? AND kind='process' ORDER BY rowid DESC LIMIT 1",(pid,)).fetchone()
        completed=db.execute("SELECT id FROM jobs WHERE project_id=? AND kind='process' AND status='completed' ORDER BY rowid DESC LIMIT 1",(pid,)).fetchone()
        visual_job=db.execute("SELECT id,status FROM jobs WHERE project_id=? AND kind='vision' ORDER BY rowid DESC LIMIT 1",(pid,)).fetchone()
        history=db.execute("SELECT h.candidate_id,h.applied_revisions FROM candidate_history h JOIN jobs j ON j.id=h.candidate_id WHERE j.project_id=? AND h.undone=0 ORDER BY h.rowid DESC LIMIT 1",(pid,)).fetchone()
    current_stems=list_stems(pid,authorization)
    if not current_stems: completed=None
    undoable=history['candidate_id'] if history and {r['id']:r['revision'] for r in current_stems}==json.loads(history['applied_revisions']) else None
    return {'latest_undoable_candidate_id':undoable,'project_id':pid,'status':p['status'],'filename':p['filename'],'metadata':json.loads(p['metadata']) if p['metadata'] else None,'visual_metadata':json.loads(p['visual']) if p['visual'] else None,'stems':current_stems,'latest_job':dict(latest) if latest else None,'last_completed_job_id':completed['id'] if completed else None,'latest_visual_job':dict(visual_job) if visual_job else None}

@app.get('/api/projects/{pid}/original')
def original(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['original']: raise HTTPException(404,'Project has no uploaded media')
    return FileResponse(p['original'],filename=p['filename'],content_disposition_type='inline')

@app.get('/api/projects/{pid}/uploaded-original')
def uploaded_original(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    path=p['uploaded_original'] or p['original']
    if not path: raise HTTPException(404,'Project has no uploaded media')
    return FileResponse(path,filename=p['filename'],content_disposition_type='inline')

@app.get('/api/projects/{pid}/original-audio')
def original_audio(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(404,'Project has no extracted audio')
    return FileResponse(p['audio'],media_type='audio/wav')

@app.get('/api/projects/{pid}/preview')
def preview(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['original'] or not json.loads(p['metadata'])['has_video']: raise HTTPException(404,'Project has no video')
    path=project_dir(pid)/'extracted'/'preview.mp4'
    if not path.exists():
        # Projects uploaded before previews were built at upload time, and retries after a
        # failed upload transcode. One process owns the media cache; avoid duplicate work.
        with _preview_lock:
            if not path.exists():
                try: build_preview(pid,p['original'])
                except RuntimeError as exc: raise HTTPException(422,str(exc)) from exc
    return FileResponse(path,media_type='video/mp4',content_disposition_type='inline')

@app.get('/api/projects/{pid}/visual-metadata')
def get_visual(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['visual']: raise HTTPException(404,'No visual analysis available')
    return json.loads(p['visual'])

class ObjectAnalysisRequest(BaseModel):
    concept: str = Field(default='auto',min_length=1,max_length=80)

@app.post('/api/projects/{pid}/visual-analyze',status_code=202)
def analyze_visual(pid: str, body: ObjectAnalysisRequest = ObjectAnalysisRequest(), authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['original'] or not json.loads(p['metadata'])['has_video']: raise HTTPException(409,'Upload a video first')
    from .detector import status
    vision=status()
    if not vision['available']:
        return JSONResponse({'detail':vision['reason'],'code':'vision_unavailable'},status_code=502 if vision.get('retryable') else 503)
    if vision.get('selection_only'):
        if not body.concept.strip(): raise HTTPException(422,'Enter an object to find.')
        if body.concept.strip()=='auto' and not vision.get('automatic'):
            raise HTTPException(503,'Automatic object discovery is unavailable. Find a specific object with SAM instead.')
        if body.concept.strip()!='auto' and vision.get('prompted') is False:
            raise HTTPException(503,'SAM object refinement is unavailable; automatic object discovery is still available.')
        job=start(pid,'vision',{'object_selection':True,'concept':body.concept.strip(), 'source_sha256':file_sha256(p['original'])})
        return {'job_id':job,'status':'queued','prepared':False}
    prepare=False
    try: validate_timeline(p['original'], json.loads(p['metadata'])['frame_rate'])
    except UnfixableTimeline as exc: raise HTTPException(422,str(exc)) from exc
    except ValueError as exc:
        # Phone recordings (VFR, offset starts, rotation) get a compatible copy first.
        with connect() as db:
            if db.execute('SELECT 1 FROM stems WHERE project_id=? LIMIT 1',(pid,)).fetchone():
                raise HTTPException(409,f'{exc} Object detection needs a compatible video copy, which can only be prepared before separating sounds; existing layers were preserved.') from exc
        prepare=True
    except (RuntimeError, KeyError, TypeError) as exc: raise HTTPException(422,str(exc)) from exc
    try: job=enqueue(pid,'vision',{'prepare':prepare},stage='queued')
    except sqlite3.IntegrityError: raise HTTPException(409,'A job of this type is already active')
    return {'job_id':job,'status':'queued','prepared':prepare}

@app.post('/api/projects/{pid}/prepare-visual',status_code=202)
def prepare_visual(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['original'] or not json.loads(p['metadata'])['has_video']: raise HTTPException(409,'Upload a video first')
    with connect() as db:
        if db.execute('SELECT 1 FROM stems WHERE project_id=? LIMIT 1',(pid,)).fetchone(): raise HTTPException(409,'Prepare a compatible video copy before separating sounds, to preserve existing layers.')
    return {'job_id':start(pid,'normalize',{}),'status':'queued'}

@app.get('/api/prepare-jobs/{jid}')
def prepare_job(jid: str, authorization: str | None=Header(None)):
    return get_job(jid,'normalize',authorization)

@app.get('/api/visual-jobs/{jid}')
def visual_job(jid: str, authorization: str | None=Header(None)):
    return get_job(jid,'vision',authorization)

@app.put('/api/projects/{pid}/visual-metadata')
def visual(pid: str, body: dict, authorization: str | None=Header(None)):
    auth(pid,authorization)
    try: reject_media_refs(body)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    with connect() as db: db.execute('UPDATE projects SET visual=? WHERE id=?',(json.dumps(body),pid))
    return {'status':'saved'}

def refuse_during_refine(pid):
    """A refine job rewrites the target and residual files; queued work must not snapshot them."""
    with connect() as db:
        if db.execute("SELECT 1 FROM jobs WHERE project_id=? AND kind='refine' AND status IN ('queued','processing')",(pid,)).fetchone():
            raise HTTPException(409,'Wait for the active refine job to finish.')

def start(pid,kind,payload):
    try: return enqueue(pid,kind,payload)
    except sqlite3.IntegrityError: raise HTTPException(409,'A job of this type is already active')

@app.post('/api/projects/{pid}/discover',status_code=202)
def discover_sounds(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio'] or not p['metadata']: raise HTTPException(409,'Upload audio first')
    return {'job_id':start(pid,'discovery',{'audio_sha256':file_sha256(p['audio'])}),'status':'queued'}

@app.get('/api/discovery-jobs/{jid}')
def discovery_job(jid: str, authorization: str | None=Header(None)):
    return get_job(jid,'discovery',authorization)

@app.post('/api/projects/{pid}/agent')
def project_agent(pid: str, body: AgentRequest, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    with connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
    body.has_project_stems=bool(rows)
    # A music split's canonical label and instrument tags let "remove the piano" find the "other" track.
    body.stems=[{'id':r['id'],'label':r['name'],'duration':r['duration'],'canonical_label':r['label'],'tags':[t['label'] for t in json.loads(r['tags'])] if r['tags'] else None} for r in rows]
    body.tracks=[{'stemId':r['id'],'gain':r['volume'],'muted':bool(r['muted']),'solo':bool(r['solo']),'automation':[{'t':k['time'],'value':k['value']} for k in json.loads(r['automation'])]} for r in rows]
    if p['metadata']: body.duration=json.loads(p['metadata'])['duration']
    # Never let a replacement instruction fall through to a separation-only planner.
    import re
    if re.search(r'\b(?:replace|replacing|swap|substitute)\b',body.prompt,re.I) or re.search(r'\b(?:make|turn|change)\b.+\b(?:sound\s+like|into)\b',body.prompt,re.I):
        return {'kind':'clarify','description':None,'remove':False,'actions':[],'message':'To replace a sound, refresh the editor and use the sound replacement workflow.'}
    # A browser tab left open across an editor update may still send sound
    # generation requests to this older planning route. Never let its generic
    # planner incorrectly say that the configured generator cannot add sounds.
    if re.search(r'\b(?:add|play)\b',body.prompt,re.I):
        from .replacement import parse_replacement
        try: addition=parse_replacement(body.prompt,body.duration)
        except ValueError: addition=None
        if addition and addition.get('mode')=='add':
            return {'kind':'clarify','description':None,'remove':False,'actions':[], 'message':'Refresh the editor to enable this sound-generation request, then submit it again.'}
    from .effect_group_commands import plan_group_command
    group_plan=plan_group_command(body.prompt,rows)
    if group_plan is not None:return group_plan
    parsed=parse(body.prompt)
    if parsed and parsed.get('stem_name'):
        name=parsed['stem_name'].lower()
        matches=[r for r in rows if name in r['name'].lower()]
        if len(matches)>1:
            return {'kind':'clarify','description':None,'remove':False,'actions':[], 'message':'Choose a layer: '+', '.join(r['name'] for r in matches)}
        if parsed['action']=='mute_range':
            if parsed['end']>body.duration: raise HTTPException(422,'Edit interval extends beyond the audio timeline')
            if matches:
                return {'kind':'edit','description':None,'remove':False,'actions':[{'type':'mute_range','stemId':matches[0]['id'],'start':parsed['start'],'end':parsed['end']}],'message':'Timed mute ready.'}
            return {'kind':'separate','description':description(name),'remove':True,'actions':[], 'guidance':{'anchors':[]},'edit_scope':{'action':'mute','interval':{'start':parsed['start'],'end':parsed['end']}},'message':'Preview the timed mute before applying it.'}
    try: return agent_plan(body)
    except AgentError as exc: raise HTTPException(502,str(exc)) from exc

@app.get('/api/projects/{pid}/capabilities')
def service_capabilities(pid: str, authorization: str | None=Header(None)):
    auth(pid,authorization)
    from .media import capabilities
    from .detector import status
    return {**capabilities(),'vision':status()}

def separation_options(profile, candidates):
    """Select request options from the separator actually serving this app."""
    from .media import capabilities
    caps = capabilities()
    if profile == 'quality' and not caps.get('quality'):
        raise HTTPException(422, caps.get('reason') or 'Quality retry is unavailable for this separator')
    return options(profile, candidates, backend=caps.get('backend'))

def source_kind(p):
    """Routing tie-breaker for the audio service: an upload without a video stream is likely a song."""
    return 'video' if p['metadata'] and json.loads(p['metadata']).get('has_video') else 'audio'

@app.get('/api/projects/{pid}/route')
def route(pid: str, authorization: str | None=Header(None)):
    """Dry run of the audio service's router, so the editor can suggest a split. Separates nothing."""
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    from .media import InferenceError, route as route_audio
    try: return route_audio(p['audio'],source_kind(p))
    except (InferenceError,RuntimeError,ValueError) as exc: raise HTTPException(502,str(exc)) from exc

@app.post('/api/projects/{pid}/process',status_code=202)
def process(pid: str, body: Process=Process(), authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    refuse_during_refine(pid)
    # With no prompts, auto lets the service decide (a song becomes music stems); prompted needs one.
    if os.getenv('INFERENCE_ADAPTER', 'http') == 'http' and ((not body.prompts and body.mode=='prompted') or any(not (prompt if isinstance(prompt,str) else prompt.text).strip() for prompt in body.prompts)):
        raise HTTPException(422, 'At least one nonempty audio prompt is required for HTTP inference')
    if body.append and body.mode=='music_stems': raise HTTPException(422,'A music split replaces the layers; it cannot be added to an earlier separation.')
    try: validate_timing(body.prompts,body.guidance,body.edit_scope,probe(p['audio'])['duration'])
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    targets=len(body.stems or []) if body.mode=='music_stems' else len(body.prompts)
    if body.edit_scope.action != 'keep' and targets!=1: raise HTTPException(422,'An edit requires exactly one sound target')
    payload=body.model_dump(exclude={'append','reranking_candidates','guidance','profile','passes','mode','stems'})
    payload['anchors']=body.guidance.model_dump()['anchors']
    payload['options']=with_passes(separation_options(body.profile,body.reranking_candidates),body.passes)
    # Only an explicit mode overrides the service's AUDIO_ROUTING_DEFAULT.
    if 'mode' in body.model_fields_set: payload['options']['mode']=body.mode
    if body.stems: payload['options']['stems']=body.stems
    payload['hints']={'source_kind':source_kind(p)}
    with connect() as db: payload['_baseline']={r['id']:r['revision'] for r in db.execute('SELECT id,revision FROM stems WHERE project_id=?',(pid,))}
    if body.append:
        with connect() as db: rows=db.execute("SELECT id,revision,start_offset FROM stems WHERE project_id=? AND lower(name)='residual'",(pid,)).fetchall()
        if len(rows)!=1: raise HTTPException(409,'Adding a sound requires exactly one residual stem from an earlier separation.')
        if rows[0]['start_offset'] != 0: raise HTTPException(409,'Reset the residual start offset to zero before adding another sound.')
        payload['_append']={'id':rows[0]['id'],'revision':rows[0]['revision']}
    try: payload=prepare_payload(p, payload)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc: raise HTTPException(422, str(exc)) from exc
    return {'job_id':start(pid,'process',payload),'status':'queued'}

@app.post('/api/projects/{pid}/candidates',status_code=202)
def create_candidate(pid: str, body: CandidateRequest, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    object_context=None
    if body.object_selection:
        if body.visual_prompt: raise HTTPException(422,'Choose object context or audio visual conditioning, not both.')
        from .object_selection import validate_context
        try: object_context=validate_context(p,body.object_selection.model_dump())
        except (ValueError,KeyError,TypeError,OSError) as exc: raise HTTPException(422,str(exc)) from exc
    refuse_during_refine(pid)
    with connect() as db: rows=db.execute('SELECT * FROM stems WHERE project_id=?',(pid,)).fetchall()
    if body.stem_id and body.source_stem_id:raise HTTPException(422,'Choose either retry or a source layer.')
    if body.source_stem_id and (object_context or body.visual_prompt):raise HTTPException(422,'Object guidance is not available for an individual layer.')
    if (object_context and not body.stem_id and not body.source_stem_id and body.edit_scope.action in ('mute','lower')
            and body.edit_scope.interval is None and not body.processing_interval and body.processing_time is None
            and body.profile=='preview' and body.passes in (None,1) and not body.guidance.anchors):
        matches=[r for r in rows if r['role']=='target' and (r['prompt'] or '').strip().casefold()==description(body.prompt).casefold()]
        if len(matches)==1:
            snapshot={'rows':[dict(r) for r in rows],'source':p['audio'],'source_sha256':file_sha256(p['audio']),
                      'project_audio':p['audio'],'target_id':matches[0]['id'],'residual_id':None,
                      'stem_hashes':{r['id']:file_sha256(r['path']) for r in rows}}
            payload={'_candidate':snapshot,'_reuse_target_id':matches[0]['id'],'_object_context':object_context,
                     'edit_scope':body.edit_scope.model_dump(),'prompts':[description(body.prompt)]}
            return {'candidate_id':start(pid,'candidate',payload),'status':'queued'}
    selected=next((r for r in rows if r['id']==body.source_stem_id),None)
    if body.source_stem_id and not selected:raise HTTPException(404,'Source layer not found in this project.')
    targets=[r for r in rows if r['id']==body.stem_id and r['name'].lower()!='residual']
    residuals=[r for r in rows if r['name'].lower()=='residual']
    # Creative edits can add effect layers to an otherwise untouched original
    # without creating a separation residual. Split that original as a selected
    # layer, preserving its mute ranges and the independently editable effects.
    promote_original=False
    if not body.stem_id and not selected and not residuals and not body.visual_prompt and not object_context:
        originals=[r for r in rows if r['name'].casefold()=='original audio' and r['path']==p['audio'] and r['role'] is None]
        if len(originals)==1 and all(r['id']==originals[0]['id'] or r['role']=='effect' for r in rows):
            selected=originals[0]
            promote_original=True
    if body.stem_id and (len(rows)!=2 or len(targets)!=1 or len(residuals)!=1):
        raise HTTPException(409,'Replacing an older target in a multitrack project requires its earlier source snapshot. Preview a new target from the current residual instead.')
    if not body.stem_id and not selected and rows and len(residuals)!=1:
        raise HTTPException(409,'Choose the layer to search in before separating another sound.')
    if not selected and any(r['start_offset'] != 0 for r in targets+residuals):
        raise HTTPException(409,'Preview requires the source layers to start at zero.')
    source=selected['path'] if selected else (p['audio'] if body.stem_id or not rows else residuals[0]['path'])
    source_offset=float(selected['start_offset']) if selected else 0.0
    try: validate_timing([],body.guidance,body.edit_scope,probe(p['audio'])['duration'] if selected else probe(source)['duration'])
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    if body.processing_interval and body.processing_time is not None: raise HTTPException(422, 'Choose a range or a timestamp, not both.')
    duration=probe(source)['duration']
    if body.processing_time is not None and not source_offset<=body.processing_time<(source_offset+duration if selected else source_offset+duration+1e-9): raise HTTPException(422, 'Timestamp is outside the selected source layer.')
    if body.processing_interval:
        if body.processing_interval.end-body.processing_interval.start > 600: raise HTTPException(422, 'Choose a processing interval of at most 10 minutes.')
        if body.processing_interval.start < source_offset-1e-6 or body.processing_interval.end > source_offset+duration+1e-6: raise HTTPException(422, 'Selection extends beyond the selected source layer.')
    if (body.processing_interval or body.processing_time is not None) and body.visual_prompt: raise HTTPException(422, 'Visual guidance is not available for a selected audio range.')
    sound=description(body.prompt)
    if not sound or sound=='residual': raise HTTPException(422,'Enter a sound description; residual is a reserved layer name.')
    snapshot={'rows':[dict(r) for r in rows], 'source':source,
              'source_sha256':file_sha256(source), 'project_audio':p['audio'], 'target_id':body.stem_id,
              'residual_id':selected['id'] if selected else (residuals[0]['id'] if residuals else None),
              'promote_original':promote_original,
              **({'source_layer_id':selected['id'],'source_offset':source_offset} if selected else {})}
    payload={'prompts':[sound],'visual_prompt':body.visual_prompt or {},
             'anchors':body.guidance.model_dump()['anchors'], 'edit_scope':body.edit_scope.model_dump(),
             'profile':body.profile, 'options':{**with_passes(separation_options(body.profile,body.reranking_candidates),body.passes),'seed':secrets.randbelow(2**31)},'hints':{'source_kind':source_kind(p)},
             '_candidate':snapshot, '_object_context':object_context, '_processing_interval':({'start':body.processing_interval.start-source_offset,'end':body.processing_interval.end-source_offset} if body.processing_interval else None), '_processing_time':(body.processing_time-source_offset if body.processing_time is not None else None)}
    if selected:
        payload['options']['mode']='prompted'
        translated=[]
        for anchor in payload['anchors']:
            if anchor['start']<source_offset or anchor['end']>source_offset+duration:raise HTTPException(422,'Guidance must lie within the selected layer.')
            translated.append({**anchor,'start':anchor['start']-source_offset,'end':anchor['end']-source_offset})
        payload['anchors']=translated
    try: payload=prepare_payload(p,payload)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc: raise HTTPException(422,str(exc)) from exc
    return {'candidate_id':start(pid,'candidate',payload),'status':'queued'}



class GenerateEffectRequest(BaseModel):
    prompt: str = Field(min_length=1,max_length=300)

@app.post('/api/projects/{pid}/sound-effects/generate',status_code=202)
def generate_effect(pid: str, body: GenerateEffectRequest, authorization: str | None=Header(None)):
    auth(pid,authorization)
    if not os.getenv('SOUNDCANVAS_SFX_CLI','').strip(): raise HTTPException(503,'Sound generation is unavailable; upload an effect instead.')
    return {'job_id':start(pid,'sfx_generation',body.model_dump()),'status':'queued'}

@app.get('/api/sound-effect-jobs/{jid}')
def effect_job(jid: str, authorization: str | None=Header(None)):
    return get_job(jid,'sfx_generation',authorization)

@app.post('/api/projects/{pid}/sound-effects/upload')
async def upload_effect(pid: str, file: UploadFile=File(...), authorization: str | None=Header(None)):
    auth(pid,authorization)
    directory=project_dir(pid)/'effects';directory.mkdir(parents=True,exist_ok=True)
    aid=uid();temp=directory/(aid+'.upload');out=directory/(aid+'.wav');size=0
    try:
        with temp.open('wb') as target:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>20*1024*1024: raise HTTPException(413,'Sound effect must be under 20 MB.')
                target.write(chunk)
        info=probe(temp)
        if not 0<info['duration']<=15: raise HTTPException(422,'Sound effect must be between 0 and 15 seconds.')
        run('ffmpeg','-v','error','-y','-i',str(temp),'-vn','-ar','48000','-ac','2','-c:a','pcm_f32le',str(out))
        return {'asset_id':aid,'duration':probe(out)['duration'],'source':'uploaded'}
    except HTTPException:
        out.unlink(missing_ok=True);raise
    except (RuntimeError,ValueError,KeyError) as exc:
        out.unlink(missing_ok=True);raise HTTPException(422,'Could not read that sound effect.') from exc
    finally: temp.unlink(missing_ok=True);await file.close()

@app.get('/api/projects/{pid}/sound-effects/{aid}/audio')
def effect_audio(pid: str, aid: str, authorization: str | None=Header(None)):
    auth(pid,authorization)
    from .replacement import asset_path
    try: path=asset_path(pid,aid)
    except ValueError as exc: raise HTTPException(404,'Sound effect not found.') from exc
    return FileResponse(path,media_type='audio/wav')

class ReplacementRequest(BaseModel):
    asset_id: str = Field(min_length=32,max_length=32)
    effect_label: str = Field(min_length=1,max_length=100)
    source_prompt: str = Field(min_length=1,max_length=200)
    source_stem_id: str | None = Field(default=None,max_length=100)
    mode: Literal['replace','add']='replace'
    intervals: list[Interval] = Field(min_length=1,max_length=100)
    @model_validator(mode='after')
    def ordered(self):
        if self.mode=='add' and self.source_stem_id:raise ValueError('A source layer is only used when replacing a sound.')
        if self.mode=='replace' and len(self.intervals)>20: raise ValueError('At most 20 separation ranges are supported.')
        for a,b in zip(self.intervals,self.intervals[1:]):
            if a.end>b.start: raise ValueError('Intervals must be ordered and not overlap.')
        if any(i.end-i.start>10 for i in self.intervals): raise ValueError('Each sound selection must be at most 10 seconds.')
        return self

@app.post('/api/projects/{pid}/replacement-preview',status_code=202)
def replacement_preview(pid: str, body: ReplacementRequest, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first.')
    refuse_during_refine(pid)
    from .candidates import safe_path
    from .replacement import asset_path
    try:
        source=safe_path(pid,p['audio']);effect=asset_path(pid,body.asset_id)
        if any(i.end>probe(source)['duration'] for i in body.intervals): raise HTTPException(422,'Select times within the clip.')
        with connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
        selected=next((r for r in rows if r['id']==body.source_stem_id),None)
        if body.source_stem_id and not selected:raise HTTPException(404,'Source layer not found in this project.')
        if selected and any(i.start<selected['start_offset'] or i.end>selected['start_offset']+selected['duration']+1/48000 for i in body.intervals):raise HTTPException(422,'Select times within the chosen source layer.')
        if body.mode=='replace' and not selected and rows and (len([r for r in rows if r['name'].lower()=='residual' and r['start_offset']==0])!=1): raise HTTPException(409,'Replacement needs one unshifted residual layer. Choose a source layer or use a fresh project.')
        snapshot={'rows':rows,'source':str(source),'source_sha256':file_sha256(source),'project_audio':p['audio'],'stem_hashes':{r['id']:file_sha256(safe_path(pid,r['path'])) for r in rows}}
    except (ValueError,OSError) as exc: raise HTTPException(422,'Project audio or sound effect is unavailable.') from exc
    return {'candidate_id':start(pid,'candidate',{'_replacement':True,'_candidate':snapshot,'asset_sha256':file_sha256(effect),**body.model_dump()}),'status':'queued'}

class AutoReplacementRequest(BaseModel):
    request: str = Field(min_length=1,max_length=500)
    source_stem_id: str | None = Field(default=None,max_length=100)

@app.post('/api/projects/{pid}/automatic-replacement',status_code=202)
def automatic_replacement(pid: str, body: AutoReplacementRequest, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first.')
    if not os.getenv('SOUNDCANVAS_SFX_CLI','').strip(): raise HTTPException(503,'Sound generation is unavailable; use an uploaded effect instead.')
    refuse_during_refine(pid)
    from .candidates import safe_path
    from .replacement import parse_replacement
    from .effect_followup import matching_effects
    from .layer_scope import extract_layer_scope,resolve_layer
    try:
        source=safe_path(pid,p['audio'])
        with connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
        request,layer_name=extract_layer_scope(body.request)
        selected=resolve_layer(layer_name,rows) if layer_name else next((r for r in rows if r['id']==body.source_stem_id),None)
        if body.source_stem_id and not selected:raise HTTPException(404,'Source layer not found in this project.')
        if body.source_stem_id and selected and selected['id']!=body.source_stem_id:raise HTTPException(422,'The named layer and selected layer do not match.')
        plan=parse_replacement(request,probe(source)['duration'])
        visual_snapshot={}
        if plan.get('object_track_id') is not None:
            from .object_selection import appearances
            found=appearances(p,plan['object_track_id'],plan['object_label'])
            visual_snapshot={'visual_analysis_id':found['analysis_id'],'visual_source_sha256':found['source_sha256']}
        if selected and plan.get('mode')=='add':raise HTTPException(422,'A source layer is only used when replacing a sound.')
        if selected and plan.get('time') is not None and not selected['start_offset']<=plan['time']<selected['start_offset']+selected['duration']:raise HTTPException(422,'The timestamp is outside the selected source layer.')
        known=matching_effects(rows,plan['source']) if plan.get('mode')!='add' else []
        if plan.get('mode')!='add' and not selected and not known and rows and not any(r.get('role')=='effect' for r in rows) and len([r for r in rows if r['name'].lower()=='residual' and r['start_offset']==0])!=1: raise HTTPException(409,'I need one unshifted residual layer to replace sounds. Choose a source layer or try a fresh project.')
        snapshot={'rows':rows,'source':str(source),'source_sha256':file_sha256(source),'project_audio':p['audio'],'stem_hashes':{r['id']:file_sha256(safe_path(pid,r['path'])) for r in rows},**visual_snapshot}
    except (ValueError,OSError) as exc: raise HTTPException(422,str(exc)) from exc
    return {'candidate_id':start(pid,'candidate',{'_auto_replacement':True,'_candidate':snapshot,'request':request,'original_request':body.request,'source_stem_id':selected['id'] if selected else None}),'status':'queued'}

class LayerEffectsRequest(BaseModel):
    stem_id: str | None = Field(default=None,max_length=100)
    effect_group_id: str | None = Field(default=None,max_length=100)
    bass: float = Field(default=0,ge=-12,le=12,allow_inf_nan=False)
    treble: float = Field(default=0,ge=-12,le=12,allow_inf_nan=False)
    highpass: float = Field(default=0,ge=0,le=200,allow_inf_nan=False)
    compression: float = Field(default=1,ge=1,le=6,allow_inf_nan=False)

class EffectClipEdit(BaseModel):
    stem_id: str = Field(min_length=1,max_length=100)
    expected_revision: int = Field(ge=0)
    operation: Literal['move','delete']
    start_offset: float | None = Field(default=None,ge=0,le=86400,allow_inf_nan=False)
    @model_validator(mode='after')
    def fields_match(self):
        if self.operation=='move' and self.start_offset is None:raise ValueError('Choose a new position.')
        if self.operation=='delete' and self.start_offset is not None:raise ValueError('Deletion does not take a position.')
        return self

@app.post('/api/projects/{pid}/effect-clip-edit',status_code=202)
def effect_clip_edit(pid: str,body: EffectClipEdit,authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']:raise HTTPException(409,'Upload media first.')
    refuse_during_refine(pid)
    from .candidates import safe_path
    with connect() as db:rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
    item=next((r for r in rows if r['id']==body.stem_id and r.get('role')=='effect'),None)
    if not item:raise HTTPException(404,'Effect clip not found.')
    if item['revision']!=body.expected_revision:raise HTTPException(409,'The clip changed; reload and try again.')
    try:
        source=safe_path(pid,p['audio']);snapshot={'rows':rows,'source':str(source),'source_sha256':file_sha256(source),'project_audio':p['audio'],'stem_hashes':{r['id']:file_sha256(safe_path(pid,r['path'])) for r in rows}}
        if body.start_offset is not None and body.start_offset>=probe(source)['duration']:raise HTTPException(422,'Move the clip within the project.')
    except (ValueError,OSError) as exc:raise HTTPException(409,'Project audio is unavailable; reload and try again.') from exc
    return {'candidate_id':start(pid,'candidate',{'_clip_edit':True,'_candidate':snapshot,**body.model_dump()}),'status':'queued'}

@app.post('/api/projects/{pid}/effects-preview',status_code=202)
def effects_preview(pid: str, body: LayerEffectsRequest, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    refuse_during_refine(pid)
    from .candidates import safe_path
    with connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
    if body.effect_group_id and body.stem_id: raise HTTPException(422,'Choose a layer or a group, not both.')
    if body.effect_group_id and not any((r['role']=='effect' and r['effect_group_id']==body.effect_group_id) or (r['role']=='target' and r.get('sound_group_id')==body.effect_group_id) for r in rows):
        raise HTTPException(422,'Select a sound group in this project.')
    if not body.effect_group_id and ((rows and not any(r['id']==body.stem_id for r in rows)) or (not rows and body.stem_id is not None)):
        raise HTTPException(422,'Select a layer in this project.')
    try:
        source=safe_path(pid,p['audio'])
        snapshot={'rows':rows,'source':str(source),'source_sha256':file_sha256(source),'project_audio':p['audio'],
                  'stem_hashes':{r['id']:file_sha256(safe_path(pid,r['path'])) for r in rows}}
    except (ValueError,OSError) as exc: raise HTTPException(409,'Project audio is unavailable; reload and try again.') from exc
    effects=body.model_dump(exclude={'stem_id','effect_group_id'})
    return {'candidate_id':start(pid,'candidate',{'_effects':True,'_candidate':snapshot,'stem_id':body.stem_id,'effect_group_id':body.effect_group_id,'effects':effects}),'status':'queued'}

class CinematicRequest(BaseModel):
    style: Literal['subtle', 'cinematic', 'dramatic'] = 'cinematic'

@app.post('/api/projects/{pid}/cinematic-preview',status_code=202)
def cinematic_preview(pid: str, body: CinematicRequest = CinematicRequest(), authorization: str | None = Header(None)):
    p = auth(pid, authorization)
    if not p['audio']: raise HTTPException(409, 'Upload media first')
    refuse_during_refine(pid)
    from .candidates import safe_path
    with connect() as db: rows = [dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?', (pid,))]
    try:
        source = safe_path(pid, p['audio'])
        hashes = {r['id']:file_sha256(safe_path(pid, r['path'])) for r in rows}
        snapshot = {'rows':rows, 'source':str(source), 'source_sha256':file_sha256(source),
                    'project_audio':p['audio'], 'stem_hashes':hashes}
    except (ValueError, OSError) as exc: raise HTTPException(409, 'Project audio is unavailable; reload and try again.') from exc
    return {'candidate_id':start(pid,'candidate',{'_cinematic':True, '_candidate':snapshot, 'style':body.style}), 'status':'queued'}

class DirectorRequest(BaseModel):
    request: str = Field(min_length=1,max_length=1000)
    intent: Literal['cinematic','dramatic','cool','funny'] = 'cinematic'
    style: Literal['subtle','cinematic','dramatic'] = 'cinematic'

@app.post('/api/projects/{pid}/censorship-preview',status_code=202)
def censorship_preview(pid: str, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    from .censorship import available
    if not available(): raise HTTPException(503,'Local transcription is not configured. Install faster-whisper and configure a local model.')
    refuse_during_refine(pid)
    from .candidates import safe_path
    with connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
    try:
        source=safe_path(pid,p['audio'])
        snapshot={'rows':rows,'source':str(source),'source_sha256':file_sha256(source),'project_audio':p['audio'],
                  'stem_hashes':{r['id']:file_sha256(safe_path(pid,r['path'])) for r in rows}}
    except (ValueError,OSError) as exc: raise HTTPException(409,'Project audio is unavailable; reload and try again.') from exc
    return {'candidate_id':start(pid,'candidate',{'_censorship':True,'_candidate':snapshot}),'status':'queued'}

@app.post('/api/projects/{pid}/director-preview',status_code=202)
def director_preview(pid: str, body: DirectorRequest, authorization: str | None=Header(None)):
    p=auth(pid,authorization)
    if not p['audio']: raise HTTPException(409,'Upload media first')
    refuse_during_refine(pid)
    from .candidates import safe_path
    with connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?',(pid,))]
    try:
        source=safe_path(pid,p['audio'])
        snapshot={'rows':rows,'source':str(source),'source_sha256':file_sha256(source),'project_audio':p['audio'],
                  'stem_hashes':{r['id']:file_sha256(safe_path(pid,r['path'])) for r in rows}}
    except (ValueError,OSError) as exc: raise HTTPException(409,'Project audio is unavailable; reload and try again.') from exc
    return {'candidate_id':start(pid,'candidate',{'_director':True,'_candidate':snapshot,'style':body.style,'intent':body.intent,'request':body.request}),'status':'queued'}

def candidate_record(cid, authorization):
    with connect() as db: row=db.execute("SELECT * FROM jobs WHERE id=? AND kind='candidate'",(cid,)).fetchone()
    if not row: raise HTTPException(404,'Candidate not found')
    auth(row['project_id'],authorization)
    return row

class DirectorAdjustment(BaseModel):
    stem_id: str = Field(min_length=1,max_length=100)
    enabled: bool

@app.post('/api/candidates/{cid}/director-adjust',status_code=202)
def director_adjust(cid: str, body: DirectorAdjustment, authorization: str | None=Header(None)):
    row=candidate_record(cid,authorization)
    if row['status']!='completed' or not row['output']: raise HTTPException(409,'Preview is not ready.')
    data=json.loads(row['output']); provenance=data.get('provenance') or {}
    if provenance.get('kind')!='director' or not any(c.get('target_stem_id')==body.stem_id for c in provenance.get('changes',[])):
        raise HTTPException(422,'This creative change is unavailable.')
    refuse_during_refine(row['project_id'])
    return {'candidate_id':start(row['project_id'],'candidate',{'_director_adjust':True,'parent_id':cid,'stem_id':body.stem_id,'enabled':body.enabled}), 'status':'queued'}

class ReplacementAdjustment(BaseModel):
    operation: Literal['toggle','move','add','alignment']
    event_id: str | None = Field(default=None,max_length=100)
    enabled: StrictBool | None = None
    time: float | None = Field(default=None,ge=0,le=86400,allow_inf_nan=False)
    alignment: Literal['peak','onset'] | None = None
    @model_validator(mode='after')
    def fields_match(self):
        if self.operation=='toggle' and (not self.event_id or self.enabled is None): raise ValueError('Choose a placement and its state.')
        if self.operation=='move' and (not self.event_id or self.time is None): raise ValueError('Choose a placement and a time.')
        if self.operation=='add' and self.time is None: raise ValueError('Choose a time.')
        if self.operation=='alignment' and self.alignment is None: raise ValueError('Choose an alignment.')
        return self

@app.post('/api/candidates/{cid}/replacement-adjust',status_code=202)
def replacement_adjust(cid: str,body: ReplacementAdjustment,authorization: str | None=Header(None)):
    row=candidate_record(cid,authorization)
    if row['status']!='completed' or not row['output']:raise HTTPException(409,'Preview is not ready.')
    output=json.loads(row['output']);provenance=output.get('provenance') or {}
    if provenance.get('kind')!='replacement' or not output.get('_replacement_edit'):raise HTTPException(422,'This preview cannot be adjusted.')
    if body.event_id and not any(e.get('id')==body.event_id for e in provenance.get('events',[])):raise HTTPException(422,'This placement is unavailable.')
    refuse_during_refine(row['project_id'])
    return {'candidate_id':start(row['project_id'],'candidate',{'_replacement_adjust':True,'parent_id':cid,**body.model_dump()}),'status':'queued'}

@app.get('/api/candidates/{cid}')
def candidate_status(cid: str,authorization: str | None=Header(None)):
    row=candidate_record(cid,authorization); output=json.loads(row['output']) if row['output'] else None
    return {'id':cid,'project_id':row['project_id'],'status':row['status'],'progress':row['progress'],'error':row['error'],
        'stage':row['stage'] if row['status']=='processing' else None,
        'audio_url':f'/api/candidates/{cid}/audio' if output and row['status']=='completed' else None,
        'mix_url':f'/api/candidates/{cid}/audio?part=mix' if output and 'mix' in output else None,
        'original_url':f'/api/candidates/{cid}/audio?part=original' if output and 'original' in output else None,
        'provenance':output.get('provenance') if output else None,'inference':output.get('inference') if output else None,'recovery':output.get('recovery') if output else None}

@app.get('/api/candidates/{cid}/audio')
def candidate_audio(cid: str,part: Literal['target','residual','mix','original']='target',authorization: str | None=Header(None)):
    row=candidate_record(cid,authorization)
    if row['status']!='completed' or not row['output']: raise HTTPException(409,'Candidate is not ready')
    data=json.loads(row['output'])
    if part not in data: raise HTTPException(409,'This older preview has no mix comparison; generate a new preview.')
    path=Path(data[part]['path']).resolve()
    if not path.is_relative_to((project_dir(row['project_id'])/'stems').resolve()) or not path.is_file(): raise HTTPException(404,'Candidate audio not found')
    return FileResponse(path,media_type='audio/wav')

@app.post('/api/candidates/{cid}/apply')
def apply_candidate(cid: str,authorization: str | None=Header(None)):
    row=candidate_record(cid,authorization)
    if row['status']!='completed' or not row['output']: raise HTTPException(409,'Candidate is not ready')
    from .candidates import apply
    return apply(row)

@app.post('/api/candidates/{cid}/undo')
def undo_candidate(cid: str,authorization: str | None=Header(None)):
    from .candidates import undo
    return undo(candidate_record(cid,authorization))

@app.post('/api/projects/{pid}/stems/{sid}/refine',status_code=202)
def refine_stem(pid: str, sid: str, body: RefineRequest=RefineRequest(), authorization: str | None=Header(None)):
    """Rerun the stem's prompt on the residual and fold what it finds back into the stem."""
    auth(pid,authorization)
    with connect() as db:
        rows=db.execute('SELECT * FROM stems WHERE project_id=?',(pid,)).fetchall()
        busy=db.execute("SELECT kind FROM jobs WHERE project_id=? AND status IN ('queued','processing') LIMIT 1",(pid,)).fetchone()
    target=next((r for r in rows if r['id']==sid),None)
    if not target: raise HTTPException(404,'Stem not found')
    if is_residual(target) or target['role'] not in (None,'target'): raise HTTPException(409,'Only a target stem can be refined; the residual is what refinement searches.')
    residuals=[r for r in rows if is_residual(r)]
    if len(residuals)!=1: raise HTTPException(409,'Refining requires exactly one residual stem in the project.')
    if busy: raise HTTPException(409,f"Wait for the active {busy['kind']} job to finish before refining.")
    if target['start_offset']!=0 or residuals[0]['start_offset']!=0: raise HTTPException(409,'Reset the start offset of the stem and the residual to zero before refining.')
    payload={'stem_id':sid,'residual_stem_id':residuals[0]['id'],'passes':body.passes or 1,
             'revisions':{sid:target['revision'],residuals[0]['id']:residuals[0]['revision']}}
    return {'job_id':start(pid,'refine',payload),'status':'queued'}

@app.get('/api/refine-jobs/{jid}')
def refine_job(jid: str, authorization: str | None=Header(None)):
    return get_job(jid,'refine',authorization)

@app.get('/api/jobs/{jid}')
def job(jid: str, authorization: str | None=Header(None)):
    # The editor polls every job here (processing and refine share the response shape).
    with connect() as db: row=db.execute("SELECT kind FROM jobs WHERE id=? AND kind IN ('process','refine')",(jid,)).fetchone()
    return get_job(jid,row['kind'] if row else 'process',authorization)

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
    ranges=values.get('mute_ranges')
    if ranges is not None: values['mute_ranges']=json.dumps(ranges)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM stems WHERE project_id=? AND id=?',(pid,sid)).fetchone()
        if not row: raise HTTPException(404,'Stem not found')
        if expected is not None and row['revision'] != expected:
            raise HTTPException(409,'Stem changed since it was loaded; reload to resolve the conflict')
        if ranges and any(end > row['duration'] + values.get('start_offset',row['start_offset']) for _,end in ranges):
            raise HTTPException(422,'Mute range extends beyond the stem')
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
