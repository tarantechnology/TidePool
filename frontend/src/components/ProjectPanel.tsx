import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import type { Anchor, EditScope, Capabilities, AgentPlan, BackendConfig, ProjectInfo, SavedStem, VisualMetadata } from '../api/backend'
import { asVisualMetadata } from '../visual'
import type { VisualSelection } from '../visual'
import { separationCommand } from '../separationCommand'
import type { Stem, TrackState } from '../types'
import type { ProjectStemsState } from '../state/projectStems'
import { addMuteRange, getCapabilities, undoCandidate, createCandidate, getCandidate, applyCandidate, planProjectCommand, createProject, downloadExport, fetchStemBlobUrl, getExport, getJob, getMessages, getProject, getSavedStems, localCommand, startExport, storeVisualMetadata, updateStemSettings, uploadProject } from '../api/backend'
import { pollJob, toJobStatus } from '../api/jobs'
import { HttpStemSource } from '../api/stemSources'
import type { StemSource } from '../api/stemSources'

import { SoundGuidance } from './SoundGuidance'
import { CandidateComparison } from './CandidateComparison'
import { GuidedByChip, ObjectsPanel } from './ObjectsPanel'

const field = 'w-full rounded-lg border border-line bg-white/[0.04] px-2 py-1.5 text-xs text-zinc-100 placeholder:text-zinc-500'
const button = 'rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-black disabled:opacity-40'
const secondary = 'rounded-lg border border-line px-3 py-1.5 text-xs text-zinc-200 disabled:opacity-40'
interface Props {
  baseUrl: string
  onProjectState: (state:ProjectStemsState)=>void
  onLoadStems: (source: StemSource, label: string) => Promise<void>
  onVideo: (url: string | null) => void
  onOriginalAudio: (url:string|null) => void
  flush: () => Promise<void>
  onVisual: (metadata: VisualMetadata | null) => void
  selectedVisual: VisualSelection | null
  onClearSelection: () => void
  /** Select an object from the object list, the same way a click on the video does. */
  onSelectVisual: (selection: VisualSelection) => void
  onSeek: (seconds: number) => void
  /** The detections the editor is showing, for the object list. */
  visual: VisualMetadata | null
}

export interface ProjectPanelHandle {
  separateVideo: (file: File, description: string, remove: boolean) => Promise<void>
  uploadMedia: (file:File) => Promise<void>
  visualAction: (selection:VisualSelection, description:string, action:'isolate'|'mute'|'lower') => Promise<void>
  planCommand: (prompt:string, video:File|null, stems:Stem[], tracks:TrackState[], duration:number) => Promise<AgentPlan>
}

/** Project credentials remain in component memory; the user may copy them for resuming. */
export const ProjectPanel = forwardRef<ProjectPanelHandle, Props>(function ProjectPanel({baseUrl, onProjectState, onLoadStems, onVideo, onOriginalAudio, flush, onVisual, selectedVisual, onClearSelection, onSelectVisual, onSeek, visual}, ref) {
  const [project, setProject] = useState<ProjectInfo | null>(null)
  const [token, setToken] = useState('')
  const [resumeId, setResumeId] = useState('')
  const [resumeToken, setResumeToken] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [prompt, setPrompt] = useState('')
  const [selection, setSelection] = useState<unknown>(null)
  const [visualName, setVisualName] = useState('')
  const [selectionName, setSelectionName] = useState('')
  const [rangeStart, setRangeStart] = useState('')
  const [rangeEnd, setRangeEnd] = useState('')
  const [jobId, setJobId] = useState('')
  const [format, setFormat] = useState<'mp4'|'wav'|'mp3'>('mp4')
  const [command, setCommand] = useState('')
  const [retryPrompt,setRetryPrompt]=useState('')
  const [retryStems,setRetryStems]=useState<SavedStem[]>([])
  const [retryStem,setRetryStem]=useState('')
  const [anchors,setAnchors]=useState<Anchor[]>([])
  const [editAction,setEditAction]=useState<EditScope['action']>('keep')
  const [timedEdit,setTimedEdit]=useState(false)
  const [editStart,setEditStart]=useState(0)
  const [editEnd,setEditEnd]=useState(1)
  const [capabilities,setCapabilities]=useState<Capabilities|null>(null)
  const [undoId,setUndoId]=useState('')
  const [mockStems,setMockStems]=useState<boolean|null>(null)
  const [candidateMock,setCandidateMock]=useState<boolean|null>(null)
  const [candidateWarnings,setCandidateWarnings]=useState<string[]>([])
  const [mixUrl,setMixUrl]=useState('')
  const [candidateId,setCandidateId]=useState('')
  const [candidateUrl,setCandidateUrl]=useState('')
  const [currentUrl,setCurrentUrl]=useState('')
  const [settingsStems, setSettingsStems] = useState<SavedStem[]>([])
  const [settingId, setSettingId] = useState('')
  const [stemName, setStemName] = useState('')
  const [offset, setOffset] = useState('0')
  const [clearRanges, setClearRanges] = useState(false)
  const [messages, setMessages] = useState<Array<{id:string;text:string}>>([])
  const [resumeVisual, setResumeVisual] = useState<{projectId:string;jobId:string}|null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const abort = useRef<AbortController | null>(null)
  const attachedVideo = useRef<File | null>(null)
  const mounted = useRef(true)
  useEffect(() => { mounted.current=true; return () => { mounted.current=false; abort.current?.abort() } }, [])
  useEffect(() => () => { if(candidateUrl) URL.revokeObjectURL(candidateUrl) },[candidateUrl])
  useEffect(() => () => { if(currentUrl) URL.revokeObjectURL(currentUrl) },[currentUrl])
  useEffect(() => { if (selectedVisual) { setSelection(selectedVisual); setSelectionName(selectedVisual.label); setRangeStart(String(selectedVisual.interval.start_seconds)); setRangeEnd(String(selectedVisual.interval.end_seconds)) } }, [selectedVisual])
  useEffect(() => () => { if(mixUrl) URL.revokeObjectURL(mixUrl) },[mixUrl])
  const projectId=project?.project_id
  useEffect(()=>{let active=true;if(projectId && token)void getCapabilities({baseUrl,token},projectId).then(value=>{if(active)setCapabilities(value)}).catch(()=>{if(active)setCapabilities(null)});return()=>{active=false}},[baseUrl,token,projectId])
  useEffect(()=>{onProjectState({projectId:projectId ?? null,mockStems,warnings:candidateWarnings})},[projectId,mockStems,candidateWarnings,onProjectState])
  const config: BackendConfig = {baseUrl, token}
  const run = async (fn: () => Promise<void>) => {
    setError(''); setMessage(''); setBusy(true)
    try { await fn() } catch (e) { if(mounted.current)setError(e instanceof Error ? e.message : String(e)) }
    finally { if(mounted.current)setBusy(false) }
  }
  const loadVideo = async (cfg: BackendConfig, info: ProjectInfo) => {
    if (!info.metadata?.has_video) { onVideo(null); return }
    // Authenticated media is fetched into a blob; the token never enters a URL.
    onVideo(await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(info.project_id)}/preview`, 3_600_000))
  }
  const track = async (cfg: BackendConfig, id: string) => {
    abort.current?.abort(); const controller = new AbortController(); abort.current = controller
    const result = await pollJob(async signal => toJobStatus(await getJob(cfg, id, signal)), {
      signal: controller.signal, onUpdate: status => setMessage(`Finding sound: ${status.state}${status.progress !== undefined ? ` ${status.progress}%` : ''}`),
    })
    if (result.state === 'failed') throw new Error(result.error || 'Processing failed')
    const job = await getJob(cfg, id)
    await onLoadStems(new HttpStemSource(cfg, id), 'Project stems')
    setJobId(id)
    const model = job.inference?.model
    setMockStems(model?.is_mock ?? null)
    const summary = model?.is_mock ? 'Ready — test stems; no AI separation was performed.' : model?.is_mock === false ? 'Stems ready to edit.' : 'Stems ready; model identity unavailable.'
    setMessage([summary, ...(job.inference?.warnings || [])].join(' '))
    return job.inference?.warnings || []
  }
  const uploadMedia = (media: File) => run(async () => {
    await flush()
    const created = await createProject(baseUrl)
    if (!created.access_token) throw new Error('Project token was not returned.')
    const cfg = {baseUrl, token:created.access_token}
    setProject(created); setToken(created.access_token); setResumeId(created.project_id); setResumeToken(created.access_token)
    setMockStems(null);setJobId(''); setUndoId('');setAnchors([]); setSettingsStems([]); setMessages([]); setRetryStems([]); setCandidateId(''); setCandidateUrl(''); setCurrentUrl(''); onOriginalAudio(null); await onLoadStems({load: async () => []}, 'No stems yet'); onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
    setMessage('Uploading and extracting audio…')
    await uploadProject(cfg, created.project_id, media)
    attachedVideo.current = media
    const info = await getProject(cfg, created.project_id); setProject(info)
    const original=await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(created.project_id)}/original-audio`, 3_600_000)
    onOriginalAudio(original)
    await onLoadStems({load:async()=>[{id:'project-original',label:'Original audio',kind:'sfx',url:original,duration:0}]}, 'Original audio')
    setFormat(info.metadata?.has_video ? 'mp4' : 'wav')
    await loadVideo(cfg, info)
    setMessage('Upload complete. Add a sound prompt and process.')
  })
  const resume = () => run(async () => {
    if (!resumeId.trim() || !resumeToken.trim()) throw new Error('Enter both project ID and token.')
    await flush()
    const cfg = {baseUrl,token:resumeToken.trim()}; const info = await getProject(cfg,resumeId.trim())
    attachedVideo.current = null
    setUndoId('');setCandidateId('');setCandidateUrl('');setMixUrl('');setCurrentUrl('');setRetryStems([]);setAnchors([])
    setToken(cfg.token); setProject(info); setJobId(info.last_completed_job_id || ''); setSettingsStems([]); setMessages([])
    setFormat(info.metadata?.has_video ? 'mp4' : 'wav'); setSelection(null); onClearSelection(); setVisualName(''); setSelectionName('')
    try { onVisual(info.visual_metadata ? asVisualMetadata(info.visual_metadata) : null) } catch { onVisual(null) }
    await loadVideo(cfg, info)
    const original=info.metadata ? await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(info.project_id)}/original-audio`, 3_600_000) : null
    onOriginalAudio(original)
    const visualJob = info.latest_visual_job
    // The Objects section follows it, without holding up the rest of the restore.
    setResumeVisual(visualJob && (visualJob.status === 'queued' || visualJob.status === 'processing') ? {projectId:info.project_id,jobId:visualJob.id} : null)
    const latest = info.latest_job
    if (latest && (latest.status === 'queued' || latest.status === 'processing')) await track(cfg, latest.id)
    else if (info.last_completed_job_id) { await onLoadStems(new HttpStemSource(cfg,info.last_completed_job_id), 'Project stems'); setMessage('Project restored.') }
    else { await onLoadStems({load: async () => original ? [{id:'project-original',label:'Original audio',kind:'sfx',url:original,duration:0}] : []}, original ? 'Original audio' : 'No stems yet'); setMessage('Project restored. Upload or start processing.') }
  })
  const previewSound = async (cfg:BackendConfig, info:ProjectInfo, sound:string, scope:EditScope, visual:unknown={}, stemId:string|null=null, quality=false, suppliedAnchors=anchors) => {
    await flush()
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMixUrl('')
    const id=await createCandidate(cfg,info.project_id,stemId,sound,quality?2:1,{guidance:{anchors:suppliedAnchors},edit_scope:scope,visual_prompt:visual,profile:quality?'quality':'preview'})
    for (;;) {
      if(!mounted.current)throw new Error('Preview cancelled')
      const job=await getCandidate(cfg,id)
      if(job.status==='failed')throw new Error(job.error || 'Preview failed. Try a sound description with a time anchor.')
      if(job.status==='completed') {
        if(!job.audio_url || !job.mix_url || !job.original_url)throw new Error('Preview audio is incomplete')
        const urls=await Promise.all([fetchStemBlobUrl(cfg,job.audio_url),fetchStemBlobUrl(cfg,job.mix_url),fetchStemBlobUrl(cfg,job.original_url)])
        setCandidateMock(job.inference?.model?.is_mock ?? null);setCandidateWarnings(job.inference?.warnings || [])
        setCandidateUrl(urls[0]);setMixUrl(urls[1]);setCurrentUrl(urls[2]);setCandidateId(id)
        setMessage(['Preview ready. Listen to the target and edited mix, then Apply preview.',...(job.inference?.warnings || [])].join(' '));return
      }
      setMessage(`Preparing preview: ${job.status}`);await new Promise(resolve=>setTimeout(resolve,1000))
    }
  }
  const process = (quality=false) => run(async () => {
    if (!project) return
    const instruction=separationCommand(prompt)
    const sound=instruction?.description || prompt.trim()
    const scope:EditScope=instruction?.editScope || {action:instruction?.remove?'mute':editAction,...(timedEdit && editAction==='mute'?{interval:{start:editStart,end:editEnd}}:{})}
    let visual:unknown={}
    if(selection)visual={...selection as object,interval:{start_seconds:Number(rangeStart),end_seconds:Number(rangeEnd)}}
    await previewSound(config,project,sound,scope,visual,null,quality)
  })
  useImperativeHandle(ref, () => ({ uploadMedia, visualAction: async (chosen, description, action) => {
    if (!project || !token) throw new Error('Upload a video first.')
    if (!description.trim()) throw new Error('Describe the sound to isolate.')
    if (busy) throw new Error('A project operation is already running.')
    setBusy(true); setError(''); setMessage('Preparing visual selection…')
    try {
      await flush()
      setPrompt(description);setEditAction(action==='isolate'?'keep':action)
      await previewSound(config,project,description,{action:action==='isolate'?'keep':action},chosen)
    } catch(e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { setBusy(false) }
  }, separateVideo: async (video, description, remove) => {
    if (busy) throw new Error('A project operation is already running. Please wait.')
    setError(''); setMessage('Uploading video for separation…'); setBusy(true)
    try {
      await flush()
      const created=await createProject(baseUrl)
      if (!created.access_token) throw new Error('Project token was not returned.')
      const cfg={baseUrl,token:created.access_token}
      setProject(created); setToken(cfg.token); setResumeId(created.project_id); setResumeToken(cfg.token)
      setJobId(''); setSettingsStems([]); setMessages([]); onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
      await uploadProject(cfg,created.project_id,video)
      attachedVideo.current=video
      const info=await getProject(cfg,created.project_id); setProject(info); setFormat('mp4')
      setMessage('Separating audio…')
      await previewSound(cfg,info,description,{action:remove?'mute':'keep'})
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { setBusy(false) }
  }, planCommand: async (text, video, stems, tracks, duration) => {
    if (busy) throw new Error('A project operation is already running. Please wait.')
    setError(''); setMessage('Understanding your request…'); setBusy(true)
    try {
      await flush()
      let current=project, currentToken=token, currentJob=jobId
      if (video && attachedVideo.current !== video) {
        const created=await createProject(baseUrl)
        if (!created.access_token) throw new Error('Project token was not returned.')
        currentToken=created.access_token
        const cfg={baseUrl,token:currentToken}
        setProject(created); setToken(currentToken); setResumeId(created.project_id); setResumeToken(currentToken)
        setJobId(''); currentJob=''; setSettingsStems([]); setMessages([])
        onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
        setMessage('Uploading video…')
        await uploadProject(cfg,created.project_id,video); attachedVideo.current=video
        current=await getProject(cfg,created.project_id); setProject(current); setFormat('mp4')
      }
      if (!current || !currentToken) throw new Error('Upload a video in Project first to use AI commands.')
      const cfg={baseUrl,token:currentToken}
      const hasStems=Boolean(currentJob)
      const result=await planProjectCommand(cfg,current.project_id,{prompt:text, stems:hasStems ? stems.map(({id,label,kind,duration:d})=>({id,label,kind,duration:d})) : [],tracks:hasStems ? tracks : [],duration:hasStems ? duration : (current.metadata?.duration || duration),has_project_stems:hasStems,selected_visual:selectedVisual ? {label:selectedVisual.label,start:selectedVisual.interval.start_seconds,end:selectedVisual.interval.end_seconds} : null})
      if(result.kind==='edit' && Array.isArray(result.actions) && result.actions.some(a=>a.type==='mute_range')) {
        if(result.actions.some(a=>a.type!=='mute_range'))throw new Error('Apply timed mute separately from other commands.')
        for(const action of result.actions as Array<{stemId:string;start:number;end:number}>) {
          const saved=await getSavedStems(cfg,current.project_id);const stem=saved.find(s=>s.id===action.stemId)
          if(!stem)throw new Error('The selected layer is unavailable')
          await addMuteRange(cfg,current.project_id,stem,action.start,action.end)
        }
        if(currentJob)await onLoadStems(new HttpStemSource(cfg,currentJob),'Project stems')
        result.actions=[];result.message='Timed mute applied; existing automation preserved.'
      }
      if (result.kind === 'separate') {
        if (!result.description?.trim()) throw new Error('The requested sound is missing.')
        if (result.use_visual && !selectedVisual) throw new Error('The selected object is no longer available; choose it again.')
        setPrompt(result.description);setEditAction(result.edit_scope?.action || (result.remove?'mute':'keep'))
        await previewSound(cfg,current,result.description,result.edit_scope || {action:result.remove?'mute':'keep'},result.use_visual?selectedVisual:{},null,false,result.guidance?.anchors || [])
        result.message='Preview ready. Listen and apply it in Project.'
      } else setMessage(result.message)
      return result
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { setBusy(false) }
  }}))
  const visualAnalyzed = (metadata: VisualMetadata) => {
    onVisual(metadata); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('Analysis complete')
  }
  /** After the backend re-encoded a compatible copy: reload the video, and the original audio if nothing was separated yet. */
  const reloadPreparedMedia = async () => {
    if (!project) return
    const info=await getProject(config,project.project_id); setProject(info)
    if (!jobId) {
      const original=await fetchStemBlobUrl(config,`/api/projects/${encodeURIComponent(project.project_id)}/original-audio`,3_600_000)
      onOriginalAudio(original)
      await onLoadStems({load:async()=>[{id:'project-original',label:'Original audio',kind:'sfx',url:original,duration:0}]},'Original audio')
    }
    await loadVideo(config,info); onVisual(null); onClearSelection(); setSelection(null)
  }
  const finishExport = async (id: string, extension: string) => {
    for (;;) {
      const info=await getExport(config,id)
      if (info.status === 'failed') throw new Error(info.error || 'Export failed')
      if (info.status === 'completed') {
        if (!info.download_url) throw new Error('Export has no download URL')
        await downloadExport(config,info.download_url,`soundcanvas.${extension}`); setMessage('Export downloaded.'); return
      }
      setMessage(`Export: ${info.status}`); await new Promise(resolve => setTimeout(resolve,1000))
    }
  }
  const exportFile = () => run(async () => {
    if (!project) return
    await flush()
    const id = await startExport(config,project.project_id,format)
    await finishExport(id, format)
  })
  const readJson = async (input: File): Promise<unknown> => {
    try { return JSON.parse(await input.text()) as unknown } catch { throw new Error(`Invalid JSON in ${input.name}`) }
  }
  const refreshSettings = () => run(async () => {
    if (!project) return
    await flush(); const stems=await getSavedStems(config,project.project_id)
    setSettingsStems(stems)
    if (stems[0]) { setSettingId(stems[0].id); setStemName(stems[0].name); setOffset(String(stems[0].start_offset || 0)); setClearRanges(false) }
    setMessage('Stem settings loaded.')
  })
  const saveSettings = () => run(async () => {
    if (!project || !jobId) return
    const stem=settingsStems.find(s => s.id===settingId); const position=Number(offset)
    if (!stem || !stemName.trim() || !Number.isFinite(position) || position<0) throw new Error('Enter a name and a nonnegative offset.')
    await flush()
    await updateStemSettings(config,project.project_id,stem,{name:stemName.trim(),start_offset:position,...(clearRanges ? {mute_ranges:[]} : {})})
    await onLoadStems(new HttpStemSource(config,jobId),'Project stems')
    setSettingsStems([]); setMessage('Stem settings saved. Reload settings to edit again.')
  })
  const prepareRetry = () => run(async () => {
    if (!project) return
    await flush()
    const stems=await getSavedStems(config,project.project_id)
    const targets=stems.filter(s=>s.name.toLowerCase()!=='residual')
    if(stems.length!==2 || targets.length!==1) throw new Error('Retry preview currently supports one isolated sound and its residual. Existing layers are unchanged.')
    setRetryStems(targets);setRetryStem(targets[0].id);setRetryPrompt(targets[0].name)
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMessage('Describe the sound and generate a new preview.')
  })
  const retry = (quality=false) => run(async () => {
    if(!project || !retryStem) return
    await previewSound(config,project,retryPrompt,{action:'keep'},selection || {},retryStem,quality)
  })
  const useRetry = () => run(async () => {
    if(!candidateId || !project) return
    await flush();const applied=await applyCandidate(config,candidateId)
    await onLoadStems(new HttpStemSource(config,applied.job_id),'Project stems')
    setMockStems(candidateMock);setJobId(applied.job_id);setUndoId(candidateId);setCandidateId('');setRetryStems([])
    setProject(await getProject(config,project.project_id))
    setMessage('Preview applied. Existing layer edits were preserved.')
  })
  const undoPreview = () => run(async () => {
    if(!undoId || !project)return
    await flush();const result=await undoCandidate(config,undoId)
    const info=await getProject(config,project.project_id);setProject(info)
    const saved=await getSavedStems(config,project.project_id)
    if(saved.length){await onLoadStems(new HttpStemSource(config,result.job_id),'Project stems');setJobId(result.job_id)}
    else {setJobId('');const url=await fetchStemBlobUrl(config,`/api/projects/${project.project_id}/original-audio`);await onLoadStems({load:async()=>[{id:'project-original',label:'Original audio',kind:'sfx',url,duration:info.metadata?.duration || 0}]},'Original audio')}
    setUndoId('');setMessage('Previous audio restored.')
  })
  const sendCommand = () => run(async () => {
    if (!project || !command.trim()) return
    await flush(); const result = await localCommand(config,project.project_id,command)
    setMessage(result.message)
    if (result.status === 'accepted' && result.export_id) await finishExport(result.export_id, 'mp4')
    if (result.status === 'applied' && jobId) await onLoadStems(new HttpStemSource(config,jobId),'Project stems')
  })
  return <section className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-line bg-ink-900/80 p-3 text-zinc-200" aria-label="Project controls">
    <div className="mb-2 flex items-center justify-between"><h2 className="text-sm font-semibold">Project</h2><span className="text-[10px] text-zinc-500">{project?.status || 'New'}</span></div>
    <div className="space-y-2">
      <label className="block text-xs">Media<input className={`${field} mt-1`} type="file" accept=".mp4,.mov,.wav,.mp3" onChange={e => setFile(e.target.files?.[0] || null)} /></label>
      <button className={button} disabled={busy || !file} onClick={()=>{if(file) void uploadMedia(file)}}>Create & upload</button>
      {project && <details className="text-xs"><summary className="cursor-pointer">Project credentials</summary><p className="mt-1 break-all">ID: {project.project_id}</p><label className="block mt-1">Token<input className={field} readOnly value={token} type="text" onFocus={e => e.currentTarget.select()} /></label><p className="text-zinc-500">Copy these to resume this project later.</p></details>}
      <details className="text-xs"><summary className="cursor-pointer">Resume a project</summary><div className="mt-2 space-y-1"><input className={field} value={resumeId} onChange={e=>setResumeId(e.target.value)} placeholder="Project ID" aria-label="Project ID"/><input className={field} type="password" value={resumeToken} onChange={e=>setResumeToken(e.target.value)} placeholder="Project token" aria-label="Project token"/><button className={secondary} disabled={busy} onClick={resume}>Resume</button></div></details>
      {project && <>
        <label className="block text-xs">Sound description<input className={`${field} mt-1`} value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="dog barking" /></label>
        {project.metadata?.has_video && <ObjectsPanel key={project.project_id} config={config} projectId={project.project_id} vision={capabilities?.vision} visual={visual} selection={selectedVisual} disabled={busy} resumeJobId={resumeVisual?.projectId===project.project_id ? resumeVisual.jobId : null} canPrepare={!jobId} onAnalyzed={visualAnalyzed} onSelect={onSelectVisual} onSeek={onSeek} onPrepared={reloadPreparedMedia}/>}
        <details className="text-xs"><summary className="cursor-pointer">Advanced: import detections</summary><p className="my-1 text-zinc-400">Import detection and selection JSON files that match this video, instead of Find objects.</p><label className="block">Detections<input className={field} type="file" accept=".json" disabled={busy} onChange={e=>{const f=e.target.files?.[0]; if(f) void run(async()=>{const data=asVisualMetadata(await readJson(f));await storeVisualMetadata(config,project.project_id,data);onVisual(data);setVisualName(f.name);setMessage('Detections saved.')})}} /></label>{visualName && <span>{visualName}</span>}<label className="block mt-1">Selection<input className={field} type="file" accept=".json" disabled={busy} onChange={e=>{const f=e.target.files?.[0]; if(f) void run(async()=>{const data=await readJson(f) as VisualSelection; if (!data?.interval) throw new Error('Selection JSON needs an interval.'); setSelection(data);setRangeStart(String(data.interval.start_seconds));setRangeEnd(String(data.interval.end_seconds));setSelectionName(f.name);setMessage('Selection ready.')})}} /></label>{selectionName && <><button className="text-accent" onClick={()=>{setSelection(null);onClearSelection();setSelectionName('')}}>Clear {selectionName}</button><div className="flex gap-1"><label>Start (seconds)<input className={field} type="number" min="0" step="any" value={rangeStart} onChange={e=>setRangeStart(e.target.value)} /></label><label>End (seconds)<input className={field} type="number" min="0" step="any" value={rangeEnd} onChange={e=>setRangeEnd(e.target.value)} /></label></div></>}</details>
        <SoundGuidance duration={project.metadata?.duration || 1} anchors={anchors} onChange={setAnchors}/>
        <label className="block text-xs">Edit after separation<select aria-label="Edit after separation" className={field} value={editAction} onChange={e=>setEditAction(e.target.value as EditScope['action'])}><option value="keep">Keep isolated layer</option><option value="mute">Mute target</option><option value="lower">Lower target 50%</option></select></label>
        {editAction==='mute' && <div className="text-xs"><label><input type="checkbox" checked={timedEdit} onChange={e=>setTimedEdit(e.target.checked)}/>Mute only an edit interval</label>{timedEdit && <div className="flex gap-1"><label>Edit start<input aria-label="Edit start" className={field} type="number" min={0} value={editStart} onChange={e=>setEditStart(Number(e.target.value))}/></label><label>Edit end<input aria-label="Edit end" className={field} type="number" min={0} value={editEnd} onChange={e=>setEditEnd(Number(e.target.value))}/></label></div>}</div>}
        {selection !== null && <GuidedByChip selection={selection as {label?:unknown;track_id?:unknown}} start={Number(rangeStart)} end={Number(rangeEnd)} onClear={()=>{setSelection(null);onClearSelection();setSelectionName('')}}/>}
        <div className="flex gap-1"><button className={button} disabled={busy || !project.metadata || !prompt.trim()} onClick={()=>void process()}>Preview</button><button className={secondary} disabled={busy || !prompt.trim() || !capabilities?.quality} onClick={()=>void process(true)}>Quality retry</button></div>
        {!capabilities?.quality && <p className="text-xs text-zinc-400">{capabilities?.reason || 'Checking quality retry availability…'}</p>}
        {project.metadata?.audio_quality?.findings.map(f=><p key={f} className="text-xs text-amber-300">{f}</p>)}
        {candidateId && <><CandidateComparison key={candidateId} original={currentUrl} edited={mixUrl} target={candidateUrl}/><button className={button} disabled={busy} onClick={useRetry}>Apply preview</button></>}
        {selectionName && (candidateId || error) && <button className={secondary} disabled={busy} onClick={()=>{setSelection(null);setSelectionName('');onClearSelection();setCandidateId('');setMessage('Mark where you hear the sound with I hear it here, then Preview to retry with text and time.')}}>Retry with text + time</button>}
        {undoId && <button className={secondary} disabled={busy} onClick={undoPreview}>Undo applied preview</button>}
        <div className="flex gap-2"><select className={field} aria-label="Export format" value={format} onChange={e=>setFormat(e.target.value as typeof format)}><option value="mp4" disabled={!project.metadata?.has_video}>MP4 video</option><option value="wav">WAV audio</option><option value="mp3">MP3 audio</option></select><button className={secondary} disabled={busy || !jobId} onClick={exportFile}>Export</button></div>
        {jobId && <details className="text-xs"><summary className="cursor-pointer">Stem settings</summary><button className={`${secondary} mt-1`} disabled={busy} onClick={refreshSettings}>Load stem settings</button>{settingsStems.length>0 && <div className="mt-1 space-y-1"><select className={field} aria-label="Stem" value={settingId} onChange={e=>{const stem=settingsStems.find(s=>s.id===e.target.value);if(stem){setSettingId(stem.id);setStemName(stem.name);setOffset(String(stem.start_offset || 0));setClearRanges(false)}}}>{settingsStems.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select><label>Name<input className={field} value={stemName} onChange={e=>setStemName(e.target.value)} /></label><label>Start offset (seconds)<input className={field} type="number" min="0" step="any" value={offset} onChange={e=>setOffset(e.target.value)} /></label><label className="flex items-center gap-1"><input type="checkbox" checked={clearRanges} onChange={e=>setClearRanges(e.target.checked)} />Clear remote mute ranges</label><button className={secondary} disabled={busy} onClick={saveSettings}>Save stem settings</button></div>}</details>}
        {jobId && <details className="text-xs"><summary className="cursor-pointer">Try another separation</summary>
          <p className="my-1 text-zinc-400">Replace a target in a two-layer project. For multitrack projects, Preview above splits the current residual; replacing an older target requires an earlier source snapshot.</p>
          <button className={secondary} disabled={busy} onClick={prepareRetry}>Choose sound</button>
          {retryStems.length>0 && <div className="space-y-2"><label>Retry sound description<input className={field} value={retryPrompt} onChange={e=>setRetryPrompt(e.target.value)}/></label><button className={secondary} disabled={busy} onClick={()=>void retry()}>Preview replacement</button><button className={secondary} disabled={busy || !capabilities?.quality} onClick={()=>void retry(true)}>Quality replacement</button></div>}
        </details>}
        <details className="text-xs"><summary className="cursor-pointer">Remote command test</summary><p className="my-1 text-zinc-500">Local webhook; messaging delivery is not connected.</p><div className="flex gap-1"><input className={field} value={command} onChange={e=>setCommand(e.target.value)} placeholder="Mute the dog"/><button className={secondary} disabled={busy || !command.trim()} onClick={sendCommand}>Send</button></div><button className={`${secondary} mt-1`} disabled={busy} onClick={()=>void run(async()=>{setMessages(await getMessages(config,project.project_id));setMessage('Messages updated.')})}>Refresh messages</button>{messages.map(m=><p key={m.id} className="mt-1 text-zinc-400">{m.text}</p>)}</details>
      </>}
      {busy && <p role="status" className="text-xs text-accent">{message || 'Working…'}</p>}
      {!busy && message && <p role="status" className="text-xs text-accent">{message}</p>}
      {error && <p role="alert" className="break-words text-xs text-red-400">{error}</p>}
    </div>
  </section>
})
