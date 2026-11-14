import { forwardRef, useEffect, useEffectEvent, useImperativeHandle, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { Anchor, EditScope, Capabilities, AgentPlan, BackendConfig, ProjectInfo, Routing, SavedStem, VisualMetadata } from '../api/backend'
import { asVisualMetadata } from '../visual'
import type { VisualSelection } from '../visual'
import { musicCommand, separationCommand } from '../separationCommand'
import { separationSettings } from '../agent/separationSettings'
import { muteExistingStem } from '../agent/planOutcome'
import { splitSuggestion } from '../state/projectStems'
import type { Stem, TrackState } from '../types'
import type { ProjectStemsState } from '../state/projectStems'
import { addMuteRange, describeRouting, getRoute, splitTracks, toRouting, getCapabilities, undoCandidate, createCandidate, getCandidate, applyCandidate, planProjectCommand, createProject, downloadExport, fetchStemBlobUrl, getExport, getJob, getMessages, getProject, getSavedStems, localCommand, startExport, storeVisualMetadata, updateStemSettings, uploadProject } from '../api/backend'
import { jobProgressMessage, pollJob, toJobStatus } from '../api/jobs'
import { CANCELLED_MESSAGE, isAbort } from '../api/errors'
import { clampPasses, passChoices, refineBlockReason, refineConflictMessage, runRefine } from '../api/refine'
import { HttpStemSource } from '../api/stemSources'
import type { StemSource } from '../api/stemSources'

import { SoundGuidance } from './SoundGuidance'
import { CandidateComparison } from './CandidateComparison'
import { GuidedByChip, ObjectsPanel } from './ObjectsPanel'
import { PassesSelect } from './PassesSelect'
import { DropLoader } from './DropLoader'
import { FEATURES } from '../features'
import { FeatureTag } from './FeatureTag'

const field = 'w-full rounded-lg border border-line bg-surface px-2 py-1.5 text-xs text-fg placeholder:text-subtle focus:border-accent/50 focus:outline-none focus:ring-2 focus:ring-accent/15'
const button = 'rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-[#1b62b3] disabled:opacity-40'
const secondary = 'rounded-lg border border-line bg-surface px-3 py-1.5 text-xs text-fg transition-colors hover:border-line-strong hover:bg-surface-2 disabled:opacity-40'
interface Props {
  baseUrl: string
  readyToRestore: boolean
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
  /** Where the preview comparison card renders, over the timeline. */
  previewSlot: HTMLElement | null
  /** What the panel is busy with, for the timeline's working state; null when idle. */
  onBusy?: (label: string | null) => void
}

export interface ProjectPanelHandle {
  separateVideo: (file: File, description: string, remove: boolean) => Promise<void>
  uploadMedia: (file:File) => Promise<void>
  visualAction: (selection:VisualSelection, description:string, action:'isolate'|'mute'|'lower') => Promise<void>
  planCommand: (prompt:string, video:File|null, stems:Stem[], tracks:TrackState[], duration:number) => Promise<AgentPlan>
  /** Split the uploaded song into one track per instrument. */
  splitSong: () => Promise<void>
}

/** The project token stays in this browser tab and is sent only as a bearer header. */
export const ProjectPanel = forwardRef<ProjectPanelHandle, Props>(function ProjectPanel({baseUrl, readyToRestore, onProjectState, onLoadStems, onVideo, onOriginalAudio, flush, onVisual, selectedVisual, onClearSelection, onSelectVisual, onSeek, visual, previewSlot, onBusy}, ref) {
  const [project, setProject] = useState<ProjectInfo | null>(null)
  const [token, setToken] = useState('')
  const [resumeId, setResumeId] = useState('')
  const [resumeToken, setResumeToken] = useState('')
  const [resumeVisual, setResumeVisual] = useState<{projectId:string;jobId:string}|null>(null)
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
  const [passes, setPasses] = useState(1)
  const [retryPasses, setRetryPasses] = useState(1)
  const [refining, setRefining] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  // A manual open/close lasts until the flow moves on (a new project, preview or apply).
  const [manualStep, setManualStep] = useState<{key:string;id:StepId|'none'}|null>(null)
  const [error, setError] = useState('')
  // The router's answer for this upload, and which separator the last split used.
  const [suggestion, setSuggestion] = useState<{projectId:string;text:string}|null>(null)
  const [routing, setRouting] = useState<Routing|null>(null)
  const abort = useRef<AbortController | null>(null)
  const attachedVideo = useRef<File | null>(null)
  const mounted = useRef(true)
  const busyRef = useRef(false)
  const restored = useRef(false)
  useEffect(() => { mounted.current=true; return () => { mounted.current=false; abort.current?.abort() } }, [])
  useEffect(() => () => { if(candidateUrl) URL.revokeObjectURL(candidateUrl) },[candidateUrl])
  useEffect(() => () => { if(currentUrl) URL.revokeObjectURL(currentUrl) },[currentUrl])
  useEffect(() => { if (selectedVisual) { setSelection(selectedVisual); setSelectionName(selectedVisual.label); setRangeStart(String(selectedVisual.interval.start_seconds)); setRangeEnd(String(selectedVisual.interval.end_seconds)) } }, [selectedVisual])
  useEffect(() => () => { if(mixUrl) URL.revokeObjectURL(mixUrl) },[mixUrl])
  const projectId=project?.project_id
  useEffect(()=>{let active=true;if(projectId && token)void getCapabilities({baseUrl,token},projectId).then(value=>{if(active)setCapabilities(value)}).catch(()=>{if(active)setCapabilities(null)});return()=>{active=false}},[baseUrl,token,projectId])
  const audioOnly = project?.metadata?.has_video === false
  // A split replaces every layer, so it is offered only before the first separation.
  const canSplit = Boolean(project?.metadata) && !jobId && !candidateId && capabilities?.musicStems?.available !== false
  const suggested = canSplit && suggestion && suggestion.projectId === projectId ? suggestion.text : null
  useEffect(()=>{onProjectState({projectId:projectId ?? null,mockStems,warnings:candidateWarnings,audioOnly,split:canSplit ? {suggested} : null})},[projectId,mockStems,candidateWarnings,audioOnly,canSplit,suggested,onProjectState])
  useEffect(()=>{onBusy?.(busy ? (message || 'Working…') : null)},[busy,message,onBusy])
  const config: BackendConfig = {baseUrl, token}
  const choices = passChoices(capabilities?.max_passes)
  const run = async (fn: () => Promise<void>) => {
    if (busyRef.current) return
    busyRef.current = true
    setError(''); setMessage(''); setBusy(true)
    try { await fn() } catch (e) { if(mounted.current)setError(e instanceof Error ? e.message : String(e)) }
    finally { busyRef.current = false; if(mounted.current)setBusy(false) }
  }
  const loadVideo = async (cfg: BackendConfig, info: ProjectInfo) => {
    if (!info.metadata?.has_video) { onVideo(null); return }
    // Authenticated media is fetched into a blob; the token never enters a URL.
    onVideo(await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(info.project_id)}/preview`, 3_600_000))
  }
  const rememberProject = (id: string, credential: string) => {
    setResumeId(id); setResumeToken(credential)
    try { sessionStorage.setItem(`soundcanvas-project:${baseUrl}`, JSON.stringify({id, token:credential})) } catch { /* Manual credentials remain available. */ }
  }
  async function waitCandidate(cfg:BackendConfig, projectId:string, id:string) {
    const began=Date.now()
    for (;;) {
      if(!mounted.current)throw new Error('Preview cancelled')
      const job=await getCandidate(cfg,id)
      if(job.status==='failed') { try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${projectId}`) } catch { /* Ignore unavailable storage. */ }; throw new Error(job.error || 'Preview failed. Try a sound description with a time anchor.') }
      if(job.status==='completed') {
        if(!job.audio_url || !job.mix_url || !job.original_url)throw new Error('Preview audio is incomplete')
        const urls=await Promise.all([fetchStemBlobUrl(cfg,job.audio_url),fetchStemBlobUrl(cfg,job.mix_url),fetchStemBlobUrl(cfg,job.original_url)])
        setCandidateMock(job.inference?.model?.is_mock ?? null);setCandidateWarnings(job.inference?.warnings || [])
        setCandidateUrl(urls[0]);setMixUrl(urls[1]);setCurrentUrl(urls[2]);setCandidateId(id)
        setMessage(['Preview ready. Compare it above the timeline, then apply.',...(job.inference?.warnings || [])].join(' '));return
      }
      setMessage(jobProgressMessage({state:job.status === 'queued' ? 'queued' : 'processing',progress:job.progress},Date.now()-began,'Separating the sound'));await new Promise(resolve=>setTimeout(resolve,1000))
    }
  }
  /** Ask the router what the upload is, in the background; a song gets "Split into tracks" offered. */
  const suggestRoute = (cfg: BackendConfig, id: string) => {
    void getRoute(cfg, id).then(r => { const text = splitSuggestion(r); if (mounted.current && text) setSuggestion({projectId:id, text}) }).catch(() => { /* Only a hint; the split button still works. */ })
  }
  /** Run a music split as the project's separation and load its tracks. */
  const splitInto = async (cfg: BackendConfig, info: ProjectInfo, stems: string[] | null, scope?: EditScope) => {
    await flush()
    setMessage('Splitting into tracks…')
    const id = await splitTracks(cfg, info.project_id, stems, scope)
    const controller = new AbortController(); abort.current = controller
    const began = Date.now()
    const result = await pollJob(async signal => toJobStatus(await getJob(cfg, id, signal)), {signal:controller.signal, onUpdate: status => setMessage(jobProgressMessage(status, Date.now()-began, 'Splitting into tracks'))})
    if (result.state === 'failed') throw new Error(result.error || 'The split failed.')
    const job = await getJob(cfg, id)
    await onLoadStems(new HttpStemSource(cfg, id), 'Project stems')
    const how = toRouting(job.routing)
    setJobId(id); setMockStems(job.inference?.model?.is_mock ?? null); setCandidateWarnings(job.inference?.warnings || []); setRouting(how); setSuggestion(null)
    setPrompt(stems?.join(', ') || 'All instruments'); setProject(await getProject(cfg, info.project_id))
    const count = job.stems?.filter(s => s.role !== 'residual').length ?? 0
    setMessage([`Split into ${count} track${count === 1 ? '' : 's'}.`, ...(job.inference?.warnings || [])].join(' '))
  }
  const resume = (id: string, credential: string) => run(async () => {
    if (!id.trim() || !credential.trim()) throw new Error('Enter both project ID and token.')
    await flush()
    const cfg = {baseUrl, token:credential.trim()}
    const info = await getProject(cfg, id.trim())
    attachedVideo.current = null; setFile(null)
    setUndoId(''); setCandidateId(''); setCandidateUrl(''); setMixUrl(''); setCurrentUrl(''); setRetryStems([]); setAnchors([]); setRouting(null); setSuggestion(null)
    setToken(cfg.token); setProject(info); rememberProject(info.project_id, cfg.token)
    setJobId(info.last_completed_job_id || ''); setSettingsStems([]); setMessages([])
    setFormat(info.metadata?.has_video ? 'mp4' : 'wav'); setSelection(null); onClearSelection(); setVisualName(''); setSelectionName('')
    try { onVisual(info.visual_metadata ? asVisualMetadata(info.visual_metadata) : null) } catch { onVisual(null) }
    await loadVideo(cfg, info)
    const original = info.metadata ? await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(info.project_id)}/original-audio`, 3_600_000) : null
    onOriginalAudio(original)
    const visualJob = info.latest_visual_job
    setResumeVisual(visualJob && (visualJob.status === 'queued' || visualJob.status === 'processing') ? {projectId:info.project_id, jobId:visualJob.id} : null)
    const latest = info.latest_job
    if (latest && (latest.status === 'queued' || latest.status === 'processing')) {
      const controller = new AbortController(); abort.current = controller
      const began = Date.now()
      const result = await pollJob(async signal => toJobStatus(await getJob(cfg, latest.id, signal)), {signal:controller.signal, onUpdate: status => setMessage(jobProgressMessage(status, Date.now()-began, 'Finding sound'))})
      if (result.state === 'failed') throw new Error(result.error || 'Processing failed')
      const job = await getJob(cfg, latest.id)
      await onLoadStems(new HttpStemSource(cfg, latest.id), 'Project stems')
      setJobId(latest.id); setMockStems(job.inference?.model?.is_mock ?? null); setRouting(toRouting(job.routing))
    } else if (info.last_completed_job_id) await onLoadStems(new HttpStemSource(cfg, info.last_completed_job_id), 'Project stems')
    else {
      await onLoadStems({load: async () => original ? [{id:'project-original',label:'Original audio',kind:'sfx',url:original,duration:0}] : []}, original ? 'Original audio' : 'No stems yet')
      if (info.metadata) suggestRoute(cfg, info.project_id)
    }
    setMessage('Project restored.')
    try {
      const saved = sessionStorage.getItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`)
      if (saved) { const pending = JSON.parse(saved); if (typeof pending.id === 'string' && typeof pending.sound === 'string') { setPrompt(pending.sound); await waitCandidate(cfg, info.project_id, pending.id) } }
    } catch (error) { throw new Error(`Project restored, but the preview could not be recovered: ${error instanceof Error ? error.message : String(error)}`) }
  })
  const restoreTab = useEffectEvent(() => {
    if (restored.current) return
    restored.current = true
    try {
      const saved = sessionStorage.getItem(`soundcanvas-project:${baseUrl}`)
      if (saved) { const credentials = JSON.parse(saved); if (typeof credentials.id === 'string' && typeof credentials.token === 'string') void resume(credentials.id, credentials.token) }
    } catch { /* The manual resume form remains available. */ }
  })
  useEffect(() => { if (readyToRestore) restoreTab() }, [baseUrl, readyToRestore])
  const uploadMedia = (media: File) => run(async () => {
    await flush()
    const created = await createProject(baseUrl)
    if (!created.access_token) throw new Error('Project token was not returned.')
    const cfg = {baseUrl, token:created.access_token}
    setProject(created); setToken(created.access_token); rememberProject(created.project_id, created.access_token);
    setMockStems(null);setJobId(''); setUndoId(''); setRouting(null); setSuggestion(null);setAnchors([]); setSettingsStems([]); setMessages([]); setRetryStems([]); setCandidateId(''); setCandidateUrl(''); setCurrentUrl(''); onOriginalAudio(null); await onLoadStems({load: async () => []}, 'No stems yet'); onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
    setMessage('Uploading and extracting audio…')
    await uploadProject(cfg, created.project_id, media)
    attachedVideo.current = media
    const info = await getProject(cfg, created.project_id); setProject(info)
    const original=await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(created.project_id)}/original-audio`, 3_600_000)
    onOriginalAudio(original)
    await onLoadStems({load:async()=>[{id:'project-original',label:'Original audio',kind:'sfx',url:original,duration:0}]}, 'Original audio')
    setFormat(info.metadata?.has_video ? 'mp4' : 'wav')
    await loadVideo(cfg, info)
    suggestRoute(cfg, created.project_id)
    setMessage(info.metadata?.has_video ? 'Uploaded. Pick a sound to separate.' : 'Uploaded. Split it into tracks, or pick a sound to separate.')
  })
  const previewSound = async (cfg:BackendConfig, info:ProjectInfo, sound:string, scope:EditScope, visual:unknown={}, stemId:string|null=null, quality=false, suppliedAnchors=anchors, passCount=1) => {
    await flush()
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMixUrl('')
    try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`) } catch { /* Ignore unavailable storage. */ }
    const id=await createCandidate(cfg,info.project_id,stemId,sound,quality?2:1,{guidance:{anchors:suppliedAnchors},edit_scope:scope,visual_prompt:visual,profile:quality?'quality':'preview',passes:passCount})
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`, JSON.stringify({id,sound})) } catch { /* Preview still works without tab recovery. */ }
    await waitCandidate(cfg,info.project_id,id)
  }
  const process = (quality=false) => run(async () => {
    if (!project) return
    const instruction=separationCommand(prompt)
    const sound=instruction?.description || prompt.trim()
    const scope:EditScope=instruction?.editScope || {action:instruction?.remove?'mute':editAction,...(timedEdit && editAction==='mute'?{interval:{start:editStart,end:editEnd}}:{})}
    let visual:unknown={}
    if(selection)visual={...selection as object,interval:{start_seconds:Number(rangeStart),end_seconds:Number(rangeEnd)}}
    await previewSound(config,project,sound,scope,visual,null,quality,anchors,clampPasses(passes,choices))
  })
  const splitSong = () => run(async () => {
    if (!project) return
    await splitInto(config, project, null)
  })
  useImperativeHandle(ref, () => ({ uploadMedia, splitSong, visualAction: async (chosen, description, action) => {
    if (!project || !token) throw new Error('Upload a video first.')
    if (!description.trim()) throw new Error('Describe the sound to isolate.')
    if (busyRef.current) throw new Error('A project operation is already running.')
    busyRef.current = true; setBusy(true); setError(''); setMessage('Preparing visual selection…')
    try {
      await flush()
      setPrompt(description);setEditAction(action==='isolate'?'keep':action)
      await previewSound(config,project,description,{action:action==='isolate'?'keep':action},chosen)
    } catch(e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { busyRef.current = false; setBusy(false) }
  }, separateVideo: async (video, description, remove) => {
    if (busyRef.current) throw new Error('A project operation is already running. Please wait.')
    busyRef.current = true; setError(''); setMessage('Uploading video for separation…'); setBusy(true)
    try {
      await flush()
      const created=await createProject(baseUrl)
      if (!created.access_token) throw new Error('Project token was not returned.')
      const cfg={baseUrl,token:created.access_token}
      setProject(created); setToken(cfg.token); rememberProject(created.project_id, cfg.token);
      setJobId(''); setSettingsStems([]); setMessages([]); onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
      await uploadProject(cfg,created.project_id,video)
      attachedVideo.current=video
      const info=await getProject(cfg,created.project_id); setProject(info); setFormat('mp4')
      setMessage('Separating audio…')
      await previewSound(cfg,info,description,{action:remove?'mute':'keep'})
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { busyRef.current = false; setBusy(false) }
  }, planCommand: async (text, video, stems, tracks, duration) => {
    if (busyRef.current) throw new Error('A project operation is already running. Please wait.')
    busyRef.current = true; setError(''); setMessage('Understanding your request…'); setBusy(true)
    try {
      await flush()
      let current=project, currentToken=token, currentJob=jobId
      if (video && attachedVideo.current !== video) {
        const created=await createProject(baseUrl)
        if (!created.access_token) throw new Error('Project token was not returned.')
        currentToken=created.access_token
        const cfg={baseUrl,token:currentToken}
        setProject(created); setToken(currentToken); rememberProject(created.project_id, currentToken);
        setJobId(''); currentJob=''; setSettingsStems([]); setMessages([])
        onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
        setMessage('Uploading video…')
        await uploadProject(cfg,created.project_id,video); attachedVideo.current=video
        current=await getProject(cfg,created.project_id); setProject(current); setFormat('mp4')
      }
      if (!current || !currentToken) throw new Error('Upload a video or song first to use AI commands.')
      const cfg={baseUrl,token:currentToken}
      const hasStems=Boolean(currentJob)
      // Without the AI planner, plain "isolate X" / "remove X" still works through the local parser.
      const local=separationCommand(text), music=musicCommand(text)
      let result:AgentPlan=await planProjectCommand(cfg,current.project_id,{prompt:text, stems:hasStems ? stems.map(({id,label,kind,duration:d})=>({id,label,kind,duration:d})) : [],tracks:hasStems ? tracks : [],duration:hasStems ? duration : (current.metadata?.duration || duration),has_project_stems:hasStems,selected_visual:selectedVisual ? {label:selectedVisual.label,start:selectedVisual.interval.start_seconds,end:selectedVisual.interval.end_seconds} : null}).catch((e:unknown):AgentPlan=>{
        if(music)return {kind:'separate',description:music.description,remove:music.remove,separation_mode:'music_stems',stems:music.stems,...(music.remove?{edit_scope:{action:'mute'}}:{}),use_visual:false,actions:[],message:''}
        if(!local)throw e;return {kind:'separate',description:local.description,remove:local.remove,...(local.editScope?{edit_scope:local.editScope}:{}),use_visual:false,actions:[],message:''}})
      // "Remove the vocals" on a split song mutes its Vocals track; no new separation needed.
      if (result.kind === 'separate' && hasStems) {
        const muted = muteExistingStem(result, await getSavedStems(cfg, current.project_id))
        if (muted) { setMessage(muted.message); return muted }
      }
      if (result.kind === 'separate' && result.separation_mode === 'music_stems') {
        if (hasStems) throw new Error('This project already has layers. Upload the song again to split it into tracks.')
        const scope:EditScope|undefined = result.edit_scope && result.edit_scope.action !== 'keep' ? result.edit_scope : result.remove ? {action:'mute'} : undefined
        await splitInto(cfg, current, result.stems ?? null, scope)
        result = {...result, message: result.remove ? 'Split done; the vocals are muted.' : 'Split into tracks.'}
        return result
      }
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
        const {scope, anchors:guidance} = separationSettings(result, anchors, editAction, timedEdit, editStart, editEnd)
        setPrompt(result.description); setEditAction(scope.action)
        await previewSound(cfg,current,result.description,scope,result.use_visual?selectedVisual:{},null,false,guidance,clampPasses(passes,choices))
        result.message='Preview ready. Compare it above the timeline, then apply.'
      } else setMessage(result.message)
      return result
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { busyRef.current = false; setBusy(false) }
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
        await downloadExport(config,info.download_url,`tidepool.${extension}`); setMessage('Export downloaded.'); return
      }
      setMessage('Rendering your export…'); await new Promise(resolve => setTimeout(resolve,1000))
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
  /** Rerun the selected target's prompt on the residual, then reload both with their new audio. */
  const extractMore = () => run(async () => {
    const stem=settingsStems.find(s => s.id===settingId)
    if (!project || !jobId || !stem) return
    await flush()
    abort.current?.abort(); const controller = new AbortController(); abort.current = controller
    setRefining(true)
    try {
      await runRefine(config, project.project_id, stem.id, {
        signal: controller.signal, onUpdate: status => setMessage(`Extracting more: ${status.state}${status.progress !== undefined ? ` ${status.progress}%` : ''}`),
      })
    } catch (e) {
      if (!mounted.current) return
      if (isAbort(e) || controller.signal.aborted) { setMessage(CANCELLED_MESSAGE); return }
      const refused = refineConflictMessage(e)
      if (refused === null) throw e
      setMessage(refused); return
    } finally { if (mounted.current) setRefining(false) }
    // Same reload as every other job: fresh blob URLs, decoded again by the engine.
    await onLoadStems(new HttpStemSource(config, jobId), 'Project stems')
    if (!mounted.current) return
    setSettingsStems(await getSavedStems(config, project.project_id))
    setMessage('Extracted more of the sound. The target and residual were reloaded.')
  })
  const settingStem = settingsStems.find(s => s.id===settingId)
  const refineBlocked = settingStem ? refineBlockReason(settingStem, settingsStems, {busy, supported: capabilities?.refine}) : 'Load stem settings first.'
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
    await previewSound(config,project,retryPrompt,{action:'keep'},selection || {},retryStem,quality,anchors,clampPasses(retryPasses,choices))
  })
  const useRetry = () => run(async () => {
    if(!candidateId || !project) return
    await flush();const applied=await applyCandidate(config,candidateId)
    await onLoadStems(new HttpStemSource(config,applied.job_id),'Project stems')
    setMockStems(candidateMock);setJobId(applied.job_id);setUndoId(candidateId);setCandidateId('');setRetryStems([])
    try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`) } catch { /* Ignore unavailable storage. */ }
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
  const hasProject = Boolean(project)
  const ready = Boolean(project?.metadata)
  const durationSeconds = project?.metadata?.duration
  // Where the user is in Upload, Pick a sound, Preview and apply, Export.
  const current: StepId = !ready ? 'upload' : candidateId ? 'preview' : jobId ? 'export' : 'sound'
  const flowKey = `${project?.project_id ?? ''}|${candidateId}|${jobId}`
  const openStep = manualStep?.key === flowKey ? manualStep.id : null
  const setOpenStep = (next: StepId | 'none' | ((prev: StepId | 'none' | null) => StepId | 'none')) => setManualStep({key:flowKey,id:typeof next === 'function' ? next(openStep) : next})
  const open = (id: StepId) => openStep === id || (openStep === null && current === id)
  const toggle = (id: StepId) => setOpenStep(prev => (prev === id || (prev === null && current === id)) ? 'none' : id)
  const status = (id: StepId): StepStatus => {
    const order: StepId[] = ['upload','sound','preview','export']
    if (id === 'upload' && ready) return 'done'
    if (id === 'sound' && (candidateId || jobId)) return 'done'
    if (id === 'preview' && jobId && !candidateId) return 'done'
    return order.indexOf(id) <= order.indexOf(current) ? 'active' : 'todo'
  }
  const fileName = file?.name || project?.filename
  const comparison = candidateId ? (
    <div className="rise-in shadow-card w-[min(34rem,calc(100%-2rem))] rounded-2xl border border-line bg-surface p-4" aria-label="Preview ready">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-1.5 text-[11px] font-medium text-accent"><FeatureTag name={FEATURES.compare}/>Preview ready</p>
          <p className="mt-0.5 text-sm font-semibold text-fg">{prompt.trim() || 'Separated sound'}</p>
        </div>
        <button type="button" className="rounded-full px-2 py-1 text-xs text-muted hover:bg-tint/[0.06] hover:text-fg" onClick={() => { setCandidateId(''); if (project) { try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`) } catch { /* Ignore unavailable storage. */ } } setMessage('Preview discarded.') }}>Discard</button>
      </div>
      <CandidateComparison key={candidateId} original={currentUrl} edited={mixUrl} target={candidateUrl}/>
      <div className="mt-3 flex items-center gap-2">
        <button className={`${button} flex-1 py-2 text-sm`} disabled={busy} onClick={useRetry}>Apply to timeline</button>
        <button className={secondary} disabled={busy || !capabilities?.quality} title={capabilities?.quality ? 'Deep dive: a slower, higher-quality pass' : (capabilities?.reason || '')} onClick={() => void process(true)}>{FEATURES.quality}</button>
      </div>
      {selectionName && <button className="mt-2 text-xs text-accent hover:underline" disabled={busy} onClick={()=>{setSelection(null);setSelectionName('');onClearSelection();setCandidateId('');setOpenStep('sound');setMessage('Mark where you hear the sound, then preview again using text and time.')}}>Not right? Retry with text and time instead of the object</button>}
    </div>
  ) : null

  return <section className="shadow-card flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-line bg-surface text-fg" aria-label="Project controls">
    <div className="min-h-0 flex-1 overflow-y-auto p-2">
      <ol className="space-y-1">
        <Step n={1} title="Upload" status={status('upload')} open={open('upload')} onToggle={() => toggle('upload')}
          summary={ready ? `${fileName || 'Video'}${durationSeconds ? ` · ${durationSeconds.toFixed(1)}s` : ''}` : undefined}>
          <label className={`flex cursor-pointer flex-col items-center gap-1 rounded-xl border border-dashed px-3 py-3 text-center transition-colors ${file ? 'border-accent/50 bg-accent-soft' : 'border-line-strong hover:border-accent/50 hover:bg-accent-soft/60'}`}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-accent" aria-hidden="true"><path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
            <span className="text-xs font-medium text-fg">{file ? file.name : 'Choose a video or audio file'}</span>
            <span className="text-[11px] text-subtle">MP4, MOV, WAV, MP3, FLAC or M4A. You can also drop it anywhere.</span>
            <input className="sr-only" type="file" accept=".mp4,.mov,.wav,.mp3,.flac,.m4a,video/*,audio/*" aria-label="Media" disabled={busy} onChange={e => { const f=e.target.files?.[0] || null; setFile(f); if (f) void uploadMedia(f) }} />
          </label>
          {project && <details className="mt-2 text-xs"><summary className="cursor-pointer">Project credentials</summary><p className="mt-1 break-all">ID: {project.project_id}</p><label className="block mt-1">Token<input className={field} readOnly value={token} type="text" onFocus={e=>e.currentTarget.select()}/></label><p className="text-muted">Keep these private. Copy them to resume after closing this tab.</p></details>}
          <details className="mt-2 text-xs"><summary className="cursor-pointer">Resume a project</summary><div className="mt-2 space-y-1"><input className={field} value={resumeId} onChange={e=>setResumeId(e.target.value)} placeholder="Project ID" aria-label="Project ID"/><input className={field} type="password" value={resumeToken} onChange={e=>setResumeToken(e.target.value)} placeholder="Project token" aria-label="Project token"/><button className={secondary} disabled={busy} onClick={()=>void resume(resumeId,resumeToken)}>Resume</button></div></details>
        </Step>

        <Step n={2} title="Pick a sound" tag={FEATURES.separate} status={status('sound')} open={open('sound') && hasProject} onToggle={() => hasProject && toggle('sound')}
          summary={status('sound') === 'done' && prompt.trim() ? prompt.trim() : undefined}>
          <div className="space-y-3">
            {project?.metadata?.has_video && capabilities?.visual === false && <p className="text-muted">The current separator does not support visual object guidance.</p>}
            {project?.metadata?.has_video && <ObjectsPanel key={project.project_id} config={config} projectId={project.project_id} vision={capabilities?.vision} visual={visual} selection={selectedVisual} disabled={busy || capabilities?.visual === false} resumeJobId={resumeVisual?.projectId===project.project_id ? resumeVisual.jobId : null} canPrepare={!jobId} onAnalyzed={visualAnalyzed} onSelect={onSelectVisual} onSeek={onSeek} onPrepared={reloadPreparedMedia}/>}
            <div className="rounded-xl bg-surface-2 p-3 text-xs">
              {prompt.trim() ? <div className="flex items-center gap-2"><span className="text-subtle">Target</span><span className="rounded-full bg-accent-soft px-2 py-0.5 font-medium text-accent">{prompt.trim()}</span></div>
                : <p className="text-muted">{visual ? 'Click an object in the video, or describe' : 'Describe'} the sound in the bar below, like <span className="text-fg">“isolate the dog barking”</span>.</p>}
              {selection !== null && <div className="mt-2"><GuidedByChip selection={selection as {label?:unknown;track_id?:unknown}} start={Number(rangeStart)} end={Number(rangeEnd)} onClear={()=>{setSelection(null);onClearSelection();setSelectionName('')}}/></div>}
            </div>
            {project && <details className="group text-xs">
              <summary className="cursor-pointer list-none text-muted hover:text-fg"><span className="inline-block transition-transform group-open:rotate-90">›</span> Fine-tune</summary>
              <div className="mt-2 space-y-2">
                {capabilities?.spanAnchors === false ? <p className="text-muted">The current separator does not use sound time guidance. Timed edits are still available below.</p> : <SoundGuidance duration={project.metadata?.duration || 1} anchors={anchors} onChange={setAnchors}/>}
                <label className="block">After separating<select aria-label="Edit after separation" className={`${field} mt-1`} value={editAction} onChange={e=>setEditAction(e.target.value as EditScope['action'])}><option value="keep">Keep it as its own layer</option><option value="mute">Mute it</option><option value="lower">Lower it by half</option></select></label>
                {editAction==='mute' && <div><label className="flex items-center gap-1.5"><input type="checkbox" className="accent-accent" checked={timedEdit} onChange={e=>setTimedEdit(e.target.checked)}/>Only mute between</label>{timedEdit && <div className="mt-1 flex gap-1"><label>Start<input aria-label="Edit start" className={field} type="number" min={0} value={editStart} onChange={e=>setEditStart(Number(e.target.value))}/></label><label>End<input aria-label="Edit end" className={field} type="number" min={0} value={editEnd} onChange={e=>setEditEnd(Number(e.target.value))}/></label></div>}</div>}
                <PassesSelect className={field} value={clampPasses(passes,choices)} choices={choices} onChange={setPasses} disabled={busy}/>
              </div>
            </details>}
            {project && <button className={`${button} w-full py-2`} disabled={busy || !project.metadata || !prompt.trim()} onClick={()=>void process()}>Separate with {FEATURES.separate}</button>}
            {routing && <p className="text-[11px] text-subtle" title="How the audio service separated this project">{describeRouting(routing)}</p>}
            {(project?.metadata?.audio_quality?.findings.length ?? 0) > 0 && <details className="text-xs text-amber-700"><summary className="cursor-pointer">Notes about this clip ({project?.metadata?.audio_quality?.findings.length})</summary><ul className="mt-1 list-disc space-y-1 pl-4">{project?.metadata?.audio_quality?.findings.map(f=><li key={f}>{f}</li>)}</ul></details>}
          </div>
        </Step>

        <Step n={3} title="Preview and apply" tag={FEATURES.compare} status={status('preview')} open={open('preview') && (Boolean(candidateId) || Boolean(undoId))} onToggle={() => toggle('preview')}
          summary={status('preview') === 'done' ? 'Applied to the timeline' : undefined}>
          <div className="space-y-2 text-xs">
            {candidateId ? <p className="text-muted">Compare the original, the edited mix and the isolated sound above the timeline, then apply.</p> : null}
            {undoId && <button className={secondary} disabled={busy} onClick={undoPreview}>Undo the last apply</button>}
          </div>
        </Step>

        <Step n={4} title="Export" tag={FEATURES.export} status={status('export')} open={open('export') && Boolean(jobId)} onToggle={() => jobId && toggle('export')}>
          {project && <div className="flex gap-2"><select className={field} aria-label="Export format" value={format} onChange={e=>setFormat(e.target.value as typeof format)}><option value="mp4" disabled={!project.metadata?.has_video}>MP4 video</option><option value="wav">WAV audio</option><option value="mp3">MP3 audio</option></select><button className={button} disabled={busy || !jobId} onClick={exportFile}>Export</button></div>}
        </Step>
      </ol>
    </div>

    {(busy || message || error) && <div className="border-t border-line px-3 py-2 text-xs">
      {busy && <p className="flex items-center gap-2 text-accent"><DropLoader size={22}/>{message || 'Working…'}</p>}
      {!busy && message && !error && <p role="status" className="text-muted">{message}</p>}
      {error && <p role="alert" className="break-words text-red-600">{error}</p>}
    </div>}

    <details className="border-t border-line text-xs">
      <summary className="cursor-pointer list-none px-3 py-2 text-subtle hover:text-fg">Advanced</summary>
      <div className="max-h-56 space-y-3 overflow-y-auto px-3 pb-3">
        {project && <div><p className="font-medium text-fg">Import detections</p><p className="my-1 text-muted">Detection and selection JSON files that match this video, instead of Find objects.</p><label className="block">Detections<input className={field} type="file" accept=".json" disabled={busy} onChange={e=>{const f=e.target.files?.[0]; if(f) void run(async()=>{const data=asVisualMetadata(await readJson(f));await storeVisualMetadata(config,project.project_id,data);onVisual(data);setVisualName(f.name);setMessage('Detections saved.')})}} /></label>{visualName && <span>{visualName}</span>}<label className="block mt-1">Selection<input className={field} type="file" accept=".json" disabled={busy} onChange={e=>{const f=e.target.files?.[0]; if(f) void run(async()=>{const data=await readJson(f) as VisualSelection; if (!data?.interval) throw new Error('Selection JSON needs an interval.'); setSelection(data);setRangeStart(String(data.interval.start_seconds));setRangeEnd(String(data.interval.end_seconds));setSelectionName(f.name);setMessage('Selection ready.')})}} /></label>{selectionName && <><button className="text-accent" onClick={()=>{setSelection(null);onClearSelection();setSelectionName('')}}>Clear {selectionName}</button><div className="flex gap-1"><label>Start (seconds)<input className={field} type="number" min="0" step="any" value={rangeStart} onChange={e=>setRangeStart(e.target.value)} /></label><label>End (seconds)<input className={field} type="number" min="0" step="any" value={rangeEnd} onChange={e=>setRangeEnd(e.target.value)} /></label></div></>}</div>}
        {jobId && <div><p className="font-medium text-fg">Layer settings</p><button className={`${secondary} mt-1`} disabled={busy} onClick={refreshSettings}>Load layer settings</button>{settingsStems.length>0 && <div className="mt-1 space-y-1"><select className={field} aria-label="Stem" value={settingId} onChange={e=>{const stem=settingsStems.find(s=>s.id===e.target.value);if(stem){setSettingId(stem.id);setStemName(stem.name);setOffset(String(stem.start_offset || 0));setClearRanges(false)}}}>{settingsStems.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select><label>Name<input className={field} value={stemName} onChange={e=>setStemName(e.target.value)} /></label><label>Start offset (seconds)<input className={field} type="number" min="0" step="any" value={offset} onChange={e=>setOffset(e.target.value)} /></label><label className="flex items-center gap-1"><input type="checkbox" checked={clearRanges} onChange={e=>setClearRanges(e.target.checked)} />Clear remote mute ranges</label><button className={secondary} disabled={busy} onClick={saveSettings}>Save layer settings</button><div className="flex items-center gap-1"><span title={refineBlocked ?? 'Rerun this sound\'s prompt on the residual to pull out what the first pass only quieted.'}><button className={secondary} disabled={Boolean(refineBlocked)} onClick={extractMore}>{FEATURES.refine}: extract more</button></span>{refining && <button className={secondary} onClick={() => abort.current?.abort()}>Cancel</button>}</div>{refineBlocked && !busy && <p className="text-muted">{refineBlocked}</p>}</div>}</div>}
        {jobId && <div><p className="font-medium text-fg">Try another separation</p>
          <p className="my-1 text-muted">Replace a target in a two-layer project. For multitrack projects, a new preview splits the current residual.</p>
          <button className={secondary} disabled={busy} onClick={prepareRetry}>Choose sound</button>
          {retryStems.length>0 && <div className="mt-1 space-y-2"><label>Sound to find instead<input className={field} value={retryPrompt} onChange={e=>setRetryPrompt(e.target.value)}/></label><PassesSelect className={field} value={clampPasses(retryPasses,choices)} choices={choices} onChange={setRetryPasses} disabled={busy}/><div className="flex gap-1"><button className={secondary} disabled={busy} onClick={()=>void retry()}>Preview replacement</button><button className={secondary} disabled={busy || !capabilities?.quality} onClick={()=>void retry(true)}>Higher quality</button></div></div>}
        </div>}
        {project && <div><p className="font-medium text-fg">Remote command test</p><div className="mt-1 flex gap-1"><input className={field} value={command} onChange={e=>setCommand(e.target.value)} placeholder="Mute the dog"/><button className={secondary} disabled={busy || !command.trim()} onClick={sendCommand}>Send</button></div><button className={`${secondary} mt-1`} disabled={busy} onClick={()=>void run(async()=>{setMessages(await getMessages(config,project.project_id));setMessage('Messages updated.')})}>Refresh messages</button>{messages.map(m=><p key={m.id} className="mt-1 text-muted">{m.text}</p>)}</div>}
      </div>
    </details>
    {previewSlot && comparison && createPortal(comparison, previewSlot)}
  </section>
})

type StepId = 'upload' | 'sound' | 'preview' | 'export'
type StepStatus = 'done' | 'active' | 'todo'

function Step({ n, title, tag, status, open, onToggle, summary, children }: { n: number; title: string; tag?: string; status: StepStatus; open: boolean; onToggle: () => void; summary?: string; children: ReactNode }) {
  const item = useRef<HTMLLIElement>(null)
  // Keep the step the flow just moved to in view: the panel is short and scrolls.
  useEffect(() => { if (open) item.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }) }, [open])
  return <li ref={item} className={`rounded-xl transition-colors ${open ? 'bg-surface-2/70' : ''}`}>
    <button type="button" onClick={onToggle} aria-expanded={open} className="flex w-full items-center gap-2.5 rounded-xl px-2 py-1.5 text-left hover:bg-surface-2">
      <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold transition-colors ${status === 'done' ? 'bg-accent text-white' : status === 'active' ? 'bg-accent-soft text-accent ring-1 ring-accent/40' : 'bg-surface-2 text-faint'}`}>
        {status === 'done' ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"/></svg> : n}
      </span>
      <span className="min-w-0 flex-1">
        <span className={`flex items-center gap-1.5 text-[13px] font-medium ${status === 'todo' ? 'text-faint' : 'text-fg'}`}>{title}{tag && <FeatureTag name={tag} className={status === 'todo' ? 'opacity-50' : ''} />}</span>
        {summary && !open && <span className="block truncate text-[11px] text-subtle">{summary}</span>}
      </span>
    </button>
    {open && <div className="rise-in px-2 pb-3 pl-10">{children}</div>}
  </li>
}
