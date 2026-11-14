import { forwardRef, useEffect, useEffectEvent, useImperativeHandle, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { Anchor, EditScope, Capabilities, AgentPlan, BackendConfig, ProjectInfo, Routing, SavedStem, VisualMetadata, CinematicStyle, DirectorIntent, DiscoveredSound, DirectorChange, LayerEffects, FxPreviewRequest, ReplacementEvent, ReplacementAdjustment, ReplacementRecovery, SfxPlacement } from '../api/backend'

import { asVisualMetadata } from '../visual'
import type { VisualSelection } from '../visual'
import { musicCommand, separationCommand } from '../separationCommand'
import { separationSettings } from '../agent/separationSettings'
import { muteExistingStem } from '../agent/planOutcome'
import { splitSuggestion } from '../state/projectStems'
import { cinematicCommand } from '../agent/cinematic'
import { replacementFollowup } from '../agent/replacementFollowup'
import { requestsSoundReplacement } from '../agent/replacementIntent'
import { placementLines, sfxFollowup } from '../agent/sfxIntent'
import { chatRoute } from '../agent/chatRoute'
import { creativeCommand, directorIntent } from '../agent/director'
import type { Stem, TrackState } from '../types'
import type { ProjectStemsState } from '../state/projectStems'
import { addMuteRange, describeRouting, getRoute, splitTracks, toRouting, getCapabilities, undoCandidate, createCandidate, createCinematicPreview, createDirectorPreview, createAutomaticReplacement, createReplacementPreview, createSfxPreview, adjustReplacementPreview, createEffectsPreview, createFxPreview, fxRequest, adjustDirectorPreview, getCandidate, startSoundDiscovery, getSoundDiscovery, applyCandidate, planProjectCommand, createProject, downloadExport, fetchStemBlobUrl, getExport, getJob, getMessages, getProject, getSavedStems, localCommand, startExport, storeVisualMetadata, updateStemSettings, uploadProject } from '../api/backend'
import { jobProgressMessage, pollJob, toJobStatus } from '../api/jobs'

import { CANCELLED_MESSAGE, isAbort } from '../api/errors'
import { clampPasses, passChoices, refineBlockReason, refineConflictMessage, runRefine } from '../api/refine'
import { HttpStemSource } from '../api/stemSources'
import type { StemSource } from '../api/stemSources'

import { SoundGuidance } from './SoundGuidance'
import { ProcessingRange } from './ProcessingRange'
import type { ProcessingRangeValue } from './ProcessingRange'
import { CandidateComparison } from './CandidateComparison'
import { ReplacementPanel } from './ReplacementPanel'
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
  /** Preview a layer without one of its applied effects (a chip on its track). */
  removeFx: (stemId: string, chipId: string, label: string) => Promise<void>
}

/** The project token stays in this browser tab and is sent only as a bearer header. */
export const ProjectPanel = forwardRef<ProjectPanelHandle, Props>(function ProjectPanel({baseUrl, readyToRestore, onProjectState, onLoadStems, onVideo, onOriginalAudio, flush, onVisual, selectedVisual, onClearSelection, onSelectVisual, onSeek, visual, onBusy}, ref) {
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
  const [processingRange,setProcessingRange]=useState<ProcessingRangeValue|null>(null)
  const [processingTime,setProcessingTime]=useState<number|null>(null)
  const [suggestions,setSuggestions]=useState<DiscoveredSound[]>([])
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
  const [candidateKind,setCandidateKind]=useState<'separation'|'cinematic'|'director'|'effects'|'fx'|'replacement'|'sfx'>('separation')
  const [sfxPlacements,setSfxPlacements]=useState<SfxPlacement[]>([])
  const [fxInfo,setFxInfo]=useState<{label:string;op:string;targets:string[]}>({label:'',op:'',targets:[]})
  const [cinematicStyle,setCinematicStyle]=useState<CinematicStyle>('cinematic')
  const [directorRequest,setDirectorRequest]=useState('Make the whole video more cinematic')
  const [directorGoal,setDirectorGoal]=useState<DirectorIntent>('cinematic')
  const [directorChanges,setDirectorChanges]=useState<DirectorChange[]>([])
  const [replacementPeaks,setReplacementPeaks]=useState<number[]>([])
  const [replacementCount,setReplacementCount]=useState<number|null>(null)
  const [replacementMode,setReplacementMode]=useState<'replace'|'add'>('replace')
  const [replacementEvents,setReplacementEvents]=useState<ReplacementEvent[]>([])
  const [replacementRecovery,setReplacementRecovery]=useState<ReplacementRecovery|null>(null)
  const [replacementAlignment,setReplacementAlignment]=useState<'peak'|'onset'>('peak')
  const [replacementAddTime,setReplacementAddTime]=useState('')
  const [previewSeek,setPreviewSeek]=useState<{time:number;id:number}|null>(null)
  const [replacementOpen,setReplacementOpen]=useState(false)
  const [replacementSource,setReplacementSource]=useState('')
  const [replacementEffect,setReplacementEffect]=useState('')
  const [effectsOpen,setEffectsOpen]=useState(false)
  const [effectsStems,setEffectsStems]=useState<SavedStem[]>([])
  const [effectsStem,setEffectsStem]=useState('')
  const [effects,setEffects]=useState<LayerEffects>({bass:0,treble:0,highpass:0,compression:1})
  const [cinematicSteps,setCinematicSteps]=useState<string[]>([])
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
  const pendingReplacement = useRef('')
  const pendingSfx = useRef('')
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
  async function waitCandidate(cfg:BackendConfig, projectId:string, id:string, label='Separating the sound') {
    const began=Date.now()
    for (;;) {
      if(!mounted.current)throw new Error('Preview cancelled')
      const job=await getCandidate(cfg,id)
      if(job.status==='failed') { if(job.recovery) setReplacementRecovery(job.recovery); try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${projectId}`) } catch { /* Ignore unavailable storage. */ }; throw new Error(job.error || 'Preview failed. Try a sound description with a time anchor.') }
      if(job.status==='completed') {
        setReplacementRecovery(null)
        if(!job.audio_url || !job.mix_url || !job.original_url)throw new Error('Preview audio is incomplete')
        const urls=await Promise.all([fetchStemBlobUrl(cfg,job.audio_url),fetchStemBlobUrl(cfg,job.mix_url),fetchStemBlobUrl(cfg,job.original_url)])
        setCandidateMock(job.inference?.model?.is_mock ?? null);setCandidateWarnings(job.inference?.warnings || [])
        setCandidateKind(job.provenance?.kind==='director' ? 'director' : job.provenance?.kind==='cinematic' ? 'cinematic' : job.provenance?.kind==='effects' ? 'effects' : job.provenance?.kind==='fx' ? 'fx' : job.provenance?.kind==='replacement' ? 'replacement' : job.provenance?.kind==='sfx_insert' ? 'sfx' : 'separation')
        setSfxPlacements(job.provenance?.kind==='sfx_insert' ? job.provenance.placements || [] : [])
        if (job.provenance?.kind==='fx') setFxInfo({label:job.provenance.label || 'Effect',op:job.provenance.op || 'chain',targets:(job.provenance.targets || []).map(t=>t.id)})
        if (job.provenance?.kind!=='cinematic' && job.provenance?.kind!=='director' && job.provenance?.kind!=='effects' && job.provenance?.kind!=='fx' && job.provenance?.kind!=='replacement' && job.provenance?.kind!=='sfx_insert') { setProcessingRange(job.provenance?.processing_interval || null); setProcessingTime(null) }
        if (job.provenance?.kind==='cinematic' || job.provenance?.kind==='director') { setCinematicSteps(job.provenance.steps || []); setCinematicStyle(job.provenance.style || 'cinematic'); if(job.provenance.kind==='director') { setDirectorRequest(job.provenance.request || 'Make the whole video more cinematic');setDirectorGoal(job.provenance.intent || 'cinematic');setDirectorChanges(job.provenance.changes || []) } }
        if(job.provenance?.kind==='replacement') {setReplacementPeaks(job.provenance.peak_times || []);setReplacementCount(job.provenance.requested_count ?? null);setReplacementMode(job.provenance.mode || 'replace');setReplacementEvents(job.provenance.events || []);setReplacementAlignment(job.provenance.alignment || 'peak')}
        setCandidateUrl(urls[0]);setMixUrl(urls[1]);setCurrentUrl(urls[2]);setCandidateId(id)
        setMessage(['Preview ready. Compare it above the timeline, then apply.',...(job.inference?.warnings || [])].join(' '));return
      }
      setMessage(job.status === 'queued' ? 'Waiting for processing…' : job.stage==='finding' ? 'Finding the sound…' : job.stage==='generating' ? 'Generating the replacement sound…' : job.stage==='separating' ? 'Replacing the sound…' : jobProgressMessage({state:'processing',progress:job.progress,stage:job.stage ?? undefined},Date.now()-began,label));await new Promise(resolve=>setTimeout(resolve,1000))

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
    // A split is whole-song; drop any range picked for prompted previews so it is not mistaken as applied.
    setProcessingRange(null); setProcessingTime(null)
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
    setUndoId(info.latest_undoable_candidate_id || ''); setCandidateId(''); setCandidateUrl(''); setMixUrl(''); setCurrentUrl(''); setRetryStems([]); setAnchors([]); setRouting(null); setSuggestion(null)

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
    setMockStems(null);setJobId(''); setUndoId('');setRouting(null);setSuggestion(null);setAnchors([]);setProcessingRange(null);setProcessingTime(null);setSuggestions([]);setReplacementOpen(false);setReplacementRecovery(null);setReplacementSource('');setReplacementEffect('');setEffectsOpen(false);setEffectsStems([]);setEffectsStem('');setEffects({bass:0,treble:0,highpass:0,compression:1}); setSettingsStems([]); setMessages([]); setRetryStems([]); setCandidateId(''); setCandidateUrl(''); setCurrentUrl(''); onOriginalAudio(null); await onLoadStems({load: async () => []}, 'No stems yet'); onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')

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
  const previewSound = async (cfg:BackendConfig, info:ProjectInfo, sound:string, scope:EditScope, visual:unknown={}, stemId:string|null=null, quality=false, suppliedAnchors=anchors, passCount=1, range:ProcessingRangeValue|null=processingRange, point:number|null=processingTime) => {
    await flush()
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMixUrl('')
    try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`) } catch { /* Ignore unavailable storage. */ }
    const id=await createCandidate(cfg,info.project_id,stemId,sound,quality?2:1,{guidance:{anchors:suppliedAnchors},edit_scope:scope,visual_prompt:visual,profile:quality?'quality':'preview',passes:passCount,...(range?{processing_interval:range}:point!==null?{processing_time:point}:{})})
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`, JSON.stringify({id,sound})) } catch { /* Preview still works without tab recovery. */ }
    await waitCandidate(cfg,info.project_id,id)
  }
  const discoverSounds = () => run(async () => {
    if (!project) return
    setMessage('Finding possible sounds…');setSuggestions([])
    const id=await startSoundDiscovery(config,project.project_id)
    for (;;) {
      const job=await getSoundDiscovery(config,id)
      if(job.status==='failed') throw new Error(job.error || 'Sound discovery failed.')
      if(job.status==='completed') {
        const result=job.result
        setSuggestions(result?.sources || [])
        setMessage(result?.sources.length ? 'Possible sounds found. Choose one and listen before applying.' : 'No supported sounds were found. You can still describe a sound manually.')
        break
      }
      setMessage('Finding possible sounds…');await new Promise(resolve=>setTimeout(resolve,1000))
    }
  })
  const previewCinematic = async (cfg:BackendConfig, info:ProjectInfo, style:CinematicStyle) => {
    await flush()
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMixUrl('');setPrompt('Cinematic audio')
    const id=await createCinematicPreview(cfg,info.project_id,style)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`, JSON.stringify({id,sound:'Cinematic audio'})) } catch { /* Preview remains available in this tab. */ }
    await waitCandidate(cfg,info.project_id,id)
  }
  const previewDirector = async (cfg:BackendConfig, info:ProjectInfo, requestText:string, intent:DirectorIntent, style:CinematicStyle) => {
    await flush()
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMixUrl('');setPrompt('Creative audio')
    const id=await createDirectorPreview(cfg,info.project_id,requestText,intent,style)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`, JSON.stringify({id,sound:'Creative audio'})) } catch { /* Preview remains available in this tab. */ }
    await waitCandidate(cfg,info.project_id,id)
  }
  const toggleDirectorChange = (change:DirectorChange) => run(async () => {
    if (!project || !candidateId) return
    setMessage('Updating preview…')
    const id=await adjustDirectorPreview(config,candidateId,change.target_stem_id,!change.enabled)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`, JSON.stringify({id,sound:'Creative audio'})) } catch { /* Preview still works without tab recovery. */ }
    await waitCandidate(config,project.project_id,id)
  })
  /** Render an effect plan as a candidate; it is compared, applied and undone like any preview. */
  const previewFx = async (cfg:BackendConfig, info:ProjectInfo, body:FxPreviewRequest, label:string) => {
    await flush()
    setCandidateId('');setCandidateUrl('');setCurrentUrl('');setMixUrl('');setPrompt(label);setMessage('Rendering the effect…')
    const id=await createFxPreview(cfg,info.project_id,body)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${info.project_id}`, JSON.stringify({id,sound:label})) } catch { /* Preview remains available in this tab. */ }
    await waitCandidate(cfg,info.project_id,id)
  }
  const adjustFx = (direction:'more'|'less') => run(async () => {
    if (!project || !candidateId) return
    await previewFx(config,project,{op:'adjust',targets:fxInfo.targets.length?fxInfo.targets:['mix'],direction,base_candidate_id:candidateId},`${direction==='more'?'More':'Less'}: ${fxInfo.label}`)
  })
  const adjustReplacement = (adjustment:ReplacementAdjustment) => run(async () => {
    if (!project || !candidateId) return
    setMessage('Updating sound placements…')
    const id=await adjustReplacementPreview(config,candidateId,adjustment)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`, JSON.stringify({id,sound:'Sound replacement'})) } catch { /* Preview remains available. */ }
    await waitCandidate(config,project.project_id,id)
  })
  const previewRecovery = () => run(async () => {
    if (!project || !replacementRecovery) return
    await flush();setMessage('Preparing an add-only preview…')
    const id=await createReplacementPreview(config,project.project_id,replacementRecovery)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`, JSON.stringify({id,sound:'Added sound'})) } catch { /* Preview remains available. */ }
    setPrompt('Added sound');await waitCandidate(config,project.project_id,id)
  })
  const openEffects = () => run(async () => {
    if (!project) return
    await flush()
    const rows=await getSavedStems(config,project.project_id)
    setEffectsStems(rows);setEffectsStem(rows[0]?.id || '');setEffectsOpen(true)
  })
  const previewEffects = () => run(async () => {
    if (!project) return
    await flush();setMessage('Preparing layer effects preview…')
    const id=await createEffectsPreview(config,project.project_id,effectsStem || null,effects)
    try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`, JSON.stringify({id,sound:'Layer effects'})) } catch { /* Preview remains available. */ }
    setPrompt('Layer effects');await waitCandidate(config,project.project_id,id)
  })
  const cinematic = (style:CinematicStyle) => run(async () => {
    if (!project) return
    setMessage('Analyzing and preparing the mix…')
    if (candidateKind==='director') await previewDirector(config,project,directorRequest,directorGoal,style)
    else await previewCinematic(config,project,style)
  })
  const process = (quality=false) => run(async () => {
    if (!project) return
    if (processingRange && (processingRange.start<0 || processingRange.end<=processingRange.start || processingRange.end>(project.metadata?.duration || 0))) throw new Error('Choose a valid processing range within the clip.')
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
  const removeFx = async (stemId:string, chipId:string, label:string) => {
    if (busyRef.current) throw new Error('A project operation is already running. Please wait.')
    if (!project) throw new Error('Open a project first.')
    busyRef.current=true;setError('');setBusy(true)
    try { await previewFx(config,project,{op:'remove',targets:[stemId],item_id:chipId},`Remove ${label}`) }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { busyRef.current=false;if(mounted.current)setBusy(false) }
  }
  useImperativeHandle(ref, () => ({ uploadMedia, splitSong, removeFx, visualAction: async (chosen, description, action) => {
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
      const plannerInput={prompt:text, stems:hasStems ? stems.map(({id,label,kind,duration:d})=>({id,label,kind,duration:d})) : [],tracks:hasStems ? tracks : [],duration:hasStems ? duration : (current.metadata?.duration || duration),has_project_stems:hasStems,selected_visual:selectedVisual ? {label:selectedVisual.label,start:selectedVisual.interval.start_seconds,end:selectedVisual.interval.end_seconds} : null}
      // Library sound effects ("vine boom when he says 'what'") come before replacement, which also matches "add X at 0:12".
      let prefetched:AgentPlan|null=null
      const sfxText=sfxFollowup(pendingSfx.current,text) || text
      pendingSfx.current=''
      // Effects ("add reverb to the vocals") and inserts ("add a vine boom at 0:12") share words; chatRoute decides.
      const route=chatRoute(sfxText,hasStems ? stems : [])
      if (!pendingReplacement.current && route.sfx) {
        const sfxPlan=await planProjectCommand(cfg,current.project_id,{...plannerInput,prompt:sfxText}).catch(()=>null)
        if (sfxPlan?.kind==='insert' && sfxPlan.inserts?.length) {
          setCandidateId('');setReplacementRecovery(null);setPrompt('Sound effects');setMessage(sfxPlan.message || 'Placing sound effects…')
          const id=await createSfxPreview(cfg,current.project_id,sfxPlan.inserts,sfxText)
          try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${current.project_id}`,JSON.stringify({id,sound:'Sound effects'})) } catch { /* Preview can still run. */ }
          await waitCandidate(cfg,current.project_id,id,'Placing sound effects')
          return {...sfxPlan,message:'Sound effect preview ready. Compare, then apply to add the new layers.'}
        }
        if (sfxPlan?.kind==='clarify' && !(sfxPlan.sfx_unmatched && requestsSoundReplacement(text))) {
          if (sfxPlan.sfx_needs_time && sfxPlan.sfx_query) pendingSfx.current=sfxPlan.sfx_query
          setMessage(sfxPlan.message); return sfxPlan
        }
        if (sfxPlan && sfxPlan.kind!=='clarify') prefetched=sfxPlan
      }
      // Without the AI planner, plain "isolate X" / "remove X" still works through the local parser.
      const replacementText=replacementFollowup(pendingReplacement.current,text) || text
      // "Make the drums sound like a phone" is a layer effect, not a sound replacement.
      const fxLocal=route.fx
      const replacement=!fxLocal && requestsSoundReplacement(replacementText)
      if (replacement) {
        setCandidateId('');setReplacementRecovery(null);setPrompt('Sound replacement');setMessage('Finding and replacing the sound…')
        pendingReplacement.current=''
        const id=await createAutomaticReplacement(cfg,current.project_id,replacementText)
        try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${current.project_id}`,JSON.stringify({id,sound:'Sound replacement'})) } catch { /* Preview can still run. */ }
        try { await waitCandidate(cfg,current.project_id,id) } catch (error) {
          if (error instanceof Error && /example timestamp/i.test(error.message)) pendingReplacement.current=replacementText
          throw error
        }
        return {kind:'replacement',description:null,remove:false,actions:[],message:'Replacement preview ready. Compare before applying.'}
      }
      pendingReplacement.current=''
      const local=separationCommand(text), music=musicCommand(text)

      const directCinematic = cinematicCommand(text, candidateKind==='cinematic'||candidateKind==='director')
      const directCreative = creativeCommand(text)
      const directTimestamp = local?.processingTime !== undefined
      let result:AgentPlan=directTimestamp && local ? {kind:'separate',description:local.description,remove:local.remove,edit_scope:local.editScope || {action:local.remove?'mute':'keep'},processing_time:local.processingTime,actions:[],message:''} : directCinematic || directCreative ? {kind:'director',description:null,remove:false,actions:[],message:''} : await (prefetched ? Promise.resolve(prefetched) : planProjectCommand(cfg,current.project_id,plannerInput)).catch((e:unknown):AgentPlan=>{
        // Without the AI planner, common effect asks still work through the rule-based parser.
        if(fxLocal)return fxLocal.kind==='clarify' ? {kind:'clarify',description:null,remove:false,actions:[],message:fxLocal.message} : {kind:fxLocal.kind,fx:fxLocal.fx,description:null,remove:false,actions:[],message:fxLocal.message}
        if(music)return {kind:'separate',description:music.description,remove:music.remove,separation_mode:'music_stems',stems:music.stems,...(music.remove?{edit_scope:{action:'mute'}}:{}),use_visual:false,actions:[],message:''}
        if(!local)throw e;return {kind:'separate',description:local.description,remove:local.remove,...(local.editScope?{edit_scope:local.editScope}:{}),use_visual:false,actions:[],message:''}})
      if ((result.kind==='fx' || result.kind==='duck' || result.kind==='loudness') && result.fx) {
        const plan=result.fx
        const label=result.kind==='duck' ? 'Auto-ducking' : result.kind==='loudness' ? `Loudness ${plan.loudness_lufs ?? -14} LUFS` : plan.adjust==='remove' ? 'Remove effect' : plan.adjust ? `${plan.adjust==='more'?'More':'Less'} effect` : text
        await previewFx(cfg,current,fxRequest(result.kind,plan,candidateKind==='fx' && candidateId ? candidateId : undefined),label)
        result.message='Effect preview ready. Compare it, then apply or undo.'
        return result
      }
      // "Remove the vocals" on a split song mutes its Vocals track; no new separation needed.
      if (result.kind === 'separate' && hasStems) {
        const muted = muteExistingStem(result, await getSavedStems(cfg, current.project_id))
        if (muted) { setMessage(muted.message); return muted }
      }
      if (result.kind === 'separate' && result.separation_mode === 'music_stems') {
        if (hasStems) throw new Error('This project already has layers. Upload the song again to split it into tracks.')
        // A song split is whole-song; the planner never pairs it with a timestamp.
        if (result.processing_time != null) throw new Error('Splitting a song into tracks covers the whole song, not one moment.')
        const scope:EditScope|undefined = result.edit_scope && result.edit_scope.action !== 'keep' ? result.edit_scope : result.remove ? {action:'mute'} : undefined
        await splitInto(cfg, current, result.stems ?? null, scope)
        result = {...result, message: result.remove ? 'Split done; the vocals are muted.' : 'Split into tracks.'}
        return result
      }
      if (result.kind === 'cinematic' || result.kind === 'director') {
        setMessage('Analyzing sounds and preparing the mix…')
        const isRefinement=Boolean(directCinematic && candidateKind==='director' && /(?:more dramatic|less dramatic|more subtle|subtler|gentler)/i.test(text))
        const goal=isRefinement ? directorGoal : directCreative || directorIntent(text)
        const requestText=isRefinement ? directorRequest : text
        await previewDirector(cfg,current,requestText,goal,directCinematic || (goal==='dramatic'?'dramatic':'cinematic'))
        result.message='Creative audio preview ready. Compare the mix before applying.'
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
        const settings = separationSettings(result, anchors, editAction, timedEdit, editStart, editEnd)
        const scope=settings.scope, guidance=result.processing_time!=null ? [] : settings.anchors
        setPrompt(result.description); setEditAction(scope.action)
        const point=result.processing_time ?? null
        const range=point!==null ? null : scope.interval || processingRange
        setProcessingTime(point);setProcessingRange(range)
        await previewSound(cfg,current,result.description,scope,result.use_visual?selectedVisual:{},null,false,guidance,clampPasses(passes,choices),range,point)
        result.message='Preview ready. Compare it above the timeline, then apply.'
      } else if (result.kind !== 'cinematic' && result.kind !== 'director') setMessage(result.message)
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
    if (candidateKind === 'separation' || candidateMock === true) setMockStems(candidateMock);setJobId(applied.job_id);setUndoId(candidateId);setCandidateId('');setRetryStems([])
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
  const current: StepId = !ready ? 'upload' : candidateId || undoId ? 'preview' : jobId ? 'export' : 'sound'
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
    <div className="pointer-events-none fixed inset-0 z-[100] flex items-center justify-center p-3 sm:p-6">
    <div role="dialog" aria-label="Preview ready" className="pointer-events-auto rise-in shadow-card max-h-[calc(100dvh-1.5rem)] w-full max-w-[34rem] overflow-y-auto rounded-2xl border border-line bg-surface p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="flex items-center gap-1.5 text-[11px] font-medium text-accent"><FeatureTag name={FEATURES.compare}/>Preview ready</p>
          <p className="mt-0.5 text-sm font-semibold text-fg">{prompt.trim() || 'Separated sound'}</p>
        </div>
        <button type="button" className="rounded-full px-2 py-1 text-xs text-muted hover:bg-tint/[0.06] hover:text-fg" onClick={() => { setCandidateId(''); if (project) { try { sessionStorage.removeItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`) } catch { /* Ignore unavailable storage. */ } } setMessage('Preview discarded.') }}>Discard</button>
      </div>
      <CandidateComparison key={candidateId} original={currentUrl} edited={mixUrl} target={candidateUrl} showTarget={candidateKind==='separation' || candidateKind==='replacement' || candidateKind==='sfx'} seekTo={previewSeek}/>
      {candidateKind==='separation' && processingRange && <div className="mt-2 text-xs text-muted">Processed {processingRange.start.toFixed(2)}–{processingRange.end.toFixed(2)} seconds. <button type="button" className="text-accent underline" onClick={()=>{setCandidateId('');setOpenStep('sound');setMessage('Adjust the processing range, then preview again.')}}>Adjust range</button></div>}
      {(candidateKind==='cinematic' || candidateKind==='director') && <div className="mt-2 text-xs text-muted"><p>{candidateKind==='director'?'Planned changes':'Whole-mix treatment'}: {cinematicSteps.join(' · ')}. No new music or sounds are added.</p>{candidateKind==='director' && directorChanges.map(change=><div key={change.target_stem_id} className="mt-2 flex items-center justify-between gap-2"><span>{change.enabled ? change.action : 'Disabled'}: {change.label} ({change.intervals.map(i=>`${i.start.toFixed(1)}–${i.end.toFixed(1)}s`).join(', ')})</span><button type="button" disabled={busy} className={secondary} onClick={()=>toggleDirectorChange(change)}>{change.enabled?'Disable':'Enable'}</button></div>)}{candidateWarnings.map(w=><p key={w} className="mt-1">{w}</p>)}<div className="mt-2 flex flex-wrap gap-1">{(['subtle','cinematic','dramatic'] as const).map(style=><button key={style} type="button" disabled={busy || style===cinematicStyle} className={secondary} onClick={()=>cinematic(style)}>{style==='cinematic'?'Balanced':style==='dramatic'?'More dramatic':'Subtle'}</button>)}</div></div>}
      {candidateKind==='replacement' && <div className="mt-3 space-y-2 text-xs text-muted">
        <p>{replacementMode==='add'?'Added effects':'Detected sound placements'}: {replacementEvents.filter(e=>e.enabled).length || replacementPeaks.length} enabled.{replacementCount!==null ? ` Requested ${replacementCount} sound${replacementCount===1?'':'s'} at each placement.` : ''}</p>
        {replacementEvents.length>0 && <><div className="flex flex-wrap gap-1" role="group" aria-label="Alignment"><button type="button" className={secondary} disabled={busy || replacementAlignment==='peak'} onClick={()=>adjustReplacement({operation:'alignment',alignment:'peak'})}>Align peaks</button><button type="button" className={secondary} disabled={busy || replacementAlignment==='onset'} onClick={()=>adjustReplacement({operation:'alignment',alignment:'onset'})}>Align attacks</button></div>
          <p>Listen to each match. {replacementMode==='replace'?'Disabling it restores the separated original at that point.':'Disabling it removes the added effect.'}</p>
          <div className="max-h-44 space-y-1 overflow-y-auto">{replacementEvents.map((event,index)=>{const t=(replacementAlignment==='onset'?event.source_onset:event.source_peak)+event.offset;return <div key={event.id} className="flex flex-wrap items-center gap-1 rounded-lg bg-surface-2 p-1"><span className="w-16 font-mono">{t.toFixed(2)}s</span><button type="button" className={secondary} onClick={()=>setPreviewSeek({time:Math.max(0,t-.25),id:Date.now()})}>Listen</button><button type="button" className={secondary} disabled={busy || !event.enabled} aria-label={`Move sound ${index+1} earlier`} onClick={()=>adjustReplacement({operation:'move',event_id:event.id,time:Math.max(0,t-.03)})}>−30ms</button><button type="button" className={secondary} disabled={busy || !event.enabled} aria-label={`Move sound ${index+1} later`} onClick={()=>adjustReplacement({operation:'move',event_id:event.id,time:t+.03})}>+30ms</button><button type="button" className={secondary} disabled={busy} onClick={()=>adjustReplacement({operation:'toggle',event_id:event.id,enabled:!event.enabled})}>{event.enabled?'Disable':'Enable'}</button></div>})}</div>
          <div className="flex items-end gap-1"><label className="flex-1">Add placement at (seconds)<input className={field} type="number" min={0} max={durationSeconds || 0} step="0.01" value={replacementAddTime} onChange={e=>setReplacementAddTime(e.target.value)}/></label><button type="button" className={secondary} disabled={busy || !replacementAddTime.trim()} onClick={()=>adjustReplacement({operation:'add',time:Number(replacementAddTime)})}>Add</button></div>
          <p>After applying, each effect is its own timeline layer, so you can adjust its volume or timing separately.</p></>}
        {candidateWarnings.map(w=><p key={w}>{w}</p>)}{error && <p role="alert" className="text-red-600">{error}</p>}<p>Compare the mix and listen for missing background sounds before applying.</p>
      </div>}
      {candidateKind==='sfx' && <div className="mt-3 space-y-1 text-xs text-muted">
        <p>New sound layers (CC0 TidePool pack). "Just the sound" plays only the inserted effects.</p>
        <ul className="max-h-40 space-y-1 overflow-y-auto">{placementLines(sfxPlacements).map((line,index)=><li key={sfxPlacements[index].id} className="flex items-center justify-between gap-2 rounded-lg bg-surface-2 p-1"><span>{line}</span><button type="button" className={secondary} onClick={()=>setPreviewSeek({time:Math.max(0,sfxPlacements[index].start-.5),id:Date.now()})}>Listen</button></li>)}</ul>
        {candidateWarnings.map(w=><p key={w}>{w}</p>)}
      </div>}
      {candidateKind==='effects' && <p className="mt-2 text-xs text-muted">Effects are rendered into the selected layer on apply. Undo restores the previous audio. Listen for clipping.</p>}
      {candidateKind==='fx' && <div className="mt-2 space-y-1 text-xs text-muted">
        <p><span className="font-medium text-fg">{fxInfo.label}</span>. Applying renders it into the layer; each effect shows as a chip on its track, and Undo restores the previous audio.</p>
        {candidateWarnings.filter(w=>!/Undo restores/.test(w)).map(w=><p key={w}>{w}</p>)}
        {(fxInfo.op==='chain' || fxInfo.op==='adjust') && <div className="flex flex-wrap gap-1 pt-1" role="group" aria-label="Effect amount"><button type="button" className={secondary} disabled={busy} onClick={()=>adjustFx('less')}>Less</button><button type="button" className={secondary} disabled={busy} onClick={()=>adjustFx('more')}>More</button></div>}
      </div>}
      <div className="mt-3 flex items-center gap-2">
        <button className={`${button} flex-1 py-2 text-sm`} disabled={busy} onClick={useRetry}>Apply to timeline</button>
        {candidateKind==='separation' && <button className={secondary} disabled={busy || !capabilities?.quality} title={capabilities?.quality ? 'Deep dive: a slower, higher-quality pass' : (capabilities?.reason || '')} onClick={() => void process(true)}>{FEATURES.quality}</button>}
      </div>
      {candidateKind==='separation' && selectionName && <button className="mt-2 text-xs text-accent hover:underline" disabled={busy} onClick={()=>{setSelection(null);setSelectionName('');onClearSelection();setCandidateId('');setOpenStep('sound');setMessage('Mark where you hear the sound, then preview again using text and time.')}}>Not right? Retry with text and time instead of the object</button>}
    </div>
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
            {project && <div><button type="button" className={`${secondary} w-full`} disabled={busy || !project.metadata} onClick={()=>setReplacementOpen(!replacementOpen)}>{replacementOpen?'Close sound replacement':'Add or replace sounds'}</button>{replacementOpen && <ReplacementPanel key={`${project.project_id}:${replacementSource}:${replacementEffect}`} config={config} projectId={project.project_id} duration={project.metadata?.duration || 0} source={replacementSource} initialEffect={replacementEffect} onSeek={onSeek} onFlush={flush} onPreview={async id=>{setPrompt('Sound replacement');try { sessionStorage.setItem(`soundcanvas-candidate:${baseUrl}:${project.project_id}`,JSON.stringify({id,sound:'Sound replacement'})) } catch { /* Preview still works. */ };await waitCandidate(config,project.project_id,id)}}/>}</div>}
            {project && <button className={`${secondary} w-full`} disabled={busy || !project.metadata} onClick={discoverSounds}>Discover sounds</button>}
            {suggestions.length>0 && <div className="space-y-1 text-xs"><p className="text-muted">Possible sounds in the original recording; times are approximate.</p>{suggestions.map(sound=><button key={sound.label} className={`${secondary} block w-full text-left`} disabled={busy} onClick={()=>{setPrompt(sound.prompt);setProcessingRange(sound.intervals[0] || null);setProcessingTime(null);setMessage('Review or adjust this approximate range, then preview.')}}>{sound.label}: {sound.intervals.slice(0,3).map(i=>`${i.start.toFixed(1)}–${i.end.toFixed(1)}s`).join(', ')}</button>)}</div>}
            {project?.metadata?.has_video && capabilities?.visual === false && <p className="text-muted">The current separator does not support visual object guidance.</p>}
            {project?.metadata?.has_video && <ObjectsPanel key={project.project_id} config={config} projectId={project.project_id} vision={capabilities?.vision} visual={visual} selection={selectedVisual} disabled={busy || capabilities?.visual === false} resumeJobId={resumeVisual?.projectId===project.project_id ? resumeVisual.jobId : null} canPrepare={!jobId} onAnalyzed={visualAnalyzed} onSelect={onSelectVisual} onSeek={onSeek} onPrepared={reloadPreparedMedia}/>}
            <div className="rounded-xl bg-surface-2 p-3 text-xs">
              {prompt.trim() ? <div className="flex items-center gap-2"><span className="text-subtle">Target</span><span className="rounded-full bg-accent-soft px-2 py-0.5 font-medium text-accent">{prompt.trim()}</span></div>
                : <p className="text-muted">{visual ? 'Click an object in the video, or describe' : 'Describe'} the sound in the bar below, like <span className="text-fg">“isolate the dog barking”</span>.</p>}
              {selection !== null && <div className="mt-2"><GuidedByChip selection={selection as {label?:unknown;track_id?:unknown}} start={Number(rangeStart)} end={Number(rangeEnd)} onClear={()=>{setSelection(null);onClearSelection();setSelectionName('')}}/></div>}
            </div>
            {project && <ProcessingRange duration={project.metadata?.duration || 1} value={processingRange} onChange={value=>{setProcessingRange(value);setProcessingTime(null)}}/>}
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
            {project && <button className={`${secondary} w-full py-2`} disabled={busy || !project.metadata} onClick={()=>void run(async()=>{if(project)await previewDirector(config,project,'Make the whole video more cinematic','cinematic','cinematic')})}>Preview cinematic audio</button>}
            {project && <div className="rounded-xl border border-line p-2 text-xs">
              <button type="button" className={`${secondary} w-full`} disabled={busy || !project.metadata} onClick={()=>effectsOpen ? setEffectsOpen(false) : openEffects()}>Adjust layer EQ &amp; compression</button>
              {effectsOpen && <div className="mt-2 space-y-2">
                <label className="block">Layer<select aria-label="Effects layer" className={`${field} mt-1`} value={effectsStem} onChange={e=>setEffectsStem(e.target.value)}>{effectsStems.length ? effectsStems.map(stem=><option key={stem.id} value={stem.id}>{stem.name}</option>) : <option value="">Original audio</option>}</select></label>
                {([{key:'bass',label:'Bass',min:-12,max:12,step:.5,unit:'dB'}, {key:'treble',label:'Treble',min:-12,max:12,step:.5,unit:'dB'}, {key:'highpass',label:'Low-cut',min:0,max:200,step:5,unit:'Hz'}, {key:'compression',label:'Compression',min:1,max:6,step:.1,unit:':1'}] as const).map(control=><label key={control.key} className="block"><span className="flex justify-between"><span>{control.label}</span><span>{effects[control.key]}{control.unit}</span></span><input className="w-full accent-accent" aria-label={control.label} type="range" min={control.min} max={control.max} step={control.step} value={effects[control.key]} onChange={e=>setEffects({...effects,[control.key]:Number(e.target.value)})}/></label>)}
                <p className="text-muted">Preview before applying. Applying renders these effects into the selected layer; later changes add to them. Undo restores the previous audio.</p>
                <div className="flex gap-2"><button type="button" className={button} disabled={busy} onClick={previewEffects}>Preview effects</button><button type="button" className={secondary} disabled={busy} onClick={()=>setEffects({bass:0,treble:0,highpass:0,compression:1})}>Reset controls</button></div>
              </div>}
            </div>}

            {(project?.metadata?.audio_quality?.findings.length ?? 0) > 0 && <details className="text-xs text-amber-700"><summary className="cursor-pointer">Notes about this clip ({project?.metadata?.audio_quality?.findings.length})</summary><ul className="mt-1 list-disc space-y-1 pl-4">{project?.metadata?.audio_quality?.findings.map(f=><li key={f}>{f}</li>)}</ul></details>}
          </div>
        </Step>

        <Step n={3} title="Preview and apply" tag={FEATURES.compare} status={status('preview')} open={open('preview') && (Boolean(candidateId) || Boolean(undoId))} onToggle={() => toggle('preview')}
          summary={status('preview') === 'done' ? 'Applied to the timeline' : undefined}>
          <div className="space-y-2 text-xs">
            {candidateId ? <p className="text-muted">{candidateKind!=='separation' ? 'Compare the original and edited mix above the timeline, then apply.' : 'Compare the original, the edited mix and the isolated sound above the timeline, then apply.'}</p> : null}
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
      {replacementRecovery && <button type="button" className={`${secondary} mt-2`} disabled={busy} onClick={previewRecovery}>Preview adding the effect without removing the original</button>}
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
    {comparison && createPortal(comparison, document.body)}
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
