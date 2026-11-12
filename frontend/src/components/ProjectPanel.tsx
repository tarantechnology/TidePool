import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import type { AgentPlan, BackendConfig, ProjectInfo, SavedStem, VisualMetadata } from '../api/backend'
import { asVisualMetadata } from '../visual'
import type { VisualSelection } from '../visual'
import { separationCommand } from '../separationCommand'
import type { Stem, TrackState } from '../types'
import { planProjectCommand, sendStemPatch, toTrackState, trackToStemPatch, createProject, downloadExport, fetchStemBlobUrl, getExport, getJob, getMessages, getProject, getSavedStems, getVisualJob, getVisualMetadata, localCommand, processProject, startExport, startVisualAnalysis, storeVisualMetadata, updateStemSettings, uploadProject } from '../api/backend'
import { pollJob, toJobStatus } from '../api/jobs'
import { HttpStemSource } from '../api/stemSources'
import type { StemSource } from '../api/stemSources'

const field = 'w-full rounded-lg border border-line bg-white/[0.04] px-2 py-1.5 text-xs text-zinc-100 placeholder:text-zinc-500'
const button = 'rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-black disabled:opacity-40'
const secondary = 'rounded-lg border border-line px-3 py-1.5 text-xs text-zinc-200 disabled:opacity-40'
interface Props {
  baseUrl: string
  onLoadStems: (source: StemSource, label: string) => Promise<void>
  onVideo: (url: string | null) => void
  flush: () => Promise<void>
  onVisual: (metadata: VisualMetadata | null) => void
  selectedVisual: VisualSelection | null
  onClearSelection: () => void
}

export interface ProjectPanelHandle {
  separateVideo: (file: File, description: string, remove: boolean) => Promise<void>
  planCommand: (prompt:string, video:File|null, stems:Stem[], tracks:TrackState[], duration:number) => Promise<AgentPlan>
}

/** Project credentials remain in component memory; the user may copy them for resuming. */
export const ProjectPanel = forwardRef<ProjectPanelHandle, Props>(function ProjectPanel({baseUrl, onLoadStems, onVideo, flush, onVisual, selectedVisual, onClearSelection}, ref) {
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
  const [settingsStems, setSettingsStems] = useState<SavedStem[]>([])
  const [settingId, setSettingId] = useState('')
  const [stemName, setStemName] = useState('')
  const [offset, setOffset] = useState('0')
  const [clearRanges, setClearRanges] = useState(false)
  const [messages, setMessages] = useState<Array<{id:string;text:string}>>([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const abort = useRef<AbortController | null>(null)
  const attachedVideo = useRef<File | null>(null)
  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => { if (selectedVisual) { setSelection(selectedVisual); setSelectionName(selectedVisual.label); setRangeStart(String(selectedVisual.interval.start_seconds)); setRangeEnd(String(selectedVisual.interval.end_seconds)) } }, [selectedVisual])
  const config: BackendConfig = {baseUrl, token}
  const run = async (fn: () => Promise<void>) => {
    setError(''); setMessage(''); setBusy(true)
    try { await fn() } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }
  const loadVideo = async (cfg: BackendConfig, info: ProjectInfo) => {
    if (!info.metadata?.has_video) { onVideo(null); return }
    // Authenticated media is fetched into a blob; the token never enters a URL.
    onVideo(await fetchStemBlobUrl(cfg, `/api/projects/${encodeURIComponent(info.project_id)}/preview`, 3_600_000))
  }
  const track = async (cfg: BackendConfig, id: string, removeDescription?: string) => {
    abort.current?.abort(); const controller = new AbortController(); abort.current = controller
    const result = await pollJob(async signal => toJobStatus(await getJob(cfg, id, signal)), {
      signal: controller.signal, onUpdate: status => setMessage(`Processing: ${status.state}${status.progress !== undefined ? ` ${status.progress}%` : ''}`),
    })
    if (result.state === 'failed') throw new Error(result.error || 'Processing failed')
    const job = await getJob(cfg, id)
    if (removeDescription) {
      if (!job.project_id) throw new Error('Processing result has no project ID.')
      const saved = await getSavedStems(cfg, job.project_id)
      const matches = saved.filter(s => s.name.toLowerCase() === removeDescription.toLowerCase())
      if (matches.length !== 1 || !saved.some(s => s.name.toLowerCase() === 'residual')) throw new Error('Separation did not return both the requested sound and residual audio. No stem was muted.')
      const stem = matches[0]
      await sendStemPatch({config:cfg,projectId:job.project_id}, stem.id, {...trackToStemPatch(toTrackState(stem)),muted:true}, stem.revision)
    }
    await onLoadStems(new HttpStemSource(cfg, id), 'Project stems')
    setJobId(id)
    const model = job.inference?.model
    const summary = model?.is_mock ? 'Ready — test stems; no AI separation was performed.' : model?.is_mock === false ? 'Stems ready to edit.' : 'Stems ready; model identity unavailable.'
    setMessage([summary, ...(job.inference?.warnings || [])].join(' '))
  }
  const createAndUpload = () => run(async () => {
    if (!file) throw new Error('Choose a media file first.')
    await flush()
    const created = await createProject(baseUrl)
    if (!created.access_token) throw new Error('Project token was not returned.')
    const cfg = {baseUrl, token:created.access_token}
    setProject(created); setToken(created.access_token); setResumeId(created.project_id); setResumeToken(created.access_token)
    setJobId(''); setSettingsStems([]); setMessages([]); await onLoadStems({load: async () => []}, 'No stems yet'); onVisual(null); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('')
    setMessage('Uploading and extracting audio…')
    await uploadProject(cfg, created.project_id, file)
    attachedVideo.current = file
    const info = await getProject(cfg, created.project_id); setProject(info)
    setFormat(info.metadata?.has_video ? 'mp4' : 'wav')
    await loadVideo(cfg, info)
    setMessage('Upload complete. Add a sound prompt and process.')
  })
  const resume = () => run(async () => {
    if (!resumeId.trim() || !resumeToken.trim()) throw new Error('Enter both project ID and token.')
    await flush()
    const cfg = {baseUrl,token:resumeToken.trim()}; const info = await getProject(cfg,resumeId.trim())
    attachedVideo.current = null
    setToken(cfg.token); setProject(info); setJobId(info.last_completed_job_id || ''); setSettingsStems([]); setMessages([])
    setFormat(info.metadata?.has_video ? 'mp4' : 'wav'); setSelection(null); onClearSelection(); setVisualName(''); setSelectionName('')
    try { onVisual(info.visual_metadata ? asVisualMetadata(info.visual_metadata) : null) } catch { onVisual(null) }
    await loadVideo(cfg, info)
    const visualJob = info.latest_visual_job
    if (visualJob && (visualJob.status === 'queued' || visualJob.status === 'processing')) await finishAnalysis(cfg, info.project_id, visualJob.id)
    const latest = info.latest_job
    if (latest && (latest.status === 'queued' || latest.status === 'processing')) await track(cfg, latest.id)
    else if (info.last_completed_job_id) { await onLoadStems(new HttpStemSource(cfg,info.last_completed_job_id), 'Project stems'); setMessage('Project restored.') }
    else { await onLoadStems({load: async () => []}, 'No stems yet'); setMessage('Project restored. Upload or start processing.') }
  })
  const process = () => run(async () => {
    if (!project) return
    const instruction = separationCommand(prompt)
    const prompts = instruction ? [instruction.description] : prompt.split(',').map(p => p.trim()).filter(Boolean)
    if (!prompts.length) throw new Error('Enter at least one sound prompt.')
    await flush()
    let visualPrompt: unknown = {}
    if (selection) {
      const start=Number(rangeStart), end=Number(rangeEnd)
      if (!rangeStart || !rangeEnd || !Number.isFinite(start) || !Number.isFinite(end) || end<=start) throw new Error('Enter a valid visual selection interval.')
      visualPrompt = {...selection as object, interval:{start_seconds:start,end_seconds:end}}
    }
    const id = await processProject(config,project.project_id,prompts,visualPrompt)
    await track(config,id,instruction?.remove ? instruction.description : undefined)
    setProject(await getProject(config,project.project_id))
  })
  useImperativeHandle(ref, () => ({ separateVideo: async (video, description, remove) => {
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
      const id=await processProject(cfg,created.project_id,[description])
      await track(cfg,id,remove ? description : undefined)
      setProject(await getProject(cfg,created.project_id))
      if (remove) setMessage(`Separated and muted ${description}. Export to download the result.`)
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
      const result=await planProjectCommand(cfg,current.project_id,{prompt:text, stems:hasStems ? stems.map(({id,label,kind,duration:d})=>({id,label,kind,duration:d})) : [],tracks:hasStems ? tracks : [],duration:hasStems ? duration : (current.metadata?.duration || duration),has_project_stems:hasStems})
      if (result.kind === 'separate') {
        if (hasStems || !result.description?.trim()) throw new Error('This separation cannot replace existing layers. Create a new project first.')
        setMessage('Separating audio on the GPU…')
        const id=await processProject(cfg,current.project_id,[result.description])
        await track(cfg,id,result.remove ? result.description : undefined)
        setProject(await getProject(cfg,current.project_id))
        setMessage(result.remove ? `Separated and muted ${result.description}. Export to download.` : `Separated ${result.description}.`)
      } else setMessage(result.message)
      return result
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); throw e }
    finally { setBusy(false) }
  }}))
  const finishAnalysis = async (cfg: BackendConfig, projectId: string, id: string) => {
    for (;;) {
      const job=await getVisualJob(cfg,id)
      if (job.status === 'failed') throw new Error(job.error || 'Visual analysis failed')
      if (job.status === 'completed') break
      setMessage(`Analyzing video: ${job.status}`); await new Promise(resolve => setTimeout(resolve,1000))
    }
    const metadata=asVisualMetadata(await getVisualMetadata(cfg,projectId))
    onVisual(metadata); onClearSelection(); setSelection(null); setSelectionName(''); setVisualName('Analysis complete')
    const count=metadata.frames.reduce((n, f) => n+f.objects.filter(o => o.track_id !== null && o.mask).length,0)
    setMessage(count ? `Analysis ready for the first ${metadata.processed_duration_seconds.toFixed(1)} seconds. Pause and click an object in the video.` : 'Analysis completed, but no tracked objects with masks were found.')
  }
  const analyze = () => run(async () => {
    if (!project) return
    const id=await startVisualAnalysis(config,project.project_id)
    await finishAnalysis(config,project.project_id,id)
  })
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
      <button className={button} disabled={busy || !file} onClick={createAndUpload}>Create & upload</button>
      {project && <details className="text-xs"><summary className="cursor-pointer">Project credentials</summary><p className="mt-1 break-all">ID: {project.project_id}</p><label className="block mt-1">Token<input className={field} readOnly value={token} type="text" onFocus={e => e.currentTarget.select()} /></label><p className="text-zinc-500">Copy these to resume this project later.</p></details>}
      <details className="text-xs"><summary className="cursor-pointer">Resume a project</summary><div className="mt-2 space-y-1"><input className={field} value={resumeId} onChange={e=>setResumeId(e.target.value)} placeholder="Project ID" aria-label="Project ID"/><input className={field} type="password" value={resumeToken} onChange={e=>setResumeToken(e.target.value)} placeholder="Project token" aria-label="Project token"/><button className={secondary} disabled={busy} onClick={resume}>Resume</button></div></details>
      {project && <>
        <label className="block text-xs">Sound prompts <span className="text-zinc-500">(comma separated)</span><input className={`${field} mt-1`} value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="dog barking, music" /></label>
        <details className="text-xs"><summary className="cursor-pointer">Visual selection (optional)</summary>{project.metadata?.has_video && <button className={`${secondary} mt-2`} disabled={busy} onClick={analyze}>Analyze video</button>}<p className="my-1 text-zinc-400">Analyze the video and click a tracked object, or import matching detection and selection JSON files.</p><label className="block">Detections<input className={field} type="file" accept=".json" disabled={busy} onChange={e=>{const f=e.target.files?.[0]; if(f) void run(async()=>{const data=asVisualMetadata(await readJson(f));await storeVisualMetadata(config,project.project_id,data);onVisual(data);setVisualName(f.name);setMessage('Detections saved.')})}} /></label>{visualName && <span>{visualName}</span>}<label className="block mt-1">Selection<input className={field} type="file" accept=".json" disabled={busy} onChange={e=>{const f=e.target.files?.[0]; if(f) void run(async()=>{const data=await readJson(f) as VisualSelection; if (!data?.interval) throw new Error('Selection JSON needs an interval.'); setSelection(data);setRangeStart(String(data.interval.start_seconds));setRangeEnd(String(data.interval.end_seconds));setSelectionName(f.name);setMessage('Selection ready.')})}} /></label>{selectionName && <><button className="text-accent" onClick={()=>{setSelection(null);onClearSelection();setSelectionName('')}}>Clear {selectionName}</button><div className="flex gap-1"><label>Start (seconds)<input className={field} type="number" min="0" step="any" value={rangeStart} onChange={e=>setRangeStart(e.target.value)} /></label><label>End (seconds)<input className={field} type="number" min="0" step="any" value={rangeEnd} onChange={e=>setRangeEnd(e.target.value)} /></label></div></>}</details>
        <button className={button} disabled={busy || !project.metadata || !prompt.trim()} onClick={process}>Separate audio</button>
        <div className="flex gap-2"><select className={field} aria-label="Export format" value={format} onChange={e=>setFormat(e.target.value as typeof format)}><option value="mp4" disabled={!project.metadata?.has_video}>MP4 video</option><option value="wav">WAV audio</option><option value="mp3">MP3 audio</option></select><button className={secondary} disabled={busy || !jobId} onClick={exportFile}>Export</button></div>
        {jobId && <details className="text-xs"><summary className="cursor-pointer">Stem settings</summary><button className={`${secondary} mt-1`} disabled={busy} onClick={refreshSettings}>Load stem settings</button>{settingsStems.length>0 && <div className="mt-1 space-y-1"><select className={field} aria-label="Stem" value={settingId} onChange={e=>{const stem=settingsStems.find(s=>s.id===e.target.value);if(stem){setSettingId(stem.id);setStemName(stem.name);setOffset(String(stem.start_offset || 0));setClearRanges(false)}}}>{settingsStems.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select><label>Name<input className={field} value={stemName} onChange={e=>setStemName(e.target.value)} /></label><label>Start offset (seconds)<input className={field} type="number" min="0" step="any" value={offset} onChange={e=>setOffset(e.target.value)} /></label><label className="flex items-center gap-1"><input type="checkbox" checked={clearRanges} onChange={e=>setClearRanges(e.target.checked)} />Clear remote mute ranges</label><button className={secondary} disabled={busy} onClick={saveSettings}>Save stem settings</button></div>}</details>}
        <details className="text-xs"><summary className="cursor-pointer">Remote command test</summary><p className="my-1 text-zinc-500">Local webhook; messaging delivery is not connected.</p><div className="flex gap-1"><input className={field} value={command} onChange={e=>setCommand(e.target.value)} placeholder="Mute the dog"/><button className={secondary} disabled={busy || !command.trim()} onClick={sendCommand}>Send</button></div><button className={`${secondary} mt-1`} disabled={busy} onClick={()=>void run(async()=>{setMessages(await getMessages(config,project.project_id));setMessage('Messages updated.')})}>Refresh messages</button>{messages.map(m=><p key={m.id} className="mt-1 text-zinc-400">{m.text}</p>)}</details>
      </>}
      {busy && <p role="status" className="text-xs text-accent">{message || 'Working…'}</p>}
      {!busy && message && <p role="status" className="text-xs text-accent">{message}</p>}
      {error && <p role="alert" className="break-words text-xs text-red-400">{error}</p>}
    </div>
  </section>
})
