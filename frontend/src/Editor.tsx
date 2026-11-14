import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, ReactNode } from 'react'
import { NO_PROJECT_HINT } from './agent/planOutcome'
import { NO_PROJECT_STEMS, showsMockBadge } from './state/projectStems'
import type { ProjectStemsState } from './state/projectStems'
import { validateActions } from './agent/validate'
import type { SyncTarget, VisualMetadata } from './api/backend'
import type { VisualSelection } from './visual'
import { loadFromSource, stemFailureMessage } from './api/loadStems'
import { FileStemSource, isAudioFile, isVideoFile, MockStemSource } from './api/stemSources'
import type { StemSource } from './api/stemSources'
import { AudioEngine } from './audio/engine'
import { resolveActions } from './audio/resolve'
import { isDurationMismatch } from './audio/videoSync'
import { EditLog } from './components/EditLog'
import { ProjectPanel } from './components/ProjectPanel'
import { Mixer } from './components/Mixer'
import { MoonIcon, MoonPullPanel, MotifStrip } from './components/MoonPull'
import type { WithLayers } from './components/MoonPull'
import { FEATURES } from './features'
import { loadRules, saveRules } from './moonpull/rules'
import type { Rule } from './moonpull/rules'
import type { ProjectPanelHandle } from './components/ProjectPanel'
import { Logo } from './components/Logo'
import { Splitter } from './components/Splitter'
import { DropLoader } from './components/DropLoader'
import { PromptBar } from './components/PromptBar'
import { Timeline } from './components/Timeline'
import { Transport } from './components/Transport'
import { VideoPanel } from './components/VideoPanel'
import { applyAction, initialTracks } from './edits'
import { downloadText } from './export/download'
import { serializeEditList } from './export/editList'
import { displayAspect, readRotation, writeRotation } from './rotation'
import type { Rotation } from './rotation'
import { useStemSync } from './hooks/useStemSync'
import { canRedo, canUndo, commit, createHistory, redo, undo } from './state/history'
import { dropRedo, nextRedo, nextUndo, pushStep, stepRedone, stepUndone } from './state/steps'
import type { ServerStep, StepStack } from './state/steps'
import { ProjectSwitcher } from './components/ProjectSwitcher'
import { captureThumb, getProject as getSavedProject, loadSteps, saveProject, saveSteps, takePendingUpload } from './projects/registry'
import type { History } from './state/history'
import { createEditorState, mergeKinds, setKindOverride } from './state/kinds'
import type { EditorState } from './state/kinds'
import type { EditAction, Keyframe, Stem, StemKind, TrackState } from './types'

type Status = 'loading' | 'ready' | 'error'

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || '/api'

const MOCK_LABEL = 'Mock stems'
const VIDEO_AUDIO_LABEL = 'Original audio'


/** Range sliders are inputs too, but undo should still work while one has focus. */
function isTextEntry(el: HTMLElement): boolean {
  if (el.tagName === 'TEXTAREA') return true
  return el.tagName === 'INPUT' && (el as HTMLInputElement).type !== 'range'
}

export default function Editor({ projectId: openProjectId = null }: { projectId?: string | null } = {}) {
  const [engine, setEngine] = useState<AudioEngine | null>(null)
  const [baseStems, setBaseStems] = useState<Stem[]>([])
  const [history, setHistory] = useState<History<EditorState>>(() => createHistory(createEditorState([])))
  const [status, setStatus] = useState<Status>('loading')
  const [error, setError] = useState('')
  const [playing, setPlaying] = useState(false)
  const [log, setLog] = useState<EditAction[]>([])
  const [stemsLabel, setStemsLabel] = useState(MOCK_LABEL)
  const [loadError, setLoadError] = useState('')
  const [syncTarget, setSyncTarget] = useState<SyncTarget | null>(null)
  const [syncIssue, setSyncIssue] = useState('')
  const [dragging, setDragging] = useState(false)
  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  const [originalSrc, setOriginalSrc] = useState<string | null>(null)
  const [compareOriginal, setCompareOriginal] = useState(false)
  const [visual, setVisual] = useState<VisualMetadata | null>(null)
  const [visualSelection, setVisualSelection] = useState<VisualSelection | null>(null)
  const [seekTo, setSeekTo] = useState<{ time: number; n: number } | null>(null)
  const seekVideo = useCallback((time: number) => setSeekTo((prev) => ({ time, n: (prev?.n ?? 0) + 1 })), [])
  const [videoDuration, setVideoDuration] = useState<number | null>(null)
  const [projectStems,setProjectStems]=useState<ProjectStemsState>(NO_PROJECT_STEMS)
  // Undo and redo over changes made on the server, once the editor's own history runs out. Kept per project.
  const [steps, setStepsState] = useState<{ projectId: string | null; stack: StepStack }>(() => ({ projectId: openProjectId, stack: loadSteps(openProjectId) }))
  const stepScope = projectStems.projectId ?? openProjectId
  if (steps.projectId !== stepScope) setStepsState({ projectId: stepScope, stack: loadSteps(stepScope) })
  const setSteps = useCallback((update: (s: StepStack) => StepStack) => setStepsState((prev) => {
    const stack = update(prev.stack)
    if (stack !== prev.stack) saveSteps(prev.projectId, stack)
    return stack === prev.stack ? prev : { ...prev, stack }
  }), [])
  // A turned clip is laid out by the shape it is shown at, so a sideways landscape clip stands as portrait.
  const [rotation, setRotation] = useState<Rotation>(0)
  const rotationProject = useRef<string | null>(null)
  useEffect(() => {
    // A project brings back the turn saved for it; a fresh upload keeps the turn chosen while it uploaded.
    const pid = projectStems.projectId ?? null
    if (pid === rotationProject.current) return
    rotationProject.current = pid
    if (!pid) return
    const saved = readRotation(pid)
    if (saved !== null) setRotation(saved)
    else writeRotation(pid, rotation)
  }, [projectStems.projectId, rotation])
  const rotate = useCallback((r: Rotation) => { setRotation(r); writeRotation(rotationProject.current, r) }, [])
  const [laneView, setLaneView] = useState<'chart' | 'mixer'>('chart')
  const [hasLocalVideo,setHasLocalVideo]=useState(false)
  const [working, setWorking] = useState<string | null>(null)
  const videoUploadInProgress = useRef(false)
  const [clipAspect, setClipAspect] = useState<number | null>(null)
  // A turned clip is laid out by the shape it is shown at.
  const videoAspect = clipAspect === null ? null : displayAspect(clipAspect, rotation)
  const portrait = videoSrc !== null && videoAspect !== null && videoAspect < 0.9
  const [layout, setLayout] = useState(readLayout)
  const mainRef = useRef<HTMLElement>(null)
  useEffect(() => { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)) } catch { /* layout just won't persist */ } }, [layout])
  const setTop = (px: number) => { setAutoTop(false); const h = stage.current?.clientHeight ?? 800; setLayout((l) => ({ ...l, top: clamp(px / h, 0.18, 0.78) })) }
  // A widescreen clip gets a video box exactly its shape at the full width, with the timeline
  // taking what's left (at least TIMELINE_MIN). Dragging the handle switches this off until reset.
  const [autoTop, setAutoTop] = useState(true)
  const stage = useRef<HTMLDivElement>(null)
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const measure = () => setStageSize({ width: el.clientWidth, height: el.clientHeight })
    measure()
    const observer = new ResizeObserver(measure); observer.observe(el)
    return () => observer.disconnect()
  }, [])
  const fittedTop = videoAspect && stageSize.width > 0 ? Math.max(140, Math.min(stageSize.width / videoAspect, stageSize.height - TIMELINE_MIN)) : null
  const setPanel = (px: number) => setLayout((l) => ({ ...l, panel: clamp(px, 260, 560) }))
  const stemsRef = useRef<Stem[]>([])
  const picker = useRef<HTMLInputElement>(null)
  const projectPanel = useRef<ProjectPanelHandle>(null)
  const originalVideo = useRef<File | null>(null)
  const loadRequest = useRef(0)
  const { tracks, kinds } = history.present
  const stems = useMemo(() => mergeKinds(baseStems, kinds), [baseStems, kinds])
  const { syncError, flush, exclusive } = useStemSync(syncTarget, tracks, setLoadError)
  const chipErrors = [syncError, loadError, syncIssue].filter(Boolean)
  const tracksRef = useRef<TrackState[]>([])
  // Moon pull rules belong to a project (or to the layers in this tab when there is none).
  const ruleScope = projectStems.projectId ?? 'local'
  const [rules, setRulesState] = useState<{ scope: string; list: Rule[] }>(() => ({ scope: ruleScope, list: loadRules(ruleScope) }))
  if (rules.scope !== ruleScope) setRulesState({ scope: ruleScope, list: loadRules(ruleScope) })
  const setRules = useCallback((update: (rules: Rule[]) => Rule[]) => setRulesState((prev) => {
    const list = update(prev.list); saveRules(prev.scope, list); return { ...prev, list }
  }), [])
  const [moonOpen, setMoonOpen] = useState(false)
  const [moonRule, setMoonRule] = useState<string | null>(null)
  const openRule = useCallback((id: string) => { setMoonRule(id); setMoonOpen(true) }, [])
  const getTracks = useCallback(() => tracksRef.current, [])

  useEffect(() => {
    tracksRef.current = tracks
  }, [tracks])

  useEffect(() => {
    stemsRef.current = stems
  }, [stems])

  useEffect(() => {
    const eng = new AudioEngine()
    const mock = new MockStemSource()
    let cancelled = false
    eng.subscribe(() => setPlaying(eng.isPlaying()))
    const loading = mock
      .load()
      .then((loadedStems) => eng.replaceStems(loadedStems))
      .then((loaded) => {
        if (cancelled || !loaded) return
        setBaseStems(loaded.stems)
        setHistory(createHistory(createEditorState(initialTracks(loaded.stems))))
        setEngine(eng)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : String(err))
        setStatus('error')
      })
    return () => {
      cancelled = true
      eng.dispose()
      // Revoking the mock's object URLs while the engine is still fetching them fails those
      // fetches (React's dev double-mount does exactly that), so wait for the load to settle.
      void loading.finally(() => mock.dispose())
    }
  }, [])

  useEffect(() => {
    engine?.setTracks(tracks)
  }, [engine, tracks])

  useEffect(() => {
    if (!videoSrc) return
    return () => URL.revokeObjectURL(videoSrc)
  }, [videoSrc])

  useEffect(() => { if (!originalSrc) return; return () => URL.revokeObjectURL(originalSrc) }, [originalSrc])
  useEffect(() => {
    if (!engine || !originalSrc || !compareOriginal) return
    const audio=new Audio(originalSrc); audio.volume=0.7
    engine.setOutputMuted(true)
    const sync=() => { const t=engine.getTime(); if (Math.abs(audio.currentTime-t)>0.25) audio.currentTime=t }
    const play=() => { sync(); void audio.play().catch(e=>setLoadError(`Original playback failed: ${String(e)}`)) }
    const off=engine.subscribe(e=>{ if(e.type==='play') play(); else {audio.pause();sync();if(e.type==='seek' && engine.isPlaying()) play()} })
    if (engine.isPlaying()) play()
    const timer=window.setInterval(()=>{if(engine.isPlaying()) sync()},500)
    return () => {window.clearInterval(timer);off();audio.pause();engine.setOutputMuted(false)}
  }, [engine,originalSrc,compareOriginal])

  /** Replace every stem. Undo history and the edit log describe the old ones, so they go too. */
  const loadStems = useCallback(
    async (source: StemSource, label: string) => {
      if (!engine) return
      // Only the newest request may touch status, errors or the engine.
      const id = ++loadRequest.current
      const isLatest = () => id === loadRequest.current
      engine.invalidateLoads()
      setLoadError('')
      setStatus('loading')
      try {
        const loaded = await loadFromSource(source, engine, isLatest, flush)
        if (!loaded) return
        setBaseStems(loaded.stems)
        setHistory(createHistory(createEditorState(loaded.tracks)))
        setLog([])
        setStemsLabel(label)
        setSyncTarget(source.syncTarget ?? null)
        setSyncIssue(source.syncIssue ?? '')
        setLoadError(stemFailureMessage(loaded.failures))
      } catch (err) {
        // The engine keeps the previous stems when a load fails.
        if (isLatest()) setLoadError(err instanceof Error ? err.message : String(err))
        throw err
      } finally {
        source.dispose?.()
        if (isLatest()) setStatus('ready')
      }
    },
    [engine, flush],
  )

  const loadFiles = useCallback(
    (files: File[]) => {
      const audio = files.filter(isAudioFile)
      if (audio.length === 0) {
        setLoadError('No audio files found.')
        return
      }
      void loadStems(new FileStemSource(audio), `${audio.length} audio file${audio.length > 1 ? 's' : ''}`).catch(() => {})
    },
    [loadStems],
  )

  /**
   * A video brings its own audio when nothing better is loaded: mock stems
   * always exist and belong to no video, and audio taken from an earlier video
   * belongs to that video. Best effort, since decoding depends on codec and
   * browser. A failure keeps the current stems and reports in the status chip.
   */

  const loadVideo = useCallback(
    (file: File, withAudio: boolean) => {
      originalVideo.current = file
      setHasLocalVideo(true)
      setVisual(null); setVisualSelection(null)
      setVideoSrc(URL.createObjectURL(file))
      setVideoDuration(null)
      setLoadError('')
      if (withAudio) void loadStems(new FileStemSource([file], VIDEO_AUDIO_LABEL), VIDEO_AUDIO_LABEL).catch(() => {})
    },
    [loadStems],
  )

  const onPickVideo = useCallback((file: File) => {
    const panel = projectPanel.current
    if (working || videoUploadInProgress.current || !panel) return
    videoUploadInProgress.current = true
    setCompareOriginal(false); setRotation(0)
    // A song has no picture: the panel shows the audio placeholder once the project knows.
    if (isAudioFile(file) && !isVideoFile(file)) { originalVideo.current = file; setHasLocalVideo(true); setVisual(null); setVisualSelection(null); setVideoSrc(null); setClipAspect(null); setVideoDuration(null) }
    else loadVideo(file, false)
    void panel.uploadMedia(file).finally(() => { videoUploadInProgress.current = false })
  }, [loadVideo, working])

  // A clip chosen on the projects page starts uploading as soon as the editor is ready.
  const pendingTaken = useRef(false)
  useEffect(() => {
    if (openProjectId || pendingTaken.current || !engine || status !== 'ready' || !projectPanel.current) return
    pendingTaken.current = true
    const file = takePendingUpload()
    if (!file) return
    if (isVideoFile(file)) onPickVideo(file)
    else void projectPanel.current.uploadMedia(file)
  }, [openProjectId, engine, status, onPickVideo])

  // Keep the project's card in the project list current: its layer count and a frame of the clip.
  const listedProject = projectStems.projectId
  useEffect(() => {
    if (!listedProject || status !== 'ready') return
    if (stemsLabel === 'Project stems') saveProject({ id: listedProject, layers: baseStems.length })
    else if (stemsLabel === VIDEO_AUDIO_LABEL) saveProject({ id: listedProject, layers: 0 })
  }, [listedProject, stemsLabel, baseStems.length, status])
  useEffect(() => {
    if (!listedProject || !videoSrc || getSavedProject(listedProject)?.thumb) return
    let live = true
    void captureThumb(videoSrc).then((thumb) => { if (live && thumb) saveProject({ id: listedProject, thumb }) })
    return () => { live = false }
  }, [listedProject, videoSrc])

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    loadFiles(Array.from(e.target.files ?? []))
    e.target.value = ''
  }

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    if (status !== 'ready') return
    const files = Array.from(e.dataTransfer.files)
    const video = files.find(isVideoFile)
    const hasAudio = files.some(isAudioFile)
    if (video) onPickVideo(video)
    else if (hasAudio || !video) loadFiles(files)
  }

  const dispatch = useCallback((action: EditAction, coalesceKey?: string) => {
    const now = Date.now()
    setHistory((h) => commit(h, { ...h.present, tracks: applyAction(h.present.tracks, action) }, { coalesceKey, now }))
    setSteps(dropRedo)
  }, [setSteps])

  const setKind = useCallback((stemId: string, kind: StemKind) => {
    setHistory((h) => commit(h, { ...h.present, kinds: setKindOverride(h.present.kinds, stemId, kind) }))
    setSteps(dropRedo)
  }, [setSteps])

  // Renaming shows the new name at once; the save runs behind it, and a failed one puts the old name back.
  const renameLayer = useCallback(async (stemId: string, name: string) => {
    if (!projectPanel.current) throw new Error('Project controls are not ready.')
    const next = name.trim()
    const previous = stemsRef.current.find((s) => s.id === stemId)?.label
    if (!next || previous === undefined || next === previous) return
    const relabel = (from: string, to: string) => setBaseStems((list) => list.map((s) => (s.id === stemId && s.label === from ? { ...s, label: to } : s)))
    relabel(previous, next)
    try { await projectPanel.current.renameLayer(stemId, next, exclusive) }
    catch (err) { relabel(next, previous); throw err }
  }, [exclusive])

  const deleteLayer = useCallback(async (stemId: string, fold: boolean) => {
    if (!projectPanel.current) throw new Error('Project controls are not ready.')
    await projectPanel.current.deleteLayer(stemId, fold)
  }, [])
  const residual = baseStems.find((s) => s.kind === 'residual') ?? null

  const runPrompt = useCallback(
    async (text: string) => {
      if (!engine) return { message: 'Engine not ready.', ok: false }
      if (!projectPanel.current) return {message:'Project controls are not ready.',ok:false}
      const req = { prompt: text, stems, tracks: tracksRef.current, duration: engine.duration }
      const plan = await projectPanel.current.planCommand(text, originalVideo.current, stems, req.tracks, engine.duration)
      if (plan.kind === 'separate' || plan.kind === 'cinematic' || plan.kind === 'director' || plan.kind === 'replacement') return {message:plan.message,ok:true}
      if (plan.kind === 'clarify') return {message:plan.message,ok:false}
      const checked = validateActions(plan.actions, stems, engine.duration)
      if (!checked.ok) return {message:`AI returned an invalid edit: ${checked.errors[0]}`,ok:false}
      const res = {actions:checked.actions,message:plan.message}
      // The result was built from the request's snapshot and can replace a whole
      // curve, so applying it over newer edits would silently lose them.
      if (stemsRef.current !== stems || tracksRef.current !== req.tracks) {
        return { message: 'Edits changed while the agent was running, so its result was not applied.', ok: false }
      }
      const { actions, warnings } = resolveActions(res.actions, engine, engine.duration)
      if (actions.length > 0) {
        setHistory((h) => commit(h, { ...h.present, tracks: actions.reduce(applyAction, h.present.tracks) }))
        setSteps(dropRedo)
        setLog((l) => [...l, ...actions])
      }
      return { message: [res.message, ...warnings].join(' '), ok: actions.length > 0 }
    },
    [engine, stems, setSteps],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || !engine) return
      const el = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'BUTTON', 'SELECT'].includes(el.tagName)) return
      e.preventDefault()
      if (engine.isPlaying()) engine.pause()
      else void engine.play()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [engine])

  // One undo for everything: the editor's own edits first, then changes made on the server
  // (applying a preview, Explode, a Current command, Moon pull), newest first. Redo walks back.
  const historyRef = useRef(history)
  const stepsRef = useRef(steps.stack)
  useEffect(() => { historyRef.current = history; stepsRef.current = steps.stack }, [history, steps.stack])
  const stepping = useRef(false)
  const restoreStep = useCallback((how: 'undo' | 'redo') => {
    const step = how === 'undo' ? nextUndo(stepsRef.current) : nextRedo(stepsRef.current)
    const panel = projectPanel.current
    if (!step || !panel || stepping.current) return
    stepping.current = true
    void panel.restoreTo(how === 'undo' ? step.before : step.after, step.label, how)
      .then(() => setSteps(how === 'undo' ? stepUndone : stepRedone))
      .catch(() => { /* the panel shows why */ })
      .finally(() => { stepping.current = false })
  }, [setSteps])
  const undoAny = useCallback(() => {
    if (canUndo(historyRef.current)) setHistory((h) => undo(h))
    else restoreStep('undo')
  }, [restoreStep])
  const redoAny = useCallback(() => {
    if (canRedo(historyRef.current)) setHistory((h) => redo(h))
    else restoreStep('redo')
  }, [restoreStep])
  const onStep = useCallback((step: ServerStep) => setSteps((s) => pushStep(s, step)), [setSteps])
  const undoStep: ServerStep | null = canUndo(history) ? null : nextUndo(steps.stack)
  const redoStep: ServerStep | null = canRedo(history) ? null : nextRedo(steps.stack)
  const canUndoAny = canUndo(history) || undoStep !== null
  const canRedoAny = canRedo(history) || redoStep !== null

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'z') return
      if (isTextEntry(e.target as HTMLElement)) return
      e.preventDefault()
      if (e.shiftKey) redoAny()
      else undoAny()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undoAny, redoAny])

  const duration = engine?.duration ?? 0
  const hasLayers = stems.length > 0
  const viewToggle = (
    <div className="flex rounded-full border border-line bg-surface p-0.5 text-[10px] shadow-sm" role="group" aria-label="Layer view">
      {(['chart', 'mixer'] as const).map((v) => (
        <button key={v} aria-pressed={laneView === v} onClick={() => setLaneView(v)}
          className={`rounded-full px-2 py-px font-medium whitespace-nowrap transition-colors ${laneView === v ? 'bg-accent text-white' : 'text-muted hover:text-fg'}`}>
          {v === 'chart' ? 'Tide chart' : 'Mixer'}
        </button>
      ))}
    </div>
  )
  const canRename = Boolean(projectStems.projectId) && stemsLabel === 'Project stems'
  const withLayers: WithLayers | null = canRename
    ? (label, work, reload) => { if (!projectPanel.current) throw new Error('Project controls are not ready.'); return projectPanel.current.withLayers(label, work, reload) }
    : null
  const setRuleAutomation = (stemId: string, keyframes: Keyframe[]) => dispatch({ type: 'set_automation', stemId, keyframes })
  const statusText = status === 'loading' ? 'Loading layers' : status === 'error' ? 'Load failed' : chipErrors[0] || (showsMockBadge(projectStems, stemsLabel) || stemsLabel === MOCK_LABEL ? 'Demo layers' : stemsLabel)

  return (
    <div
      className="relative flex h-screen flex-col overflow-hidden bg-canvas"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(e) => {
        if (e.relatedTarget === null) setDragging(false)
      }}
      onDrop={onDrop}
    >
      {dragging && (
        <div className="pointer-events-none absolute inset-3 z-50 flex flex-col items-center justify-center gap-1 rounded-2xl border-2 border-dashed border-accent/50 bg-accent-soft/80 text-sm font-medium text-accent backdrop-blur-sm">
          Drop a video to start a project
          <span className="text-xs font-normal text-muted">or audio files to replace the layers (to split a song, choose it with Upload)</span>
        </div>
      )}
      <div className="pointer-events-none absolute -top-40 left-1/2 h-80 w-[60rem] -translate-x-1/2 rounded-full bg-sky/30 blur-[120px]" />

      <header className="relative z-20 flex h-14 shrink-0 items-center gap-4 border-b border-line bg-surface/70 px-4 backdrop-blur">
        <a href="#/" className="shrink-0 text-fg transition-opacity hover:opacity-80">
          <Logo />
        </a>
        <ProjectSwitcher projectId={projectStems.projectId ?? openProjectId} fallback={openProjectId ? 'Opening project…' : 'New project'} disabled={Boolean(working)} beforeLeave={flush} />
        <div className="flex min-w-0 flex-1 items-center justify-center gap-3">
          <Transport engine={engine} playing={playing} duration={duration} />
          {originalSrc && stemsLabel === 'Project stems' && (
            <div className="flex rounded-full border border-line bg-surface p-0.5 text-[11px]" role="group" aria-label="Listen to">
              {([false, true] as const).map((orig) => (
                <button key={String(orig)} aria-pressed={compareOriginal === orig} onClick={() => setCompareOriginal(orig)}
                  className={`rounded-full px-2.5 py-1 transition-colors ${compareOriginal === orig ? 'bg-accent text-white' : 'text-muted hover:text-fg'}`}>
                  {orig ? 'Original' : 'Edited'}
                </button>
              ))}
            </div>
          )}
          {isDurationMismatch(videoDuration, duration) && (
            <span className="hidden font-mono text-[10px] text-amber-700 xl:inline" title="The video and audio lengths differ">
              Video {videoDuration?.toFixed(1)}s · audio {duration.toFixed(1)}s
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <input ref={picker} type="file" accept="audio/*" multiple hidden onChange={onPick} />
          <button type="button" aria-pressed={moonOpen} disabled={status !== 'ready' || !hasLayers} onClick={() => setMoonOpen((o) => !o)}
            title="Give a sound a meaning that holds for the whole clip"
            className={`mr-1 inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors disabled:opacity-40 ${moonOpen ? 'border-accent bg-accent text-white' : 'border-line bg-surface text-fg-2 hover:border-line-strong hover:text-fg'}`}>
            <MoonIcon />{FEATURES.symbolism}
            {rules.list.length > 0 && <span className={`rounded-full px-1.5 text-[10px] ${moonOpen ? 'bg-white/25' : 'bg-accent-soft text-accent'}`}>{rules.list.length}</span>}
          </button>
          <IconButton label={undoStep ? `Undo ${undoStep.label} (⌘Z)` : 'Undo (⌘Z)'} disabled={!canUndoAny || (Boolean(working) && !canUndo(history))} onClick={undoAny}>
            <path d="M9 14L4 9l5-5M4 9h10.5a5.5 5.5 0 010 11H11" />
          </IconButton>
          <IconButton label={redoStep ? `Redo ${redoStep.label} (⇧⌘Z)` : 'Redo (⇧⌘Z)'} disabled={!canRedoAny || (Boolean(working) && !canRedo(history))} onClick={redoAny}>
            <path d="M15 14l5-5-5-5M20 9H9.5a5.5 5.5 0 000 11H13" />
          </IconButton>
          <span
            title={chipErrors.join(' ') || statusText}
            className="ml-1 inline-flex max-w-52 items-center gap-2 truncate rounded-full border border-line bg-surface px-2.5 py-1 text-[11px] text-muted"
          >
            <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${chipErrors.length > 0 || status === 'error' ? 'bg-red-400' : status === 'ready' ? 'bg-accent' : 'animate-pulse-dot bg-faint'}`} />
            <span className="truncate">{statusText}</span>
          </span>
          <details className="relative">
            <summary aria-label="More" title="More" className="flex h-8 w-8 cursor-pointer list-none items-center justify-center rounded-full text-muted transition-colors hover:bg-tint/[0.06] hover:text-fg">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.8" /><circle cx="12" cy="12" r="1.8" /><circle cx="19" cy="12" r="1.8" /></svg>
            </summary>
            <div className="shadow-card absolute right-0 top-10 z-30 w-64 rounded-xl border border-line bg-surface p-1.5 text-xs">
              <button className={MENU_ITEM} disabled={status !== 'ready'} onClick={() => picker.current?.click()}>Import audio files</button>
              <button className={MENU_ITEM} disabled={status !== 'ready'} onClick={() => downloadText(serializeEditList(tracks), 'tidepool-edit-list.json')}>Download edit list (JSON)</button>
              <div className="mt-1 border-t border-line px-2 pt-2 pb-1">
                <p className="mb-1 font-medium text-fg">Edit history</p>
                <div className="max-h-40 overflow-y-auto"><EditLog log={log} /></div>
              </div>
            </div>
          </details>
        </div>
      </header>

      {status === 'error' ? (
        <div className="relative z-10 flex flex-1 items-center justify-center p-8 text-sm text-red-600">{error}</div>
      ) : (
        <main ref={mainRef} className="relative z-10 flex min-h-0 flex-1 p-3 pb-0">
          {/* Left: the clip and its layers. A portrait clip stands beside the timeline; a landscape one sits above it. */}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div ref={stage} className={`flex min-h-0 flex-1 ${portrait ? 'flex-row gap-3' : 'flex-col'}`}>
              <div className={portrait ? 'flex h-full shrink-0' : 'flex min-h-[140px] shrink-0'}
                style={portrait ? { aspectRatio: String(videoAspect), maxWidth: '45%' } : autoTop && fittedTop !== null && videoSrc && videoAspect ? { height: fittedTop, width: Math.min(stageSize.width, fittedTop * videoAspect), alignSelf: 'center' } : { height: `${layout.top * 100}%` }}>
            <VideoPanel
              engine={engine}
              src={videoSrc}
              onPick={onPickVideo}
              onDuration={setVideoDuration}
              onAspect={setClipAspect}
              rotation={rotation}
              onRotate={videoSrc ? rotate : undefined}
              onError={setLoadError}
              visual={visual}
              onSelection={setVisualSelection}
              selection={visualSelection}
              seekTo={seekTo}
              audioOnly={!videoSrc && projectStems.audioOnly === true}
              onVisualAction={async (selection,description,action) => { if (!projectPanel.current) throw new Error('Project controls are not ready.'); await projectPanel.current.visualAction(selection,description,action) }}
            />
              </div>
              {!portrait && (
                <Splitter direction="row" label="Resize the video and the timeline"
                  onDrag={(y) => setTop(y - (stage.current?.getBoundingClientRect().top ?? 0))} onStep={(d) => setTop((autoTop && fittedTop !== null && videoSrc ? fittedTop : layout.top * (stage.current?.clientHeight ?? 800)) + d)} onReset={() => { setAutoTop(true); setLayout((l) => ({ ...l, top: DEFAULT_LAYOUT.top })) }} />
              )}
          <div className="shadow-card relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border border-line bg-surface">
            {engine && status === 'ready' && hasLayers ? (
              <>
                {laneView === 'mixer' ? (
                  <>
                  <div className="absolute top-1.5 left-3 z-30">{viewToggle}</div>
                  <Mixer stems={stems} tracks={tracks} onAction={dispatch} onRename={canRename ? renameLayer : undefined}
                    onDelete={canRename && stems.length > 1 ? deleteLayer : undefined} residualId={residual?.id ?? null} residualLabel={residual?.label ?? null} />
                  </>
                ) : (
                  <>
                  <MotifStrip rules={rules.list} stems={stems} engine={engine} duration={duration} onOpen={openRule} />
                  <Timeline
                    engine={engine}
                    stems={stems}
                    tracks={tracks}
                    duration={duration}
                    onAction={dispatch}
                    onKindChange={setKind}
                    syncing={syncTarget !== null}
                    onRename={canRename ? renameLayer : undefined}
                    onDelete={canRename && stems.length > 1 ? deleteLayer : undefined}
                    residualId={residual?.id ?? null}
                    residualLabel={residual?.label ?? null}
                    header={viewToggle}
                  />
                  </>
                )}
              </>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
                {status === 'ready' ? (
                  <>
                    <WaveIcon />
                    <p className="text-sm font-medium text-fg">Your layers will appear here</p>
                    <p className="max-w-sm text-xs text-muted">Pick a sound in the video or describe it below. Each separated sound becomes its own layer you can mute, solo and automate.</p>
                  </>
                ) : (
                  <DropLoader size={56} label="Loading layers…" />
                )}
              </div>
            )}
            {working && (
              <div className="rise-in pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-surface/60 backdrop-blur-[2px]">
                <DropLoader size={96} label={working} />
              </div>
            )}
          </div>
            </div>
                </div>
          <Splitter direction="column" label="Resize the steps panel"
            onDrag={(x) => setPanel(window.innerWidth - x - 12)} onStep={(d) => setPanel(layout.panel - d)} onReset={() => setLayout((l) => ({ ...l, panel: DEFAULT_LAYOUT.panel }))} />
          {/* Right: the four steps, the full height of the window so nothing in them is cut off. */}
          <div className="relative flex min-h-0 shrink-0 flex-col pb-3" style={{ width: layout.panel }}>
            <ProjectPanel ref={projectPanel} baseUrl={API_BASE_URL} readyToRestore={Boolean(engine)} onLoadStems={loadStems} onVideo={url => { originalVideo.current=null; setHasLocalVideo(false); setVideoSrc(url) }} onProjectState={setProjectStems} onOriginalAudio={url=>{setCompareOriginal(false);setOriginalSrc(url)}} flush={flush} onVisual={setVisual} visual={visual} selectedVisual={visualSelection} onClearSelection={() => setVisualSelection(null)} onSelectVisual={setVisualSelection} onSeek={seekVideo} onBusy={setWorking} projectId={openProjectId} onStep={onStep} onUndo={undoAny} />
            {moonOpen && engine && hasLayers && (
              <div className="absolute inset-0 z-20 flex flex-col bg-canvas pb-3">
                <MoonPullPanel rules={rules.list} setRules={setRules} selectedId={moonRule} onSelect={setMoonRule} stems={stems} getTracks={getTracks}
                  engine={engine} duration={duration} withLayers={withLayers} onAutomation={setRuleAutomation} onClose={() => setMoonOpen(false)} />
              </div>
            )}
          </div>
        </main>
      )}

      <PromptBar disabled={status !== 'ready' || (!projectStems.projectId && !hasLocalVideo)} disabledReason={projectStems.projectId || hasLocalVideo ? '' : NO_PROJECT_HINT} onRun={runPrompt}
        split={projectStems.split ? { suggested: projectStems.split.suggested, onSplit: () => void projectPanel.current?.splitSong(), busy: Boolean(working) } : null} />
    </div>
  )
}

const LAYOUT_KEY = 'tidepool.layout'
/** The least height the timeline keeps when a widescreen clip sizes the video box. */
const TIMELINE_MIN = 200
const DEFAULT_LAYOUT = { top: 0.4, panel: 336 }
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
function readLayout(): typeof DEFAULT_LAYOUT {
  try {
    const v = JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? 'null') as { top?: unknown; panel?: unknown } | null
    if (v && typeof v.top === 'number' && typeof v.panel === 'number') return { top: clamp(v.top, 0.18, 0.78), panel: clamp(v.panel, 260, 560) }
  } catch { /* fall back to the default */ }
  return DEFAULT_LAYOUT
}

const MENU_ITEM = 'block w-full rounded-lg px-2 py-1.5 text-left text-fg transition-colors hover:bg-surface-2 disabled:opacity-40'

function IconButton({ label, disabled, onClick, children }: { label: string; disabled?: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button aria-label={label} title={label} disabled={disabled} onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded-full text-muted transition-colors hover:bg-tint/[0.06] hover:text-fg disabled:opacity-30 disabled:hover:bg-transparent">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>
    </button>
  )
}

function WaveIcon() {
  return (
    <div className="mb-1 flex h-12 items-center gap-1" aria-hidden="true">
      {[10, 22, 34, 18, 40, 26, 14, 30, 20, 8].map((h, i) => (
        <span key={i} className="w-1.5 rounded-full bg-sky" style={{ height: h, opacity: 0.5 + (i % 3) * 0.2 }} />
      ))}
    </div>
  )
}

