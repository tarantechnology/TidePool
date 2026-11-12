import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
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
import type { ProjectPanelHandle } from './components/ProjectPanel'
import { Logo } from './components/Logo'
import { PromptBar } from './components/PromptBar'
import { Timeline } from './components/Timeline'
import { Transport } from './components/Transport'
import { VideoPanel } from './components/VideoPanel'
import { applyAction, initialTracks } from './edits'
import { downloadText } from './export/download'
import { serializeEditList } from './export/editList'
import { useStemSync } from './hooks/useStemSync'
import { canRedo, canUndo, commit, createHistory, redo, undo } from './state/history'
import type { History } from './state/history'
import { createEditorState, mergeKinds, setKindOverride } from './state/kinds'
import type { EditorState } from './state/kinds'
import type { EditAction, Stem, StemKind, TrackState } from './types'

type Status = 'loading' | 'ready' | 'error'

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || '/api'

const MOCK_LABEL = 'Mock stems'
const VIDEO_AUDIO_LABEL = 'Original audio'

const HEADER_BUTTON =
  'rounded-full border border-line px-2.5 py-1 font-mono text-[11px] text-zinc-400 transition-colors hover:border-line-strong hover:bg-white/[0.04] hover:text-zinc-100 disabled:opacity-40'

/** Range sliders are inputs too, but undo should still work while one has focus. */
function isTextEntry(el: HTMLElement): boolean {
  if (el.tagName === 'TEXTAREA') return true
  return el.tagName === 'INPUT' && (el as HTMLInputElement).type !== 'range'
}

export default function Editor() {
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
  const [hasLocalVideo,setHasLocalVideo]=useState(false)
  const stemsRef = useRef<Stem[]>([])
  const picker = useRef<HTMLInputElement>(null)
  const projectPanel = useRef<ProjectPanelHandle>(null)
  const originalVideo = useRef<File | null>(null)
  const loadRequest = useRef(0)
  const { tracks, kinds } = history.present
  const stems = useMemo(() => mergeKinds(baseStems, kinds), [baseStems, kinds])
  const { syncError, flush } = useStemSync(syncTarget, tracks, setLoadError)
  const chipErrors = [syncError, loadError, syncIssue].filter(Boolean)
  const tracksRef = useRef<TrackState[]>([])

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
    mock
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
      mock.dispose()
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

  const onPickVideo = useCallback((file: File) => { setCompareOriginal(false); loadVideo(file, false); void projectPanel.current?.uploadMedia(file) }, [loadVideo])

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
  }, [])

  const setKind = useCallback((stemId: string, kind: StemKind) => {
    setHistory((h) => commit(h, { ...h.present, kinds: setKindOverride(h.present.kinds, stemId, kind) }))
  }, [])

  const runPrompt = useCallback(
    async (text: string) => {
      if (!engine) return { message: 'Engine not ready.', ok: false }
      if (!projectPanel.current) return {message:'Project controls are not ready.',ok:false}
      const req = { prompt: text, stems, tracks: tracksRef.current, duration: engine.duration }
      const plan = await projectPanel.current.planCommand(text, originalVideo.current, stems, req.tracks, engine.duration)
      if (plan.kind === 'separate') return {message:plan.message,ok:true}
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
        setLog((l) => [...l, ...actions])
      }
      return { message: [res.message, ...warnings].join(' '), ok: actions.length > 0 }
    },
    [engine, stems],
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'z') return
      if (isTextEntry(e.target as HTMLElement)) return
      e.preventDefault()
      setHistory((h) => {
        if (e.shiftKey) return canRedo(h) ? redo(h) : h
        return canUndo(h) ? undo(h) : h
      })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const duration = engine?.duration ?? 0

  return (
    <div
      className="relative flex h-screen flex-col overflow-hidden bg-ink-950"
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
        <div className="pointer-events-none absolute inset-3 z-50 flex items-center justify-center rounded-xl border border-dashed border-accent/60 bg-ink-950/70 font-mono text-xs text-accent backdrop-blur">
          Drop audio files to replace the stems, or a video
        </div>
      )}
      <div className="bg-dots fade-edges pointer-events-none absolute inset-0 opacity-70" />
      <div className="pointer-events-none absolute -top-48 left-1/2 h-80 w-[56rem] -translate-x-1/2 rounded-full bg-accent/10 blur-[130px]" />

      <header className="relative z-10 grid h-14 shrink-0 grid-cols-[1fr_auto_1fr] items-center border-b border-line bg-ink-950/60 px-4 backdrop-blur">
        <a href="#/" className="text-zinc-100 transition-opacity hover:opacity-80">
          <Logo />
        </a>
        <div className="flex items-center gap-3">
          <Transport engine={engine} playing={playing} duration={duration} />
          {originalSrc && stemsLabel === 'Project stems' && <button className={HEADER_BUTTON} onClick={()=>setCompareOriginal(v=>!v)} aria-pressed={compareOriginal}>{compareOriginal ? 'Listening: original' : 'Compare original'}</button>}
          {isDurationMismatch(videoDuration, duration) && (
            <span className="font-mono text-[10px] text-amber-300">
              Video {videoDuration?.toFixed(1)}s, audio {duration.toFixed(1)}s
            </span>
          )}
        </div>
        <div className="flex items-center justify-end gap-2">
          <input ref={picker} type="file" accept="audio/*" multiple hidden onChange={onPick} />
          <button
            disabled={status !== 'ready'}
            onClick={() => picker.current?.click()}
            className={HEADER_BUTTON}
          >
            Import audio
          </button>
          <button className={HEADER_BUTTON} disabled={!canUndo(history)} onClick={()=>setHistory(h=>undo(h))} title="Undo (⌘Z)">Undo</button>
          <button className={HEADER_BUTTON} disabled={!canRedo(history)} onClick={()=>setHistory(h=>redo(h))} title="Redo (⇧⌘Z)">Redo</button>
          <button
            disabled={status !== 'ready'}
            onClick={() => downloadText(serializeEditList(tracks), 'soundcanvas-edit-list.json')}
            className={HEADER_BUTTON}
          >
            Export edit list (JSON)
          </button>
          <span
            title={chipErrors.join(' ')}
            className="inline-flex max-w-64 items-center gap-2 truncate rounded-full border border-line px-2.5 py-1 font-mono text-[11px] text-zinc-400"
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                chipErrors.length > 0
                  ? 'bg-red-400'
                  : status === 'ready'
                  ? 'bg-accent shadow-[0_0_8px_rgb(61_255_139/0.8)]'
                  : status === 'error'
                    ? 'bg-red-400'
                    : 'animate-pulse-dot bg-zinc-400'
              }`}
            />
            {status === 'loading' && 'Loading stems'}
            {status === 'ready' && (chipErrors[0] || stemsLabel)}
            {status === 'error' && 'Load failed'}
          </span>
          {showsMockBadge(projectStems,stemsLabel) && <span className="text-xs text-amber-300">test stems — no AI separation</span>}
        </div>
      </header>

      {status === 'error' ? (
        <div className="relative z-10 flex flex-1 items-center justify-center p-8 text-sm text-red-400">{error}</div>
      ) : (
        <main className="relative z-10 flex min-h-0 flex-1 flex-col">
          <div className="flex h-[38%] min-h-[190px] shrink-0 gap-3 p-3 pb-2">
            <VideoPanel
              engine={engine}
              src={videoSrc}
              onPick={onPickVideo}
              onDuration={setVideoDuration}
              onError={setLoadError}
              visual={visual}
              onSelection={setVisualSelection}
              selection={visualSelection}
              seekTo={seekTo}
              onVisualAction={async (selection,description,action) => { if (!projectPanel.current) throw new Error('Project controls are not ready.'); await projectPanel.current.visualAction(selection,description,action) }}
            />
            <div className="flex min-h-0 w-80 shrink-0 flex-col gap-2">
              <ProjectPanel ref={projectPanel} baseUrl={API_BASE_URL} onLoadStems={loadStems} onVideo={url => { originalVideo.current=null; setHasLocalVideo(false); setVideoSrc(url) }} onProjectState={setProjectStems} onOriginalAudio={url=>{setCompareOriginal(false);setOriginalSrc(url)}} flush={flush} onVisual={setVisual} visual={visual} selectedVisual={visualSelection} onClearSelection={() => setVisualSelection(null)} onSelectVisual={setVisualSelection} onSeek={seekVideo} />
              <details className="shrink-0 text-xs text-zinc-400"><summary className="cursor-pointer">Edit history</summary><EditLog log={log} /></details>
            </div>
          </div>
          <div className="mx-3 mb-3 flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-line bg-ink-900/60 backdrop-blur">
            {engine && status === 'ready' ? (
              <Timeline
                engine={engine}
                stems={stems}
                tracks={tracks}
                duration={duration}
                onAction={dispatch}
                onKindChange={setKind}
                syncing={syncTarget !== null}
              />
            ) : (
              <div className="flex flex-1 items-center justify-center font-mono text-xs text-zinc-600">
                Loading stems...
              </div>
            )}
          </div>
        </main>
      )}

      <PromptBar disabled={status !== 'ready' || (!projectStems.projectId && !hasLocalVideo)} disabledReason={projectStems.projectId || hasLocalVideo ? '' : NO_PROJECT_HINT} agentLabel="GPT-6 Luna" onRun={runPrompt} />
    </div>
  )
}
