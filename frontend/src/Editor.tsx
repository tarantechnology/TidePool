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
  const [previewSlot, setPreviewSlot] = useState<HTMLDivElement | null>(null)
  const [working, setWorking] = useState<string | null>(null)
  const [videoAspect, setVideoAspect] = useState<number | null>(null)
  const portrait = videoSrc !== null && videoAspect !== null && videoAspect < 0.9
  const [layout, setLayout] = useState(readLayout)
  const mainRef = useRef<HTMLElement>(null)
  useEffect(() => { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)) } catch { /* layout just won't persist */ } }, [layout])
  const setTop = (px: number) => { const h = mainRef.current?.clientHeight ?? 800; setLayout((l) => ({ ...l, top: clamp(px / h, 0.18, 0.78) })) }
  const setPanel = (px: number) => setLayout((l) => ({ ...l, panel: clamp(px, 260, 560) }))
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
  const hasLayers = stems.length > 0
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
          <span className="text-xs font-normal text-muted">or audio files to replace the layers</span>
        </div>
      )}
      <div className="pointer-events-none absolute -top-40 left-1/2 h-80 w-[60rem] -translate-x-1/2 rounded-full bg-sky/30 blur-[120px]" />

      <header className="relative z-20 flex h-14 shrink-0 items-center gap-4 border-b border-line bg-surface/70 px-4 backdrop-blur">
        <a href="#/" className="shrink-0 text-fg transition-opacity hover:opacity-80">
          <Logo />
        </a>
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
          <IconButton label="Undo (⌘Z)" disabled={!canUndo(history)} onClick={() => setHistory((h) => undo(h))}>
            <path d="M9 14L4 9l5-5M4 9h10.5a5.5 5.5 0 010 11H11" />
          </IconButton>
          <IconButton label="Redo (⇧⌘Z)" disabled={!canRedo(history)} onClick={() => setHistory((h) => redo(h))}>
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
            <div className={`flex min-h-0 flex-1 ${portrait ? 'flex-row gap-3' : 'flex-col'}`}>
              <div className={portrait ? 'flex h-full shrink-0' : 'flex min-h-[140px] shrink-0'}
                style={portrait ? { aspectRatio: String(videoAspect), maxWidth: '45%' } : { height: `${layout.top * 100}%` }}>
            <VideoPanel
              engine={engine}
              src={videoSrc}
              onPick={onPickVideo}
              onDuration={setVideoDuration}
              onAspect={setVideoAspect}
              onError={setLoadError}
              visual={visual}
              onSelection={setVisualSelection}
              selection={visualSelection}
              seekTo={seekTo}
              onVisualAction={async (selection,description,action) => { if (!projectPanel.current) throw new Error('Project controls are not ready.'); await projectPanel.current.visualAction(selection,description,action) }}
            />
              </div>
              {!portrait && (
                <Splitter direction="row" label="Resize the video and the timeline"
                  onDrag={(y) => setTop(y - (mainRef.current?.getBoundingClientRect().top ?? 0))} onStep={(d) => setTop(layout.top * (mainRef.current?.clientHeight ?? 800) + d)} onReset={() => setLayout((l) => ({ ...l, top: DEFAULT_LAYOUT.top }))} />
              )}
          <div className="shadow-card relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border border-line bg-surface">
            {engine && status === 'ready' && hasLayers ? (
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
            <div ref={setPreviewSlot} className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center [&>*]:pointer-events-auto" />
          </div>
            </div>
                </div>
          <Splitter direction="column" label="Resize the steps panel"
            onDrag={(x) => setPanel(window.innerWidth - x - 12)} onStep={(d) => setPanel(layout.panel - d)} onReset={() => setLayout((l) => ({ ...l, panel: DEFAULT_LAYOUT.panel }))} />
          {/* Right: the four steps, the full height of the window so nothing in them is cut off. */}
          <div className="flex min-h-0 shrink-0 flex-col pb-3" style={{ width: layout.panel }}>
            <ProjectPanel ref={projectPanel} baseUrl={API_BASE_URL} onLoadStems={loadStems} onVideo={url => { originalVideo.current=null; setHasLocalVideo(false); setVideoSrc(url) }} onProjectState={setProjectStems} onOriginalAudio={url=>{setCompareOriginal(false);setOriginalSrc(url)}} flush={flush} onVisual={setVisual} visual={visual} selectedVisual={visualSelection} onClearSelection={() => setVisualSelection(null)} onSelectVisual={setVisualSelection} onSeek={seekVideo} previewSlot={previewSlot} onBusy={setWorking} />
          </div>
        </main>
      )}

      <PromptBar disabled={status !== 'ready' || (!projectStems.projectId && !hasLocalVideo)} disabledReason={projectStems.projectId || hasLocalVideo ? '' : NO_PROJECT_HINT} onRun={runPrompt} />
    </div>
  )
}

const LAYOUT_KEY = 'tidepool.layout'
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

