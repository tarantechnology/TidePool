import { useEffect, useEffectEvent, useMemo, useRef, useState } from 'react'
import type { BackendConfig, VisionCapability, VisualMetadata } from '../api/backend'
import { getPrepareVisual, getVisualJob, getVisualMetadata, startPrepareVisual, startVisualAnalysis } from '../api/backend'
import { pollJob, toJobStatus } from '../api/jobs'
import { IDLE, detectorStatus, followVisualJob, guidedByText, objectTracks, runFromError, selectTrack, summarize, trackColor } from '../objects'
import type { DetectionRun, StatusTone } from '../objects'
import { asVisualMetadata } from '../visual'
import type { VisualSelection } from '../visual'

const tones: Record<StatusTone, string> = {
  muted: 'border-line text-muted',
  ready: 'border-accent/40 text-accent',
  busy: 'border-sky-300 text-sky-700',
  ok: 'border-accent/40 text-accent',
  warn: 'border-amber-300 text-amber-700',
  error: 'border-red-200 text-red-600',
}
const dot: Record<StatusTone, string> = { muted: 'bg-faint', ready: 'bg-accent', busy: 'bg-sky-300 animate-pulse', ok: 'bg-accent', warn: 'bg-amber-300', error: 'bg-red-400' }

interface Props {
  config: BackendConfig
  projectId: string
  /** Undefined when the backend did not report a detector. */
  vision: VisionCapability | undefined
  visual: VisualMetadata | null
  selection: VisualSelection | null
  /** Another project operation is running. */
  disabled: boolean
  /** A visual job that was already running when the project was resumed. */
  resumeJobId: string | null
  /** Re-encoding is refused once a project has stems. */
  canPrepare: boolean
  onAnalyzed: (metadata: VisualMetadata) => void
  onSelect: (selection: VisualSelection) => void
  onSeek: (seconds: number) => void
  /** The project's video changed on the server: reload it. */
  onPrepared: () => Promise<void>
}

/** Detector status, Find objects, and the tracked objects it found. Keyed by project, so a new project starts clean. */
export function ObjectsPanel({ config, projectId, vision, visual, selection, disabled, resumeJobId, canPrepare, onAnalyzed, onSelect, onSeek, onPrepared }: Props) {
  const [run, setRun] = useState<DetectionRun>(IDLE)
  const abort = useRef<AbortController | null>(null)
  const followed = useRef<string | null>(null)
  const section = useRef<HTMLElement>(null)
  useEffect(() => () => abort.current?.abort(), [])

  const tracks = useMemo(() => (visual ? objectTracks(visual) : []), [visual])
  const found = useMemo(() => (visual ? summarize(visual) : null), [visual])
  const status = detectorStatus(vision, run, found)

  const start = async (work: (signal: AbortSignal, update: (r: DetectionRun) => void) => Promise<boolean>) => {
    abort.current?.abort()
    section.current?.scrollIntoView?.({ block: 'nearest' })
    const controller = new AbortController(); abort.current = controller
    const update = (next: DetectionRun) => { if (!controller.signal.aborted) setRun(next) }
    try {
      const prepared = await work(controller.signal, update)
      if (prepared) await onPrepared()
      controller.signal.throwIfAborted()
      update({ phase: 'loading', prepared })
      const metadata = asVisualMetadata(await getVisualMetadata(config, projectId))
      controller.signal.throwIfAborted()
      onAnalyzed(metadata)
      update({ phase: 'done', prepared })
    } catch (err) {
      // An unmount aborts too; then there is nothing left to update.
      if (controller.signal.aborted && abort.current !== controller) return
      setRun(runFromError(err))
    } finally {
      if (abort.current === controller) abort.current = null
    }
  }

  const detect = (jobId: string | null, preparedBefore = false) => start(async (signal, update) => {
    let prepared = preparedBefore
    let id = jobId
    if (!id) {
      update({ phase: 'queued', prepared })
      const started = await startVisualAnalysis(config, projectId)
      id = started.jobId; prepared ||= started.prepared
    }
    const job = id
    await followVisualJob((s) => getVisualJob(config, job, s), { prepared, signal, onUpdate: update })
    return prepared
  })

  const prepareThenDetect = () => start(async (signal, update) => {
    update({ phase: 'preparing', prepared: false })
    const id = await startPrepareVisual(config, projectId)
    const result = await pollJob(async (s) => toJobStatus(await getPrepareVisual(config, id, s)), { signal })
    if (result.state === 'failed') throw new Error(result.error || 'Preparing a compatible copy failed')
    await onPrepared()
    signal.throwIfAborted()
    update({ phase: 'queued', prepared: true })
    const started = await startVisualAnalysis(config, projectId)
    await followVisualJob((s) => getVisualJob(config, started.jobId, s), { prepared: true, signal, onUpdate: update })
    return false // already reloaded
  })

  const follow = useEffectEvent((jobId: string) => { void detect(jobId) })
  useEffect(() => {
    if (!resumeJobId || followed.current === resumeJobId) return
    followed.current = resumeJobId
    follow(resumeJobId)
  }, [resumeJobId])

  const cancel = () => { abort.current?.abort(); setRun({ phase: 'cancelled' }) }
  const pick = (trackId: number) => {
    if (!visual) return
    const chosen = selectTrack(visual, trackId)
    if (!chosen) return
    onSeek(chosen.timestamp_seconds)
    onSelect(chosen)
  }
  const progress = run.phase === 'detecting' ? run.progress : undefined

  return (
    <section ref={section} aria-label="Objects" className="space-y-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-medium text-fg">Objects in the video</h3>
      </div>
      <p role="status" className={`flex items-start gap-1.5 rounded-full border px-2 py-1 leading-snug ${tones[status.tone]}`}>
        <span aria-hidden="true" className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${dot[status.tone]}`} />
        <span className="min-w-0 break-words">{status.text}</span>
      </p>
      {status.running ? (
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-tint/10" role="progressbar" aria-label="Detection progress" aria-valuemin={0} aria-valuemax={100} {...(progress !== undefined ? { 'aria-valuenow': progress } : {})}>
            <div className={`h-full bg-accent transition-[width] ${progress === undefined ? 'w-1/3 animate-pulse' : ''}`} style={progress !== undefined ? { width: `${progress}%` } : undefined} />
          </div>
          <button type="button" className="rounded-lg border border-line px-2 py-1 text-fg" onClick={cancel}>Cancel</button>
        </div>
      ) : (
        <button type="button" className="w-full rounded-lg border border-accent/40 bg-accent-soft px-3 py-1.5 font-medium text-accent transition-colors hover:border-accent/60 disabled:opacity-40" disabled={disabled || !status.canStart} onClick={() => void detect(null)}>
          Find objects in video
        </button>
      )}
      {run.phase === 'failed' && run.status === 422 && !status.running && (canPrepare ? (
        <div className="space-y-1">
          <p className="text-muted">This server did not re-encode the video itself. Prepare a compatible copy here and detection runs again; the uploaded file is kept.</p>
          <button type="button" className="rounded-lg bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-40" disabled={disabled} onClick={() => void prepareThenDetect()}>Prepare a compatible copy</button>
        </div>
      ) : (
        <p className="text-muted">A compatible copy can only be prepared before any sound is separated. Start a new project from this video to analyse it.</p>
      ))}
      {tracks.length > 0 && (
        <>
          <p className="text-subtle">Pick one to hear just that sound.</p>
          <ul aria-label="Detected objects" className="max-h-32 space-y-0.5 overflow-y-auto">
            {tracks.map((t) => {
              const selected = selection?.track_id === t.trackId
              return (
                <li key={t.trackId}>
                  <button type="button" aria-pressed={selected} onClick={() => pick(t.trackId)}
                    className={`flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left hover:bg-tint/[0.06] ${selected ? 'bg-tint/[0.08] ring-1 ring-accent/60' : ''}`}>
                    <span aria-hidden="true" className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: trackColor(t.trackId).css }} />
                    <span className="min-w-0 flex-1 truncate text-fg">{t.label} <span className="text-subtle">#{t.trackId}</span></span>
                    <span className="shrink-0 font-mono text-[10px] text-muted">{t.firstSeconds.toFixed(1)}–{t.lastSeconds.toFixed(1)} s · {t.frames} fr</span>
                  </button>
                </li>
              )
            })}
          </ul>
        </>
      )}
    </section>
  )
}

/** Which object guides the next separation, on the separation form. */
export function GuidedByChip({ selection, start, end, onClear }: { selection: { label?: unknown; track_id?: unknown }; start: number; end: number; onClear: () => void }) {
  const color = typeof selection.track_id === 'number' ? trackColor(selection.track_id).css : undefined
  return (
    <div className="flex items-center gap-2 rounded-full border border-line bg-tint/[0.04] px-2 py-1 text-xs text-fg" aria-label="Visual guidance">
      {color && <span aria-hidden="true" className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />}
      <span className="min-w-0 flex-1 truncate">{guidedByText(selection, start, end)}</span>
      <button type="button" className="text-accent" onClick={onClear}>Clear</button>
    </div>
  )
}
