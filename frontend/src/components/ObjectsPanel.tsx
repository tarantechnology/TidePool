import { useEffect, useEffectEvent, useMemo, useRef, useState } from 'react'
import type { BackendConfig, VisionCapability, VisualMetadata } from '../api/backend'
import { getPrepareVisual, getVisualJob, getVisualMetadata, startPrepareVisual, startVisualAnalysis, storeVisualMetadata } from '../api/backend'
import { pollJob, toJobStatus } from '../api/jobs'
import { IDLE, detectorStatus, followVisualJob, guidedByText, mergeTracks, objectTracks, removeTrack, renameTrack, runFromError, selectTrack, summarize, trackColor } from '../objects'
import type { DetectionRun, ObjectTrack, StatusTone } from '../objects'
import { asVisualMetadata } from '../visual'
import type { VisualSelection } from '../visual'
import { FEATURES } from '../features'
import { FeatureTag } from './FeatureTag'

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
  /** The user fixed the detections (removed, merged or named a track); they are already saved. */
  onEdited?: (metadata: VisualMetadata, keepSelection: boolean) => void
}

/** Detector status, Find objects, and the tracked objects it found. Keyed by project, so a new project starts clean. */
export function ObjectsPanel({ config, projectId, vision, visual, selection, disabled, resumeJobId, canPrepare, onAnalyzed, onSelect, onSeek, onPrepared, onEdited }: Props) {
  const [concept, setConcept] = useState('')
  const [run, setRun] = useState<DetectionRun>(IDLE)
  const abort = useRef<AbortController | null>(null)
  const section = useRef<HTMLElement>(null)
  useEffect(() => () => abort.current?.abort(), [])

  const tracks = useMemo(() => (visual ? objectTracks(visual) : []), [visual])
  const found = useMemo(() => (visual ? summarize(visual) : null), [visual])
  const status = detectorStatus(vision, run, found)
  const automatic = !vision?.selectionOnly || vision.automatic === true
  const suggestedConcepts = [...new Set(tracks.map(t => t.label))]

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

  const detect = (jobId: string | null, preparedBefore = false, target = 'auto') => start(async (signal, update) => {
    let prepared = preparedBefore
    let id = jobId
    if (!id) {
      update({ phase: 'queued', prepared })
      const started = await startVisualAnalysis(config, projectId, target)
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
    if (!resumeJobId) return
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
  const selectedTrack = tracks.find((t) => t.trackId === selection?.track_id) ?? null

  // Fixing detections: each edit is saved to the project, and the last one can be undone.
  const [saving, setSaving] = useState(false)
  const [editError, setEditError] = useState('')
  const [lastEdit, setLastEdit] = useState<{ text: string; before: VisualMetadata } | null>(null)
  const save = async (next: VisualMetadata, text: string, keepSelection: boolean) => {
    if (!visual || !onEdited) return
    const before = visual
    setSaving(true); setEditError('')
    try {
      await storeVisualMetadata(config, projectId, next)
      onEdited(next, keepSelection)
      setLastEdit(text ? { text, before } : null)
    } catch (err) {
      setEditError(`Could not save: ${err instanceof Error ? err.message : String(err)}`)
    } finally { setSaving(false) }
  }
  const undoEdit = async () => {
    if (!lastEdit || !onEdited) return
    setSaving(true); setEditError('')
    try { await storeVisualMetadata(config, projectId, lastEdit.before); onEdited(lastEdit.before, false); setLastEdit(null) }
    catch (err) { setEditError(`Could not undo: ${err instanceof Error ? err.message : String(err)}`) }
    finally { setSaving(false) }
  }

  return (
    <section ref={section} aria-label="Objects" className="space-y-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 font-medium text-fg">Objects in the video <FeatureTag name={FEATURES.visual} /></h3>
      </div>
      <p role="status" className={`flex items-start gap-1.5 rounded-full border px-2 py-1 leading-snug ${tones[status.tone]}`}>
        <span aria-hidden="true" className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${dot[status.tone]}`} />
        <span className="min-w-0 break-words">{status.text}</span>
      </p>
      {vision?.selectionOnly && vision.prompted !== false && <details><summary className="cursor-pointer text-muted">Find a specific object with SAM</summary><label className="mt-2 block">Object to find<input aria-label="Object to find" placeholder="Optional, e.g. dog or person in a white hoodie" className="mt-1 w-full rounded border border-line bg-surface px-2 py-1" value={concept} maxLength={80} disabled={disabled || status.running} onChange={e=>setConcept(e.target.value)}/></label><button type="button" className="mt-1 rounded border border-line px-2 py-1" disabled={disabled || status.running || !concept.trim()} onClick={()=>void detect(null,false,concept.trim())}>Find with SAM</button></details>}
      {status.running ? (
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-tint/10" role="progressbar" aria-label="Detection progress" aria-valuemin={0} aria-valuemax={100} {...(progress !== undefined ? { 'aria-valuenow': progress } : {})}>
            <div className={`h-full bg-accent transition-[width] ${progress === undefined ? 'w-1/3 animate-pulse' : ''}`} style={progress !== undefined ? { width: `${progress}%` } : undefined} />
          </div>
          <button type="button" className="rounded-lg border border-line px-2 py-1 text-fg" onClick={cancel}>Cancel</button>
        </div>
      ) : (
        <button type="button" className="w-full rounded-lg border border-accent/40 bg-accent-soft px-3 py-1.5 font-medium text-accent transition-colors hover:border-accent/60 disabled:opacity-40" disabled={disabled || !status.canStart || (!automatic && !concept.trim())} onClick={() => void detect(null,false,automatic ? 'auto' : concept.trim())}>
          {automatic ? 'Find people and objects' : 'Find with SAM'}
        </button>
      )}
      {visual?.discovery_mode==='automatic' && suggestedConcepts.length>0 && vision?.prompted !== false && <div className="space-y-1"><p className="text-muted">People and common objects found automatically. Refine a category with SAM for a more precise outline:</p><div className="flex flex-wrap gap-1">{suggestedConcepts.map(label=><button type="button" key={label} disabled={disabled || status.running} className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={()=>void detect(null,false,label)}>Refine {label}</button>)}</div></div>}
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
          <p className="text-subtle">Pick an object, then choose its sound and an edit.</p>
          <ul aria-label="Detected objects" className="max-h-32 space-y-0.5 overflow-y-auto">
            {tracks.map((t) => {
              const selected = selection?.track_id === t.trackId
              return (
                <li key={t.trackId}>
                  <button type="button" aria-pressed={selected} disabled={disabled} onClick={() => pick(t.trackId)}
                    className={`flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left hover:bg-tint/[0.06] ${selected ? 'bg-tint/[0.08] ring-1 ring-accent/60' : ''}`}>
                    <span aria-hidden="true" className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: trackColor(t.trackId).css }} />
                    <span className="min-w-0 flex-1 truncate text-fg">{t.name ? <>{t.name} <span className="text-subtle">{t.label} #{t.trackId}</span></> : <>{t.label} <span className="text-subtle">#{t.trackId}</span></>}</span>
                    <span className="shrink-0 font-mono text-[10px] text-muted">{t.firstSeconds.toFixed(1)}–{t.lastSeconds.toFixed(1)} s · {t.frames} fr</span>
                  </button>
                </li>
              )
            })}
          </ul>
          {selectedTrack && visual && onEdited && <FixDetection key={selectedTrack.trackId} track={selectedTrack} others={tracks.filter(t => t.trackId !== selectedTrack.trackId)} busy={disabled || saving}
            onRemove={() => void save(removeTrack(visual, selectedTrack.trackId), `Removed ${titleOf(selectedTrack)}.`, false)}
            onMerge={(other) => void save(mergeTracks(visual, selectedTrack.trackId, other.trackId), `${titleOf(other)} is now part of ${titleOf(selectedTrack)}.`, false)}
            onRename={(name) => void save(renameTrack(visual, selectedTrack.trackId, name), '', true)} />}
        </>
      )}
      {lastEdit && <p role="status" className="flex items-center gap-2 rounded-lg bg-surface-2 px-2 py-1.5 text-muted"><span className="min-w-0 flex-1">{lastEdit.text}</span><button type="button" className="shrink-0 text-accent hover:underline disabled:opacity-40" disabled={saving} onClick={() => void undoEdit()}>Undo</button></p>}
      {editError && <p role="alert" className="text-red-600">{editError}</p>}
    </section>
  )
}

const titleOf = (t: ObjectTrack) => t.name || `${t.label} #${t.trackId}`

/**
 * Fix what the tracker got wrong about the selected object: name it, say it is the
 * same as another entry (a new number after a cut), or remove it because nothing is there.
 */
function FixDetection({ track, others, busy, onRemove, onMerge, onRename }: { track: ObjectTrack; others: ObjectTrack[]; busy: boolean; onRemove: () => void; onMerge: (other: ObjectTrack) => void; onRename: (name: string) => void }) {
  const [name, setName] = useState(track.name ?? '')
  const [same, setSame] = useState('')
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => { box.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }) }, [])
  const commitName = () => { if (name.trim() !== (track.name ?? '')) onRename(name) }
  // The same kind of object first: a person is usually merged with another person.
  const choices = [...others].sort((a, b) => Number(b.label === track.label) - Number(a.label === track.label) || a.firstSeconds - b.firstSeconds)
  const other = choices.find(t => String(t.trackId) === same)
  const field = 'min-w-0 flex-1 rounded-lg border border-line bg-surface px-2 py-1 text-fg focus:border-accent/50 focus:outline-none focus:ring-2 focus:ring-accent/15'
  return (
    <div ref={box} aria-label="Fix this detection" className="rise-in space-y-2 rounded-xl border border-line bg-surface p-2.5">
      <p className="flex items-center gap-1.5 font-medium text-fg"><span aria-hidden="true" className="h-2.5 w-2.5 rounded-sm" style={{ background: trackColor(track.trackId).css }} />{titleOf(track)}<span className="ml-auto font-mono text-[10px] font-normal text-subtle">{track.firstSeconds.toFixed(1)}–{track.lastSeconds.toFixed(1)} s</span></p>
      <label className="flex items-center gap-2 text-muted">Name
        <input className={field} aria-label="Object name" value={name} maxLength={40} placeholder={`e.g. ${track.label === 'person' ? 'Noah' : track.label}`} disabled={busy}
          onChange={e => setName(e.target.value)} onBlur={commitName} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} />
      </label>
      {choices.length > 0 && <div className="flex items-center gap-2 text-muted">
        <label className="flex min-w-0 flex-1 items-center gap-2">Same as
          <select className={field} aria-label="Same object as" value={same} disabled={busy} onChange={e => setSame(e.target.value)}>
            <option value="">Choose…</option>
            {choices.map(t => <option key={t.trackId} value={t.trackId}>{titleOf(t)} · {t.firstSeconds.toFixed(1)}–{t.lastSeconds.toFixed(1)} s</option>)}
          </select>
        </label>
        <button type="button" className="shrink-0 rounded-lg border border-accent/40 bg-accent-soft px-2 py-1 font-medium text-accent disabled:opacity-40" disabled={busy || !other} onClick={() => other && onMerge(other)}>Merge</button>
      </div>}
      <button type="button" className="w-full rounded-lg border border-line px-2 py-1 text-muted hover:border-red-200 hover:text-red-600 disabled:opacity-40" disabled={busy} onClick={onRemove}>Not a real {track.label}: remove it</button>
    </div>
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
