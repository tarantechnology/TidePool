// Object detection in the editor: what the detector status says, the list of
// tracked objects a detections document holds, their overlay colours, and the
// polling that follows a visual job. Components stay thin and render this.

import type { VisionCapability, VisualFrame, VisualJob, VisualMetadata, VisualObject } from './api/backend'
import { backendDetail, backendStatus, isAbort } from './api/errors'
import { pollJob, toJobStatus } from './api/jobs'
import type { JobStatus } from './api/jobs'
import { selectVisual } from './visual'
import type { VisualSelection } from './visual'

// ---- Detection run state ----------------------------------------------------

export type DetectionRun =
  | { phase: 'idle' }
  | { phase: 'queued'; prepared: boolean }
  | { phase: 'preparing'; prepared: boolean }
  | { phase: 'detecting'; prepared: boolean; progress?: number }
  | { phase: 'loading'; prepared: boolean }
  | { phase: 'done'; prepared: boolean }
  | { phase: 'failed'; error: string; status?: number }
  | { phase: 'unavailable'; reason: string }
  | { phase: 'cancelled' }

export const IDLE: DetectionRun = { phase: 'idle' }

export function isRunning(run: DetectionRun): boolean {
  return run.phase === 'queued' || run.phase === 'preparing' || run.phase === 'detecting' || run.phase === 'loading'
}

/** What the Objects section found, for the status line. */
export interface FoundSummary { objects: number; seconds: number }

export type StatusTone = 'muted' | 'ready' | 'busy' | 'ok' | 'warn' | 'error'
export interface DetectorStatus { text: string; tone: StatusTone; canStart: boolean; running: boolean }

const PREPARED_NOTE = ' — re-encoded a compatible copy'

export function foundText({ objects, seconds }: FoundSummary): string {
  const span = `in first ${seconds.toFixed(1)} s`
  return objects ? `${objects} object${objects === 1 ? '' : 's'} found ${span}` : `No tracked objects found ${span}`
}

function cleanModel(model: string): string {
  return model.replace(/\.pt$/i, '')
}

export function readyText(vision: VisionCapability): string {
  const parts = [vision.model && cleanModel(vision.model), vision.device?.toUpperCase()].filter(Boolean)
  const base = parts.length ? `Detector ready (${parts.join(', ')})` : 'Detector ready'
  return vision.weightsCached === false ? `${base}; weights download on first run` : base
}

/**
 * The status pill. `vision` is undefined when the backend did not report the
 * detector (an older backend, or capabilities failed to load): then Find
 * objects stays enabled and the backend has the final word.
 */
export function detectorStatus(vision: VisionCapability | undefined, run: DetectionRun, found: FoundSummary | null): DetectorStatus {
  const installed = vision?.available !== false || vision?.retryable === true
  const note = (prepared: boolean) => (prepared ? PREPARED_NOTE : '')
  switch (run.phase) {
    case 'queued': return { text: `Waiting for the detector…${note(run.prepared)}`, tone: 'busy', canStart: false, running: true }
    case 'preparing': return { text: 'Preparing video…', tone: 'busy', canStart: false, running: true }
    case 'detecting': {
      const pct = run.progress === undefined ? '' : ` ${run.progress}%`
      return { text: `Detecting objects…${pct}${note(run.prepared)}`, tone: 'busy', canStart: false, running: true }
    }
    case 'loading': return { text: `Loading detections…${note(run.prepared)}`, tone: 'busy', canStart: false, running: true }
    case 'done': return { text: `${found ? foundText(found) : 'Detection finished'}${note(run.prepared)}`, tone: found?.objects ? 'ok' : 'warn', canStart: installed, running: false }
    case 'failed': return { text: `Detection failed: ${run.error}`, tone: 'error', canStart: installed, running: false }
    case 'unavailable': return { text: `Detector not installed — ${run.reason}`, tone: 'error', canStart: false, running: false }
    case 'cancelled': return { text: 'Stopped waiting. Detection may still finish on the server.', tone: 'muted', canStart: installed, running: false }
    case 'idle': break
  }
  if (vision && !vision.available) return { text: `${vision.retryable ? 'Detector connection unavailable' : 'Detector not installed'} — ${vision.reason || 'no reason given'}`, tone: 'error', canStart: installed, running: false }
  if (found) return { text: foundText(found), tone: found.objects ? 'ok' : 'warn', canStart: true, running: false }
  if (vision) return { text: readyText(vision), tone: 'ready', canStart: true, running: false }
  return { text: 'Detector status unknown', tone: 'muted', canStart: true, running: false }
}

// ---- Visual job polling -----------------------------------------------------

export interface VisualJobStatus extends JobStatus { stage?: 'preparing' | 'detecting' }

export function toVisualJobStatus(job: VisualJob): VisualJobStatus {
  // Only the two visual stages are kept; anything else the job reports is dropped.
  const { stage: _ignored, ...status } = toJobStatus(job)
  const progress = status.progress
  return {
    ...status,
    ...(progress !== undefined ? { progress: Math.round(Math.min(100, Math.max(0, progress))) } : {}),
    ...(job.stage === 'preparing' || job.stage === 'detecting' ? { stage: job.stage } : {}),
  }
}

/** The run state a poll result shows while the job is still going. */
export function runFromJob(status: VisualJobStatus, prepared: boolean): DetectionRun {
  if (status.state === 'queued') return { phase: 'queued', prepared }
  if (status.stage === 'preparing') return { phase: 'preparing', prepared }
  return { phase: 'detecting', prepared, ...(status.progress !== undefined ? { progress: status.progress } : {}) }
}

/**
 * Follow a visual job to the end. Resolves when it completed, throws with the
 * job's own error when it failed, and rejects with the abort reason when the
 * signal fires (Cancel, or the panel unmounting).
 */
export async function followVisualJob(
  getJob: (signal?: AbortSignal) => Promise<VisualJob>,
  opts: { prepared: boolean; signal?: AbortSignal; intervalMs?: number; onUpdate?: (run: DetectionRun) => void },
): Promise<void> {
  const result = await pollJob(async (signal) => toVisualJobStatus(await getJob(signal)), {
    signal: opts.signal,
    intervalMs: opts.intervalMs,
    onUpdate: (status) => { if (status.state === 'queued' || status.state === 'processing') opts.onUpdate?.(runFromJob(status, opts.prepared)) },
  })
  if (result.state === 'failed') throw new Error(result.error || 'Visual analysis failed')
}

/** Why a run ended, as the status pill shows it. Cancelling is not a failure. */
export function runFromError(err: unknown): DetectionRun {
  if (isAbort(err)) return { phase: 'cancelled' }
  const status = backendStatus(err)
  const detail = backendDetail(err)
  if (status === 503) return { phase: 'unavailable', reason: detail || 'the server has no object detector' }
  if (status === 409) return { phase: 'failed', status, error: detail || 'Another detection is already running for this project' }
  if (status === 422) return { phase: 'failed', status, error: `this video cannot be analysed${detail ? ` (${detail})` : ''}` }
  if (status !== null) return { phase: 'failed', status, error: detail ? `${detail} (HTTP ${status})` : `the server returned HTTP ${status}` }
  return { phase: 'failed', error: err instanceof Error ? err.message : String(err) }
}

// ---- Object list ------------------------------------------------------------

/** One tracked object with masks, as a row in the object list. */
export interface ObjectTrack {
  trackId: number
  label: string
  /** Timestamp of the first and last frame it was seen in, in seconds. */
  firstSeconds: number
  lastSeconds: number
  firstFrameIndex: number
  lastFrameIndex: number
  frames: number
}

/** Only tracked objects with a mask can guide separation, so only those are listed and drawn. */
export function isSelectable(object: VisualObject): object is VisualObject & { track_id: number } {
  return object.track_id !== null && Number.isInteger(object.track_id) && Boolean(object.mask)
}

export function objectTracks(metadata: VisualMetadata): ObjectTrack[] {
  const byTrack = new Map<number, ObjectTrack & { labels: Map<string, number> }>()
  for (const frame of metadata.frames) {
    for (const object of frame.objects) {
      if (!isSelectable(object)) continue
      let row = byTrack.get(object.track_id)
      if (!row) {
        row = { trackId: object.track_id, label: object.label, firstSeconds: frame.timestamp_seconds, lastSeconds: frame.timestamp_seconds,
          firstFrameIndex: frame.frame_index, lastFrameIndex: frame.frame_index, frames: 0, labels: new Map() }
        byTrack.set(object.track_id, row)
      }
      row.frames++
      row.labels.set(object.label, (row.labels.get(object.label) ?? 0) + 1)
      if (frame.timestamp_seconds < row.firstSeconds) { row.firstSeconds = frame.timestamp_seconds; row.firstFrameIndex = frame.frame_index }
      if (frame.timestamp_seconds > row.lastSeconds) { row.lastSeconds = frame.timestamp_seconds; row.lastFrameIndex = frame.frame_index }
    }
  }
  return [...byTrack.values()]
    // Suppress one/two-frame class flicker from the fast automatic scan. SAM
    // observations remain available at their native cadence, including brief ones.
    .filter(row => metadata.discovery_mode !== 'automatic' || row.frames >= 3)
    .map(({ labels, ...row }) => {
      // A tracker can relabel an object between frames; show the label it had most often.
      let label = row.label, best = 0
      for (const [name, count] of labels) if (count > best) { label = name; best = count }
      return { ...row, label }
    })
    .sort((a, b) => a.firstSeconds - b.firstSeconds || a.trackId - b.trackId)
}

export function summarize(metadata: VisualMetadata): FoundSummary {
  return { objects: objectTracks(metadata).length, seconds: metadata.processed_duration_seconds }
}

const round = (n: number) => Math.round(n * 1e9) / 1e9

/**
 * Select a whole track from the list: the same selection a click on the video
 * makes at the object's first frame, but its interval covers the frames the
 * object was seen in. The backend wants a frame-aligned interval of 0.1 to 30
 * seconds inside the processed media, so it is clamped to that.
 */
export function selectTrack(metadata: VisualMetadata, trackId: number): VisualSelection | null {
  const tracks = objectTracks(metadata)
  const track = tracks.find((t) => t.trackId === trackId)
  if (!track) return null
  const frame = metadata.frames.find((f) => f.frame_index === track.firstFrameIndex)
  const object = frame?.objects.find((o) => isSelectable(o) && o.track_id === trackId)
  if (!frame || !object) return null
  const selection = selectVisual(metadata, frame.timestamp_seconds, object)
  if (!selection) return null
  const fps = metadata.fps
  const limit = Math.max(1, Math.round(metadata.processed_frames))
  const maxFrames = Math.max(1, Math.floor(30 * fps + 1e-9))
  const minFrames = Math.max(1, Math.ceil(0.1 * fps - 1e-9))
  let start = track.firstFrameIndex
  let end = Math.min(limit, track.lastFrameIndex + 1, start + maxFrames)
  if (end - start < minFrames) end = Math.min(limit, start + minFrames)
  if (end - start < minFrames) start = Math.max(0, end - minFrames)
  return {
    ...selection,
    interval: { start_seconds: round(start / fps), end_seconds: Math.min(metadata.processed_duration_seconds, round(end / fps)) },
  }
}

/** The chip on the separation form. */
export function guidedByText(selection: { label?: unknown; track_id?: unknown }, start: number, end: number): string {
  const label = typeof selection.label === 'string' && selection.label ? selection.label : 'object'
  const id = typeof selection.track_id === 'number' ? ` #${selection.track_id}` : ''
  const range = Number.isFinite(start) && Number.isFinite(end) ? ` (${start.toFixed(1)}–${end.toFixed(1)} s)` : ''
  return `Guided by: ${label}${id}${range}`
}

// ---- Overlay ----------------------------------------------------------------

export interface TrackColor { css: string; rgb: [number, number, number] }

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h / 30) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)]
}

/**
 * A stable colour per track id. Integer hashing scrambles neighbouring ids,
 * then the golden angle spreads hues, so tracks 7 and 8 do not look alike.
 */
export function trackColor(trackId: number): TrackColor {
  let x = (Math.trunc(trackId) | 0) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b) >>> 0
  x = (x ^ (x >>> 16)) >>> 0
  const hue = Math.round((x * 137.508) % 360)
  return { css: `hsl(${hue} 85% 60%)`, rgb: hslToRgb(hue, 0.85, 0.6) }
}

export interface OverlayBox { object: VisualObject & { track_id: number }; trackId: number; label: string; box: [number, number, number, number]; color: string; selected: boolean }

/** Boxes to draw on a frame: tracked objects with masks, the selected one last so it sits on top. */
export function overlayBoxes(frame: VisualFrame | undefined, selectedTrack: number | null): OverlayBox[] {
  if (!frame) return []
  return frame.objects
    .filter(isSelectable)
    .filter((o) => o.bbox_normalized.length === 4 && o.bbox_normalized.every(Number.isFinite))
    .map((object) => ({
      object,
      trackId: object.track_id,
      label: `${object.label} #${object.track_id}`,
      box: object.bbox_normalized as [number, number, number, number],
      color: trackColor(object.track_id).css,
      selected: object.track_id === selectedTrack,
    }))
    .sort((a, b) => Number(a.selected) - Number(b.selected))
}

// ---- "Show objects" preference ----------------------------------------------

export const SHOW_OBJECTS_KEY = 'soundcanvas.showObjects'

type Store = Pick<Storage, 'getItem' | 'setItem'>
const defaultStore = (): Store | undefined => {
  try { return globalThis.localStorage } catch { return undefined }
}

/** Defaults to on. Storage can be missing or throw (private windows, blocked site data). */
export function readShowObjects(store: Store | undefined = defaultStore()): boolean {
  try { return store?.getItem(SHOW_OBJECTS_KEY) !== '0' } catch { return true }
}

export function writeShowObjects(value: boolean, store: Store | undefined = defaultStore()): void {
  try { store?.setItem(SHOW_OBJECTS_KEY, value ? '1' : '0') } catch { /* a preference, not data */ }
}
