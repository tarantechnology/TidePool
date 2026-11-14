import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fixture from './fixtures/detections.trimmed.json'
import { toCapabilities } from './api/backend'
import type { VisionCapability, VisualJob, VisualObject } from './api/backend'
import { MAX_POLL_FAILURES } from './api/jobs'
import {
  IDLE, SHOW_OBJECTS_KEY, detectorStatus, followVisualJob, guidedByText, objectTracks, overlayBoxes, readShowObjects,
  runFromError, selectTrack, summarize, toVisualJobStatus, trackColor, writeShowObjects,
} from './objects'
import type { DetectionRun } from './objects'
import { asVisualMetadata, frameAt, selectVisual } from './visual'

const metadata = asVisualMetadata(fixture)
const ready: VisionCapability = { available: true, reason: null, model: 'yolo26n-seg', weightsCached: true, device: 'cpu' }
const missing: VisionCapability = { available: false, reason: 'ultralytics is not installed', model: null, weightsCached: null, device: null }

describe('toCapabilities', () => {
  it('maps the vision block, renaming weights_cached', () => {
    expect(toCapabilities({ quality: true, visual: false, reason: '', vision: { available: true, reason: null, model: 'yolo26n-seg', weights_cached: false, device: 'cpu' } }))
      .toEqual({ quality: true, visual: false, reason: '', vision: { available: true, reason: null, model: 'yolo26n-seg', weightsCached: false, device: 'cpu' } })
  })
  it('leaves vision out for an older backend, and nulls fields it did not send', () => {
    expect(toCapabilities({ quality: false, visual: false, reason: 'Fake adapter' })).toEqual({ quality: false, visual: false, reason: 'Fake adapter' })
    expect(toCapabilities({ vision: { available: false } }).vision).toEqual({ available: false, reason: null, model: null, weightsCached: null, device: null })
    expect(toCapabilities({ vision: { reason: 'no flag' } }).vision).toBeUndefined()
    expect(toCapabilities(null)).toEqual({ quality: false, visual: false, reason: '' })
  })
  it('keeps the multi-pass fields next to vision, dropping ones of the wrong type', () => {
    expect(toCapabilities({ quality: true, visual: false, reason: '', max_passes: 3, refine: true, vision: { available: false } }))
      .toEqual({ quality: true, visual: false, reason: '', max_passes: 3, refine: true, vision: { available: false, reason: null, model: null, weightsCached: null, device: null } })
    expect(toCapabilities({ max_passes: '3', refine: 'yes' })).toEqual({ quality: false, visual: false, reason: '' })
  })
})

describe('detectorStatus', () => {
  const found = { objects: 5, seconds: 6 }
  const text = (run: DetectionRun, vision: VisionCapability | undefined = ready, f: typeof found | null = null) => detectorStatus(vision, run, f).text

  it('says when the detector is missing, and disables the button', () => {
    expect(detectorStatus(missing, IDLE, null)).toMatchObject({ text: 'Detector not installed — ultralytics is not installed', canStart: false, tone: 'error' })
    expect(text(IDLE, { ...missing, reason: null })).toBe('Detector not installed — no reason given')
    expect(detectorStatus(ready, { phase: 'unavailable', reason: 'weights missing' }, null)).toMatchObject({ text: 'Detector not installed — weights missing', canStart: false })
  })
  it('names the model and device when ready', () => {
    expect(detectorStatus(ready, IDLE, null)).toMatchObject({ text: 'Detector ready (yolo26n-seg, CPU)', canStart: true, running: false })
    expect(text(IDLE, { ...ready, model: 'yolo26n-seg.pt', weightsCached: false })).toBe('Detector ready (yolo26n-seg, CPU); weights download on first run')
    expect(text(IDLE, { ...ready, model: null, device: null })).toBe('Detector ready')
  })
  it('stays usable when the backend does not report a detector', () => {
    expect(detectorStatus(undefined, IDLE, null)).toMatchObject({ text: 'Detector status unknown', canStart: true })
  })
  it('shows each running stage, with progress, and blocks a second start', () => {
    expect(text({ phase: 'queued', prepared: false })).toBe('Waiting for the detector…')
    expect(text({ phase: 'preparing', prepared: false })).toBe('Preparing video…')
    expect(detectorStatus(ready, { phase: 'detecting', prepared: false, progress: 30 }, null)).toMatchObject({ text: 'Detecting objects… 30%', running: true, canStart: false })
    expect(text({ phase: 'detecting', prepared: false })).toBe('Detecting objects…')
    expect(text({ phase: 'loading', prepared: false })).toBe('Loading detections…')
  })
  it('adds the re-encode note when the backend prepared a copy', () => {
    expect(text({ phase: 'detecting', prepared: true, progress: 10 })).toBe('Detecting objects… 10% — re-encoded a compatible copy')
    expect(text({ phase: 'done', prepared: true }, ready, found)).toBe('5 objects found in first 6.0 s — re-encoded a compatible copy')
  })
  it('reports what was found, after a run or from loaded detections', () => {
    expect(text({ phase: 'done', prepared: false }, ready, found)).toBe('5 objects found in first 6.0 s')
    expect(text(IDLE, ready, { objects: 1, seconds: 2.5 })).toBe('1 object found in first 2.5 s')
    expect(detectorStatus(ready, IDLE, { objects: 0, seconds: 6 })).toMatchObject({ text: 'No tracked objects found in first 6.0 s', tone: 'warn' })
    expect(text(IDLE, undefined, found)).toBe('5 objects found in first 6.0 s')
  })
  it('reports failure and cancellation, and allows a retry', () => {
    expect(detectorStatus(ready, { phase: 'failed', error: 'boom' }, null)).toMatchObject({ text: 'Detection failed: boom', canStart: true, tone: 'error' })
    expect(detectorStatus(missing, { phase: 'failed', error: 'boom' }, null).canStart).toBe(false)
    expect(detectorStatus(ready, { phase: 'cancelled' }, null)).toMatchObject({ canStart: true, running: false })
  })
})

describe('object list from a real detections document', () => {
  it('has one row per tracked object, ordered by first appearance', () => {
    const rows = objectTracks(metadata)
    expect(rows.map((r) => `${r.label} #${r.trackId}`)).toEqual([
      'person #1', 'person #2', 'car #3', 'person #4', 'truck #5', 'person #7', 'person #8', 'person #9',
    ])
    expect(rows.find((r) => r.trackId === 4)).toEqual({ trackId: 4, label: 'person', firstSeconds: 0.9, lastSeconds: 1.8, firstFrameIndex: 9, lastFrameIndex: 18, frames: 10 })
    expect(rows.find((r) => r.trackId === 7)).toMatchObject({ firstSeconds: 1.1, lastSeconds: 2.4, frames: 14 })
    expect(rows.find((r) => r.trackId === 9)).toMatchObject({ firstSeconds: 2.4, lastSeconds: 2.4, frames: 1 })
    expect(summarize(metadata)).toEqual({ objects: 8, seconds: 2.5 })
  })
  it('skips untracked objects and objects without a mask, and uses the most common label', () => {
    const frame = (i: number, objects: VisualObject[]) => ({ frame_index: i, timestamp_seconds: i / 10, valid_until_seconds: (i + 1) / 10, objects })
    const doc = { ...metadata, frames: [
      frame(0, [{ detection_id: 'a', track_id: null, label: 'dog', bbox_normalized: [0, 0, 1, 1], mask: {} }, { detection_id: 'b', track_id: 2, label: 'cat', bbox_normalized: [0, 0, 1, 1], mask: null }, { detection_id: 'c', track_id: 3, label: 'dog', bbox_normalized: [0, 0, 1, 1], mask: {} }]),
      frame(1, [{ detection_id: 'd', track_id: 3, label: 'cat', bbox_normalized: [0, 0, 1, 1], mask: {} }]),
      frame(2, [{ detection_id: 'e', track_id: 3, label: 'cat', bbox_normalized: [0, 0, 1, 1], mask: {} }]),
    ] }
    expect(objectTracks(doc)).toEqual([{ trackId: 3, label: 'cat', firstSeconds: 0, lastSeconds: 0.2, firstFrameIndex: 0, lastFrameIndex: 2, frames: 3 }])
  })
})

describe('selectTrack', () => {
  it('makes the same selection a click on its first frame makes, over the frames it was seen in', () => {
    const selection = selectTrack(metadata, 7)!
    const frame = frameAt(metadata, 1.1)!
    const clicked = selectVisual(metadata, 1.1, frame.objects.find((o) => o.track_id === 7)!)!
    expect(selection).toEqual({ ...clicked, interval: { start_seconds: 1.1, end_seconds: 2.5 } })
    expect(selection).toMatchObject({ kind: 'visual_selection', track_id: 7, frame_index: 11, timestamp_seconds: 1.1, analysis_id: metadata.analysis_id })
  })
  it('keeps a one-frame track at the 0.1 s minimum the backend accepts', () => {
    expect(selectTrack(metadata, 9)?.interval).toEqual({ start_seconds: 2.4, end_seconds: 2.5 })
  })
  it('caps a long track at 30 seconds', () => {
    const frames = Array.from({ length: 400 }, (_, i) => ({ frame_index: i, timestamp_seconds: i / 10, valid_until_seconds: (i + 1) / 10,
      objects: [{ detection_id: `f${i}`, track_id: 1, label: 'car', bbox_normalized: [0, 0, 1, 1], mask: {} }] }))
    const doc = { ...metadata, frames, processed_frames: 400, processed_duration_seconds: 40 }
    expect(selectTrack(doc, 1)?.interval).toEqual({ start_seconds: 0, end_seconds: 30 })
  })
  it('returns null for a track that is not there', () => {
    expect(selectTrack(metadata, 99)).toBeNull()
  })
})

describe('guidedByText', () => {
  it('names the object, its track and the interval', () => {
    expect(guidedByText({ label: 'person', track_id: 7 }, 0, 6)).toBe('Guided by: person #7 (0.0–6.0 s)')
    expect(guidedByText(selectTrack(metadata, 7)!, 1.1, 2.5)).toBe('Guided by: person #7 (1.1–2.5 s)')
  })
  it('copes with an imported selection that lacks fields', () => {
    expect(guidedByText({}, Number.NaN, 2)).toBe('Guided by: object')
    expect(guidedByText({ label: 'dog', track_id: null }, 1, 2)).toBe('Guided by: dog (1.0–2.0 s)')
  })
})

describe('trackColor', () => {
  it('is stable for a track id and differs between neighbours', () => {
    expect(trackColor(7)).toEqual(trackColor(7))
    expect(trackColor(7).css).not.toBe(trackColor(8).css)
    expect(trackColor(7).css).toMatch(/^hsl\(\d+ 85% 60%\)$/)
  })
  it('gives an rgb triple for the mask canvas', () => {
    for (const id of [0, 1, 2, 3, 17, 123456]) {
      const { rgb } = trackColor(id)
      expect(rgb).toHaveLength(3)
      expect(rgb.every((c) => Number.isInteger(c) && c >= 0 && c <= 255)).toBe(true)
    }
  })
  it('spreads the fixture tracks across distinct hues', () => {
    const hues = objectTracks(metadata).map((t) => trackColor(t.trackId).css)
    expect(new Set(hues).size).toBe(hues.length)
  })
})

describe('overlayBoxes', () => {
  it('draws every tracked object on the frame, colour per track, selected last', () => {
    const boxes = overlayBoxes(frameAt(metadata, 1.15), 2)
    expect(boxes.map((b) => b.label)).toEqual(['person #1', 'car #3', 'person #4', 'truck #5', 'person #7', 'person #2'])
    expect(boxes.at(-1)).toMatchObject({ trackId: 2, selected: true, color: trackColor(2).css })
    expect(overlayBoxes(undefined, null)).toEqual([])
  })
})

describe('Show objects preference', () => {
  it('defaults to on and round-trips through storage', () => {
    const data = new Map<string, string>()
    const store = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) }
    expect(readShowObjects(store)).toBe(true)
    writeShowObjects(false, store)
    expect(data.get(SHOW_OBJECTS_KEY)).toBe('0')
    expect(readShowObjects(store)).toBe(false)
  })
  it('survives storage that throws or is missing', () => {
    const broken = { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') } }
    expect(readShowObjects(broken)).toBe(true)
    expect(() => writeShowObjects(true, broken)).not.toThrow()
    expect(readShowObjects(undefined)).toBe(true)
  })
})

describe('visual job polling', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  const jobs = (...list: Array<VisualJob | Error>) => {
    const fn = vi.fn<(signal?: AbortSignal) => Promise<VisualJob>>()
    list.forEach((j) => (j instanceof Error ? fn.mockRejectedValueOnce(j) : fn.mockResolvedValueOnce(j)))
    return fn
  }

  it('reports each stage, then resolves when the job completes', async () => {
    const getJob = jobs({ status: 'queued' }, { status: 'processing', stage: 'preparing' }, { status: 'processing', stage: 'detecting', progress: 30.4 }, { status: 'completed', progress: 100 })
    const updates: DetectionRun[] = []
    const done = followVisualJob(getJob, { prepared: true, intervalMs: 10, onUpdate: (r) => updates.push(r) })
    await vi.runAllTimersAsync()
    await expect(done).resolves.toBeUndefined()
    expect(updates).toEqual([
      { phase: 'queued', prepared: true },
      { phase: 'preparing', prepared: true },
      { phase: 'detecting', prepared: true, progress: 30 },
    ])
  })

  it('maps stage and progress defensively', () => {
    expect(toVisualJobStatus({ status: 'processing', stage: 'warming', progress: 140 })).toEqual({ state: 'processing', progress: 100 })
    expect(toVisualJobStatus({ status: 'failed', error: 'no frames' })).toEqual({ state: 'failed', error: 'no frames' })
  })

  it('throws the job\'s own error when it fails', async () => {
    const done = followVisualJob(jobs({ status: 'failed', error: 'decoder crashed' }), { prepared: false, intervalMs: 10 })
    const result = expect(done).rejects.toThrow('decoder crashed')
    await vi.runAllTimersAsync()
    await result
    expect(runFromError(new Error('decoder crashed'))).toEqual({ phase: 'failed', error: 'decoder crashed' })
  })

  it('gives up after repeated poll failures', async () => {
    const getJob = jobs(...Array.from({ length: MAX_POLL_FAILURES }, () => new Error('Backend returned 500')))
    const done = followVisualJob(getJob, { prepared: false, intervalMs: 10 })
    const result = expect(done).rejects.toThrow('Backend returned 500')
    await vi.runAllTimersAsync()
    await result
  })

  it('stops on abort, which reads as cancelled rather than failed', async () => {
    const controller = new AbortController()
    const getJob = vi.fn<(signal?: AbortSignal) => Promise<VisualJob>>().mockResolvedValue({ status: 'processing' })
    const done = followVisualJob(getJob, { prepared: false, intervalMs: 1000, signal: controller.signal })
    const caught = done.catch((e: unknown) => e)
    await vi.advanceTimersByTimeAsync(10)
    controller.abort()
    const err = await caught
    expect(runFromError(err)).toEqual({ phase: 'cancelled' })
    const calls = getJob.mock.calls.length
    await vi.advanceTimersByTimeAsync(5000)
    expect(getJob.mock.calls.length).toBe(calls)
  })

  it('maps 503, 409 and 422 from starting a run', () => {
    expect(runFromError(new Error('Backend returned 503: ultralytics is not installed'))).toEqual({ phase: 'unavailable', reason: 'ultralytics is not installed' })
    expect(runFromError(new Error('Backend returned 503'))).toEqual({ phase: 'unavailable', reason: 'the server has no object detector' })
    expect(runFromError(new Error('Backend returned 409: A job of this type is already active'))).toEqual({ phase: 'failed', status: 409, error: 'A job of this type is already active' })
    expect(runFromError(new Error('Backend returned 409'))).toMatchObject({ error: 'Another detection is already running for this project' })
    expect(runFromError(new Error('Backend returned 422: Variable frame rate'))).toEqual({ phase: 'failed', status: 422, error: 'this video cannot be analysed (Variable frame rate)' })
    expect(runFromError(new Error('Backend returned 500'))).toEqual({ phase: 'failed', status: 500, error: 'the server returned HTTP 500' })
  })
})

describe('time guidance capabilities', () => {
  it('preserves explicitly unsupported anchors and leaves older services unspecified', () => {
    expect(toCapabilities({span_anchors: false}).spanAnchors).toBe(false)
    expect(toCapabilities({span_anchors: true}).spanAnchors).toBe(true)
    expect(toCapabilities({}).spanAnchors).toBeUndefined()
  })
})
