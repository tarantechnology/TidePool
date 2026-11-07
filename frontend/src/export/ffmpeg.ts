import { isAudible } from '../edits'
import type { EditList, Keyframe, Stem, TrackState } from '../types'

// Reference implementation only: Person 4 owns the real export. This exists so
// preview and export can be compared against the same math (see ffmpeg.test.ts
// and server/ffmpeg-render.test.ts). `amix normalize=0` sums inputs the way the
// Web Audio graph does. The preview's master gain is not applied here, matching
// the upstream exporter.

/** Linear gain from (t0, v0) to (t1, v1). Segments are contiguous and t0 < t1. */
export interface GainSegment {
  t0: number
  t1: number
  v0: number
  v1: number
}

/**
 * Piecewise-linear form of an automation curve over [0, duration], including
 * the flat holds before the first and after the last keyframe. Empty when the
 * curve is constant 1 (no automation).
 */
export function automationSegments(keyframes: Keyframe[], duration: number): GainSegment[] {
  if (keyframes.length === 0) return []
  const first = keyframes[0]
  const last = keyframes[keyframes.length - 1]
  const segments: GainSegment[] = []
  if (first.t > 0) segments.push({ t0: 0, t1: first.t, v0: first.value, v1: first.value })
  for (let i = 1; i < keyframes.length; i++) {
    const a = keyframes[i - 1]
    const b = keyframes[i]
    if (b.t > a.t && a.t < duration) segments.push({ t0: a.t, t1: b.t, v0: a.value, v1: b.value })
  }
  if (last.t < duration) segments.push({ t0: last.t, t1: duration, v0: last.value, v1: last.value })
  if (segments.length === 0) segments.push({ t0: 0, t1: Math.max(duration, 1), v0: last.value, v1: last.value })
  return segments
}

/** JS evaluator of the exact curve gainExpressionForTrack emits. */
export function evaluateSegments(segments: GainSegment[], t: number): number {
  if (segments.length === 0) return 1
  for (const s of segments) {
    if (t <= s.t1) return t <= s.t0 ? s.v0 : s.v0 + ((s.v1 - s.v0) * (t - s.t0)) / (s.t1 - s.t0)
  }
  return segments[segments.length - 1].v1
}

const num = (v: number) => String(Number(v.toFixed(6)))

function rampValue(s: GainSegment): string {
  if (s.v0 === s.v1) return num(s.v0)
  return `(${num(s.v0)}+(${num((s.v1 - s.v0) / (s.t1 - s.t0))})*(t-${num(s.t0)}))`
}

/**
 * ffmpeg rejects a long flat a+b+c (it failed at 99 keyframes in testing), so
 * terms are grouped in a balanced tree, which only nests log2(n) deep.
 */
function balancedSum(terms: string[]): string {
  if (terms.length <= 2) return terms.join('+')
  const mid = Math.floor(terms.length / 2)
  return `(${balancedSum(terms.slice(0, mid))})+(${balancedSum(terms.slice(mid))})`
}

/** FFmpeg `volume` expression for a track's automation, commas escaped for a filtergraph. */
export function gainExpressionForTrack(track: TrackState, duration: number): string {
  const segments = automationSegments(track.automation, duration)
  if (segments.length === 0) return '1'
  const first = segments[0]
  const last = segments[segments.length - 1]
  if (segments.every((s) => s.v0 === first.v0 && s.v1 === first.v0)) return num(first.v0)
  // Mutually exclusive gates, closed on the right like envelopeAt, with holds on
  // both sides so a stem that outlasts the timeline never extrapolates a ramp.
  const terms = [
    `lte(t,${num(first.t0)})*${num(first.v0)}`,
    ...segments.map((s) => `gt(t,${num(s.t0)})*lte(t,${num(s.t1)})*${rampValue(s)}`),
    `gt(t,${num(last.t1)})*${num(last.v1)}`,
  ]
  return balancedSum(terms).replaceAll(',', '\\,')
}

export interface FilterGraph {
  filter: string
  /** Stems to pass as ffmpeg -i inputs, in this order. Excludes inaudible tracks. */
  inputs: Stem[]
  /** Output pad of the mix. */
  output: string
}

const OUTPUT = '[mix]'
/**
 * `volume` evaluates its expression once per audio frame and holds it. Decoded
 * WAV frames are about 43ms at 48kHz, which lags steep curves audibly behind
 * the sample-accurate preview, so automated tracks get shorter frames first.
 */
const AUTOMATION_FRAME_SAMPLES = 256

/** Filter graph applying fader gain, mute/solo and automation per stem, then mixing. */
export function buildFilterGraph(editList: EditList, stems: Stem[]): FilterGraph {
  const duration = stems.reduce((d, s) => Math.max(d, s.duration), 0)
  const audible = editList.tracks.flatMap((track) => {
    const stem = stems.find((s) => s.id === track.stemId)
    return stem && isAudible(editList.tracks, track) ? [{ track, stem }] : []
  })
  if (audible.length === 0) {
    return { filter: `anullsrc=r=48000:cl=stereo:d=${num(duration)}${OUTPUT}`, inputs: [], output: OUTPUT }
  }
  const chains = audible.map(({ track }, i) => {
    const fader = `volume=${num(track.gain)}`
    const expr = gainExpressionForTrack(track, duration)
    const envelope = expr === '1' ? '' : `,asetnsamples=n=${AUTOMATION_FRAME_SAMPLES},volume=${expr}:eval=frame`
    return `[${i}:a]${fader}${envelope}[a${i}]`
  })
  const pads = audible.map((_, i) => `[a${i}]`).join('')
  const mix = `${pads}amix=inputs=${audible.length}:duration=longest:normalize=0${OUTPUT}`
  return { filter: [...chains, mix].join(';'), inputs: audible.map((a) => a.stem), output: OUTPUT }
}
