import { clamp, MAX_GAIN } from '../edits'
import type { Keyframe } from '../types'

// Loudness analysis for intensity-driven automation. Everything here measures
// signal LEVEL. It is a proxy for how heated a scene is, not emotion detection.

export const DEFAULT_WINDOW_SEC = 0.1
/** Smoothing radius in hops: 5 hops either side is about one second at the default window. */
export const DEFAULT_SMOOTH_RADIUS = 5
export const DEFAULT_LOW_THRESHOLD = 0.35
export const DEFAULT_HIGH_THRESHOLD = 0.8
export const DEFAULT_MAX_KEYFRAMES = 40

/** Round so keyframes stay readable in the edit log and in exported JSON. */
const round = (v: number) => Math.round(v * 1e4) / 1e4

/** RMS level per non-overlapping window. A trailing partial window is included. */
export function rmsEnvelope(
  samples: Float32Array,
  sampleRate: number,
  windowSec = DEFAULT_WINDOW_SEC,
): Float32Array {
  const size = Math.max(1, Math.round(sampleRate * windowSec))
  const out = new Float32Array(Math.ceil(samples.length / size))
  for (let w = 0; w < out.length; w++) {
    const start = w * size
    const end = Math.min(samples.length, start + size)
    let sum = 0
    for (let i = start; i < end; i++) sum += samples[i] * samples[i]
    out[w] = Math.sqrt(sum / (end - start))
  }
  return out
}

/** Centred moving average. The window shrinks at the edges instead of padding with zeros. */
export function smoothEnvelope(env: Float32Array, radius = DEFAULT_SMOOTH_RADIUS): Float32Array {
  const out = new Float32Array(env.length)
  for (let i = 0; i < env.length; i++) {
    const start = Math.max(0, i - radius)
    const end = Math.min(env.length - 1, i + radius)
    let sum = 0
    for (let j = start; j <= end; j++) sum += env[j]
    out[i] = sum / (end - start + 1)
  }
  return out
}

/**
 * Scale by the 95th percentile and clamp to [0, 1], so a single spike does not
 * flatten everything else. Falls back to the peak if the percentile is zero
 * (mostly silent input), and stays all zeros for pure silence.
 */
export function normalizeEnvelope(env: Float32Array): Float32Array {
  if (env.length === 0) return new Float32Array(0)
  const sorted = Float32Array.from(env).sort()
  const p95 = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)]
  const scale = p95 > 0 ? p95 : sorted[sorted.length - 1]
  if (scale <= 0) return new Float32Array(env.length)
  return env.map((v) => clamp(v / scale, 0, 1))
}

/** Combine per-stem envelopes. Shorter ones are treated as silent past their end. */
export function combineEnvelopes(envs: Float32Array[], mode: 'max' = 'max'): Float32Array {
  const length = envs.reduce((n, e) => Math.max(n, e.length), 0)
  const out = new Float32Array(length)
  if (mode === 'max') {
    for (const env of envs) {
      for (let i = 0; i < env.length; i++) out[i] = Math.max(out[i], env[i])
    }
  }
  return out
}

function smoothstep(a: number, b: number, x: number): number {
  const t = clamp((x - a) / (b - a), 0, 1)
  return t * t * (3 - 2 * t)
}

/** Keep the first and last point and any point further than epsilon (vertically) from the chord. */
function rdp(points: Keyframe[], epsilon: number): Keyframe[] {
  const keep = new Uint8Array(points.length)
  keep[0] = 1
  keep[points.length - 1] = 1
  const stack: Array<[number, number]> = [[0, points.length - 1]]
  while (stack.length > 0) {
    const [lo, hi] = stack.pop()!
    const a = points[lo]
    const b = points[hi]
    let worst = -1
    let worstDist = epsilon
    for (let i = lo + 1; i < hi; i++) {
      const p = points[i]
      const onChord = a.value + ((b.value - a.value) * (p.t - a.t)) / (b.t - a.t)
      const dist = Math.abs(p.value - onChord)
      if (dist > worstDist) {
        worst = i
        worstDist = dist
      }
    }
    if (worst >= 0) {
      keep[worst] = 1
      stack.push([lo, worst], [worst, hi])
    }
  }
  return points.filter((_, i) => keep[i] === 1)
}

export interface IntensityOptions {
  /** Gain when the envelope is at or below lowThreshold. */
  minValue?: number
  /** Gain when the envelope is at or above highThreshold. May be below minValue to duck instead. */
  maxValue?: number
  lowThreshold?: number
  highThreshold?: number
  maxKeyframes?: number
  /** Timeline length. Defaults to the envelope length. */
  duration?: number
}

/**
 * Turn a normalized envelope into gain keyframes: strictly increasing t, first
 * point at 0, last point at the duration, at most maxKeyframes points.
 */
export function intensityToKeyframes(env: Float32Array, hopSec: number, opts: IntensityOptions = {}): Keyframe[] {
  const {
    minValue = 1,
    maxValue = MAX_GAIN,
    lowThreshold = DEFAULT_LOW_THRESHOLD,
    highThreshold = DEFAULT_HIGH_THRESHOLD,
    maxKeyframes = DEFAULT_MAX_KEYFRAMES,
    duration = env.length * hopSec,
  } = opts
  const toGain = (v: number) => minValue + (maxValue - minValue) * smoothstep(lowThreshold, highThreshold, v)

  if (env.length === 0 || duration <= 0) return [{ t: 0, value: round(minValue) }]

  // Each envelope value describes a window, so it sits at the window's centre.
  const points: Keyframe[] = [{ t: 0, value: toGain(env[0]) }]
  for (let i = 0; i < env.length; i++) {
    const t = (i + 0.5) * hopSec
    if (t < duration) points.push({ t, value: toGain(env[i]) })
  }
  points.push({ t: duration, value: toGain(env[env.length - 1]) })

  let simplified = rdp(points, 0.005)
  for (let eps = 0.01; simplified.length > Math.max(2, maxKeyframes); eps *= 1.5) {
    simplified = rdp(points, eps)
  }
  return simplified.map((k) => ({ t: k.t === duration ? duration : round(k.t), value: round(k.value) }))
}
