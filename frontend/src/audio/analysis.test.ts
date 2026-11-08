import { describe, expect, it } from 'vitest'
import { envelopeAt, MAX_GAIN } from '../edits'
import {
  combineEnvelopes,
  intensityToKeyframes,
  normalizeEnvelope,
  rmsEnvelope,
  smoothEnvelope,
} from './analysis'


const SR = 1000
const HOP = 0.1

/** Quiet 0 to 6s, rising 6 to 16s, loud after, as a noisy-looking alternating signal. */
function scene(duration = 20): Float32Array {
  const out = new Float32Array(duration * SR)
  for (let i = 0; i < out.length; i++) {
    const t = i / SR
    const amp = t < 6 ? 0.05 : t < 16 ? 0.05 + 0.75 * ((t - 6) / 10) : 0.8
    out[i] = (i % 2 === 0 ? 1 : -1) * amp
  }
  return out
}

function pipeline(samples: Float32Array, duration?: number) {
  const env = normalizeEnvelope(smoothEnvelope(rmsEnvelope(samples, SR, HOP)))
  return intensityToKeyframes(env, HOP, { duration })
}

describe('rmsEnvelope', () => {
  it('computes RMS per window', () => {
    const env = rmsEnvelope(new Float32Array(200).fill(0.5), SR, HOP)
    expect(env).toHaveLength(2)
    expect(env[0]).toBeCloseTo(0.5)
  })

  it('includes a trailing partial window without dividing by the full size', () => {
    const env = rmsEnvelope(new Float32Array(150).fill(0.5), SR, HOP)
    expect(env).toHaveLength(2)
    expect(env[1]).toBeCloseTo(0.5)
  })

  it('returns an empty envelope for no samples', () => {
    expect(rmsEnvelope(new Float32Array(0), SR)).toHaveLength(0)
  })
})

describe('smoothEnvelope', () => {
  it('averages neighbours and shrinks at the edges', () => {
    const env = smoothEnvelope(Float32Array.from([0, 3, 0]), 1)
    expect(Array.from(env)).toEqual([1.5, 1, 1.5])
  })

  it('leaves a constant signal unchanged', () => {
    expect(Array.from(smoothEnvelope(new Float32Array(9).fill(0.25), 3))).toEqual(new Array(9).fill(0.25))
  })
})

describe('normalizeEnvelope', () => {
  it('ignores a lone spike by scaling to the 95th percentile', () => {
    const env = new Float32Array(100).fill(0.5)
    env[10] = 100
    const out = normalizeEnvelope(env)
    expect(out[0]).toBeCloseTo(1)
    expect(out[10]).toBe(1)
  })

  it('falls back to the peak when the 95th percentile is zero', () => {
    const env = new Float32Array(100)
    env[3] = 0.4
    const out = normalizeEnvelope(env)
    expect(out[3]).toBeCloseTo(1)
    expect(out[4]).toBe(0)
  })

  it('stays all zeros for silence and handles empty input', () => {
    expect(Array.from(normalizeEnvelope(new Float32Array(4)))).toEqual([0, 0, 0, 0])
    expect(normalizeEnvelope(new Float32Array(0))).toHaveLength(0)
  })
})

describe('combineEnvelopes', () => {
  it('takes the max per hop and pads shorter envelopes as silence', () => {
    const out = combineEnvelopes([Float32Array.from([0.5, 0.25]), Float32Array.from([0.25, 0.5, 0.75])], 'max')
    expect(Array.from(out)).toEqual([0.5, 0.5, 0.75])
  })

  it('returns empty for no sources', () => {
    expect(combineEnvelopes([])).toHaveLength(0)
  })
})

describe('intensityToKeyframes', () => {
  const kfs = pipeline(scene())

  it('stays near minValue while quiet and reaches maxValue when loud', () => {
    expect(envelopeAt(kfs, 2)).toBeCloseTo(1, 1)
    expect(envelopeAt(kfs, 18)).toBeCloseTo(MAX_GAIN, 1)
  })

  it('rises through the middle of the ramp', () => {
    const mid = envelopeAt(kfs, 11)
    expect(mid).toBeGreaterThan(1.1)
    expect(mid).toBeLessThan(1.9)
  })

  it('has at most 40 keyframes, first at 0, last at the duration, strictly increasing', () => {
    expect(kfs.length).toBeLessThanOrEqual(40)
    expect(kfs[0].t).toBe(0)
    expect(kfs[kfs.length - 1].t).toBeCloseTo(20)
    for (let i = 1; i < kfs.length; i++) expect(kfs[i].t).toBeGreaterThan(kfs[i - 1].t)
  })

  it('is deterministic', () => {
    expect(pipeline(scene())).toEqual(kfs)
  })

  it('returns a flat minValue line for silence, not NaN', () => {
    const flat = pipeline(new Float32Array(20 * SR))
    expect(flat).toEqual([
      { t: 0, value: 1 },
      { t: 20, value: 1 },
    ])
  })

  it('honours a lower maxKeyframes on a jagged signal', () => {
    const env = Float32Array.from({ length: 400 }, (_, i) => (i % 4 < 2 ? 1 : 0))
    const out = intensityToKeyframes(env, HOP, { maxKeyframes: 10 })
    expect(out.length).toBeLessThanOrEqual(10)
    expect(out[out.length - 1].t).toBeCloseTo(40)
  })

  it('can duck when maxValue is below minValue', () => {
    const env = normalizeEnvelope(smoothEnvelope(rmsEnvelope(scene(), SR, HOP)))
    const duck = intensityToKeyframes(env, HOP, { minValue: 1, maxValue: 0.3 })
    expect(envelopeAt(duck, 2)).toBeCloseTo(1, 1)
    expect(envelopeAt(duck, 18)).toBeCloseTo(0.3, 1)
  })

  it('ends exactly at a timeline duration that differs from the envelope length', () => {
    const env = new Float32Array(10).fill(1)
    const shorter = intensityToKeyframes(env, HOP, { duration: 0.93 })
    expect(shorter[shorter.length - 1].t).toBe(0.93)
    const longer = intensityToKeyframes(env, HOP, { duration: 2 })
    expect(longer[longer.length - 1].t).toBe(2)
  })

  it('handles an empty envelope', () => {
    expect(intensityToKeyframes(new Float32Array(0), HOP)).toEqual([{ t: 0, value: 1 }])
  })
})
