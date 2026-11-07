import { describe, expect, it } from 'vitest'
import { envelopeAt } from '../edits'
import type { Keyframe, Stem, TrackState } from '../types'
import { automationSegments, buildFilterGraph, evaluateSegments, gainExpressionForTrack } from './ffmpeg'

const DUR = 20
const track = (over: Partial<TrackState> = {}): TrackState => ({
  stemId: 'a',
  gain: 1,
  muted: false,
  solo: false,
  automation: [],
  ...over,
})
const stem = (id: string): Stem => ({ id, label: id, kind: 'sfx', url: 'x', duration: DUR })

const CURVES: Array<[string, Keyframe[]]> = [
  ['none', []],
  ['single', [{ t: 7, value: 0.4 }]],
  ['single at 0', [{ t: 0, value: 1.5 }]],
  ['single at the end', [{ t: DUR, value: 1.5 }]],
  [
    'starts late, ends early',
    [
      { t: 3, value: 0.2 },
      { t: 9.5, value: 2 },
      { t: 15, value: 0 },
    ],
  ],
  [
    'covers the whole timeline',
    [
      { t: 0, value: 1 },
      { t: DUR, value: 0 },
    ],
  ],
  [
    'equal-time step',
    [
      { t: 5, value: 1 },
      { t: 5, value: 0 },
      { t: 10, value: 0 },
      { t: 10.05, value: 1 },
    ],
  ],
  [
    'dense',
    Array.from({ length: 40 }, (_, i) => ({ t: i * 0.5 + 0.13, value: 1 + Math.sin(i) })),
  ],
]

describe('evaluateSegments matches envelopeAt', () => {
  it.each(CURVES)('at 200 sample times: %s', (_, keyframes) => {
    const segments = automationSegments(keyframes, DUR)
    for (let i = 0; i < 200; i++) {
      const t = (i / 199) * DUR
      expect(evaluateSegments(segments, t)).toBeCloseTo(envelopeAt(keyframes, t), 9)
    }
  })

  it('agrees exactly on the keyframe times, where a step is decided', () => {
    for (const [, keyframes] of CURVES) {
      const segments = automationSegments(keyframes, DUR)
      for (const k of keyframes) {
        expect(evaluateSegments(segments, k.t)).toBeCloseTo(envelopeAt(keyframes, k.t), 9)
      }
    }
  })
})

/** Evaluate the emitted expression text, which is what ffmpeg actually sees. */
function evalExpression(expr: string, t: number): number {
  const js = expr.replaceAll('\\,', ',')
  const fn = new Function('t', 'lte', 'gt', `return ${js}`)
  return fn(
    t,
    (a: number, b: number) => (a <= b ? 1 : 0),
    (a: number, b: number) => (a > b ? 1 : 0),
  )
}

describe('emitted expression matches envelopeAt', () => {
  it.each(CURVES)('at 200 sample times: %s', (_, keyframes) => {
    const expr = gainExpressionForTrack(track({ automation: keyframes }), DUR)
    for (let i = 0; i < 200; i++) {
      const t = (i / 199) * DUR
      expect(evalExpression(expr, t)).toBeCloseTo(envelopeAt(keyframes, t), 3)
    }
  })

  it('holds the last value past the duration, never extrapolating a final ramp', () => {
    const ramp = [
      { t: 0, value: 1 },
      { t: 4, value: 0 },
    ]
    const expr = gainExpressionForTrack(track({ automation: ramp }), 4)
    expect(evalExpression(expr, 5)).toBe(0)
    expect(evalExpression(expr, 6)).toBe(0)
  })

  it('holds the first value before a curve that starts late', () => {
    const expr = gainExpressionForTrack(track({ automation: [{ t: 2, value: 0.5 }, { t: 3, value: 1 }] }), DUR)
    expect(evalExpression(expr, 0)).toBe(0.5)
  })
})

describe('automationSegments', () => {
  it('is empty without automation, so the track stays at unity', () => {
    expect(automationSegments([], DUR)).toEqual([])
    expect(evaluateSegments([], 3)).toBe(1)
  })

  it('adds flat holds around the curve', () => {
    const segs = automationSegments(
      [
        { t: 2, value: 1 },
        { t: 4, value: 0 },
      ],
      10,
    )
    expect(segs).toEqual([
      { t0: 0, t1: 2, v0: 1, v1: 1 },
      { t0: 2, t1: 4, v0: 1, v1: 0 },
      { t0: 4, t1: 10, v0: 0, v1: 0 },
    ])
  })

  it('stops at keyframes past the duration', () => {
    const segs = automationSegments(
      [
        { t: 0, value: 1 },
        { t: 5, value: 0 },
        { t: 30, value: 1 },
        { t: 40, value: 0 },
      ],
      10,
    )
    expect(segs.map((s) => s.t0)).toEqual([0, 5])
  })
})

describe('gainExpressionForTrack', () => {
  it('is 1 without automation', () => {
    expect(gainExpressionForTrack(track(), DUR)).toBe('1')
  })

  it('escapes every comma', () => {
    const expr = gainExpressionForTrack(track({ automation: CURVES[4][1] }), DUR)
    expect(expr).toContain('\\,')
    expect(expr.replaceAll('\\,', '')).not.toContain(',')
  })

  it('uses gated terms rather than nested if()', () => {
    const expr = gainExpressionForTrack(track({ automation: CURVES[7][1] }), DUR)
    expect(expr).not.toContain('if(')
    expect(expr).toContain('gt(t\\,')
  })

  it('never prints exponents, which would be ambiguous next to t', () => {
    const expr = gainExpressionForTrack(
      track({
        automation: [
          { t: 0, value: 1e-9 },
          { t: 20, value: 2e-9 },
        ],
      }),
      DUR,
    )
    expect(expr).not.toMatch(/\de-?\d/)
  })
})

describe('buildFilterGraph', () => {
  const stems = [stem('a'), stem('b'), stem('c')]

  it('applies fader, automation and mixes without normalizing', () => {
    const list = {
      version: 1 as const,
      tracks: [
        track({ stemId: 'a', gain: 0.5, automation: CURVES[5][1] }),
        track({ stemId: 'b', gain: 1.25 }),
      ],
    }
    const graph = buildFilterGraph(list, stems)
    expect(graph.inputs.map((s) => s.id)).toEqual(['a', 'b'])
    expect(graph.filter).toContain('[0:a]volume=0.5,asetnsamples=n=256,volume=')
    expect(graph.filter).toContain(':eval=frame[a0]')
    expect(graph.filter).toContain('[1:a]volume=1.25[a1]')
    expect(graph.filter).toContain('[a0][a1]amix=inputs=2:duration=longest:normalize=0[mix]')
  })

  it('leaves muted tracks out and renumbers inputs', () => {
    const list = {
      version: 1 as const,
      tracks: [track({ stemId: 'a', muted: true }), track({ stemId: 'b' }), track({ stemId: 'c' })],
    }
    const graph = buildFilterGraph(list, stems)
    expect(graph.inputs.map((s) => s.id)).toEqual(['b', 'c'])
    expect(graph.filter).toContain('[0:a]volume=1[a0]')
  })

  it('honours solo', () => {
    const list = {
      version: 1 as const,
      tracks: [track({ stemId: 'a' }), track({ stemId: 'b', solo: true })],
    }
    expect(buildFilterGraph(list, stems).inputs.map((s) => s.id)).toEqual(['b'])
  })

  it('skips tracks with no matching stem', () => {
    const list = { version: 1 as const, tracks: [track({ stemId: 'gone' }), track({ stemId: 'a' })] }
    expect(buildFilterGraph(list, stems).inputs.map((s) => s.id)).toEqual(['a'])
  })

  it('renders silence for the full length when nothing is audible', () => {
    const list = { version: 1 as const, tracks: [track({ stemId: 'a', muted: true })] }
    const graph = buildFilterGraph(list, stems)
    expect(graph.inputs).toEqual([])
    expect(graph.filter).toBe('anullsrc=r=48000:cl=stereo:d=20[mix]')
  })
})
