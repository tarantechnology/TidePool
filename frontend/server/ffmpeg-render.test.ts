import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { applyAction, envelopeAt, initialTracks, parsePrompt } from '../src/edits'
import { buildFilterGraph } from '../src/export/ffmpeg'
import type { Keyframe, MuteRange, Stem, TrackState } from '../src/types'

// Renders through the real ffmpeg, so the preview math is compared against the
// real filter. Lives in server/ because it needs Node types, which the browser
// tsconfig does not load. Skipped where ffmpeg is not installed.
const hasFfmpeg = spawnSync('ffmpeg', ['-version']).status === 0
const SR = 48000
const DUR = 6

const dir = mkdtempSync(join(tmpdir(), 'soundcanvas-ffmpeg-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

function sine(file: string, freq: number): void {
  const src = `sine=frequency=${freq}:sample_rate=${SR}:duration=${DUR}`
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', src, join(dir, file)])
}

function render(tracks: TrackState[], stems: Stem[]): Float32Array {
  const graph = buildFilterGraph({ version: 1, tracks }, stems)
  const inputs = graph.inputs.flatMap((s) => ['-i', join(dir, s.url)])
  const out = join(dir, 'out.f32')
  execFileSync('ffmpeg', [
    '-v', 'error', '-y', ...inputs,
    '-filter_complex', graph.filter,
    '-map', graph.output, '-ac', '1', '-f', 'f32le', out,
  ])
  const bytes = readFileSync(out)
  return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.length / 4))
}

function rmsAround(data: Float32Array, t: number, win = 0.05): number {
  const start = Math.floor((t - win / 2) * SR)
  const end = Math.floor((t + win / 2) * SR)
  let sum = 0
  for (let i = start; i < end; i++) sum += data[i] * data[i]
  return Math.sqrt(sum / (end - start))
}

const stem = (id: string, url: string): Stem => ({ id, label: id, kind: 'sfx', url, duration: DUR })
const track = (stemId: string, over: Partial<TrackState> = {}): TrackState => ({
  stemId,
  gain: 1,
  muted: false,
  solo: false,
  automation: [],
  ...over,
})

describe.skipIf(!hasFfmpeg)('filter graph rendered by ffmpeg', () => {
  const curve: Keyframe[] = [
    { t: 1, value: 0.25 },
    { t: 3, value: 1.5 },
    { t: 4, value: 0.5 },
  ]

  it('matches envelopeAt times fader gain at several timestamps', () => {
    sine('a.wav', 440)
    const stems = [stem('a', 'a.wav')]
    const base = rmsAround(render([track('a')], stems), 3)
    const out = render([track('a', { gain: 0.5, automation: curve })], stems)
    for (const t of [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]) {
      const expected = base * 0.5 * envelopeAt(curve, t)
      // One audio frame of gain granularity inside the 50ms window at most.
      expect(Math.abs(rmsAround(out, t) - expected)).toBeLessThan(0.03 * base + 0.002)
    }
  })

  it('holds a long dense curve without hitting expression limits', () => {
    sine('a.wav', 440)
    const dense = Array.from({ length: 64 }, (_, i) => ({ t: (i / 63) * DUR, value: 1 + Math.sin(i / 4) }))
    const stems = [stem('a', 'a.wav')]
    const base = rmsAround(render([track('a')], stems), 3)
    const out = render([track('a', { automation: dense })], stems)
    for (const t of [1, 2, 3, 4, 5]) {
      expect(Math.abs(rmsAround(out, t) - base * envelopeAt(dense, t))).toBeLessThan(0.05 * base + 0.002)
    }
  })

  it('renders curves far beyond the agent keyframe cap', () => {
    // Timeline edits have no cap. A plain a+b+c sum failed in ffmpeg at 99 keyframes.
    sine('a.wav', 440)
    const huge = Array.from({ length: 1000 }, (_, i) => ({ t: (i / 999) * DUR, value: 1 + Math.sin(i / 40) }))
    const stems = [stem('a', 'a.wav')]
    const base = rmsAround(render([track('a')], stems), 3)
    const out = render([track('a', { automation: huge })], stems)
    for (const t of [1, 3, 5]) {
      expect(Math.abs(rmsAround(out, t) - base * envelopeAt(huge, t))).toBeLessThan(0.05 * base + 0.002)
    }
  })

  it('stays silent after a final ramp to zero when the file runs longer', () => {
    sine('a.wav', 440)
    const shortStem = { ...stem('a', 'a.wav'), duration: 4 }
    const ramp = [
      { t: 0, value: 1 },
      { t: 4, value: 0 },
    ]
    const out = render([track('a', { automation: ramp })], [shortStem])
    expect(rmsAround(out, 5)).toBe(0)
  })

  it('follows a steep ramp without lagging behind it', () => {
    // Regression: at the default frame size the render lagged by about 40ms.
    sine('a.wav', 440)
    const ramp = [
      { t: 2, value: 2 },
      { t: 2.5, value: 0 },
    ]
    const stems = [stem('a', 'a.wav')]
    const base = rmsAround(render([track('a')], stems), 3)
    const out = render([track('a', { automation: ramp })], stems)
    for (const t of [2.1, 2.2, 2.3, 2.4]) {
      expect(rmsAround(out, t, 0.01) / base).toBeCloseTo(envelopeAt(ramp, t), 1)
    }
  })

  it('sums stems without normalizing, as the Web Audio graph does', () => {
    sine('a.wav', 440)
    sine('b.wav', 1000)
    const stems = [stem('a', 'a.wav'), stem('b', 'b.wav')]
    const single = rmsAround(render([track('a')], stems), 3)
    const mixed = rmsAround(render([track('a'), track('b')], stems), 3)
    // Two uncorrelated equal sines add in power: sqrt(2) times one, not one.
    expect(mixed / single).toBeCloseTo(Math.SQRT2, 1)
  })

  it('renders silence when everything is muted', () => {
    sine('a.wav', 440)
    const out = render([track('a', { muted: true })], [stem('a', 'a.wav')])
    expect(out.length).toBeGreaterThan(0)
    expect(rmsAround(out, 3)).toBe(0)
  })
})

/** The automation expression of app/media.py, copied by hand. That file is not run here. */
function mediaPyExpression(volume: number, points: Keyframe[]): string {
  let expr = String(points[points.length - 1].value)
  for (let j = points.length - 2; j >= 0; j--) {
    const a = points[j]
    const b = points[j + 1]
    expr = `if(lt(t,${b.t}),${a.value}+(${b.value}-${a.value})*(t-${a.t})/(${b.t}-${a.t}),${expr})`
  }
  return `(${volume})*if(lt(t,${points[0].t}),${points[0].value},${expr})`
}

function renderMediaPyShape(points: Keyframe[]): Float32Array {
  const out = join(dir, 'media.f32')
  execFileSync('ffmpeg', [
    '-v', 'error', '-y', '-i', join(dir, 'a.wav'),
    '-filter_complex', `[0:a]adelay=0:all=1,volume='${mediaPyExpression(1, points)}':eval=frame[a]`,
    '-map', '[a]', '-ac', '1', '-f', 'f32le', out,
  ])
  const bytes = readFileSync(out)
  return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.length / 4))
}

describe.skipIf(!hasFfmpeg)('timed edits on the stem boundaries, rendered by ffmpeg', () => {
  const stems = [stem('a', 'a.wav')]
  const curveFor = (prompt: string) =>
    parsePrompt(prompt, [{ ...stems[0], label: 'tone' }], initialTracks(stems))
      .actions.reduce(applyAction, initialTracks(stems))[0].automation

  const cases: Array<{ prompt: string; silent: number[]; loud: number[] }> = [
    { prompt: 'mute the tone from 0 to 2 seconds', silent: [0.1, 1, 1.8], loud: [2.2, 4, 5.8] },
    { prompt: 'mute the tone from 4 to 6 seconds', silent: [4.2, 5, 5.9], loud: [0.2, 2, 3.8] },
    { prompt: 'mute the tone at 3 seconds', silent: [3.2, 4.5, 5.9], loud: [0.2, 1.5, 2.8] },
    { prompt: 'mute the tone from 0 to 6 seconds', silent: [0.1, 3, 5.9], loud: [] },
  ]

  for (const { prompt, silent, loud } of cases) {
    it(`reference expression: ${prompt}`, () => {
      sine('a.wav', 440)
      const base = rmsAround(render([track('a')], stems), 3)
      const out = render([track('a', { automation: curveFor(prompt) })], stems)
      for (const t of silent) expect(rmsAround(out, t), `t=${t}`).toBeLessThan(0.001)
      for (const t of loud) expect(Math.abs(rmsAround(out, t) - base), `t=${t}`).toBeLessThan(0.03 * base)
    })

    it(`expression shaped like app/media.py: ${prompt}`, () => {
      sine('a.wav', 440)
      const base = rmsAround(render([track('a')], stems), 3)
      const out = renderMediaPyShape(curveFor(prompt))
      for (const t of silent) expect(rmsAround(out, t), `t=${t}`).toBeLessThan(0.001)
      for (const t of loud) expect(Math.abs(rmsAround(out, t) - base), `t=${t}`).toBeLessThan(0.03 * base)
    })
  }
})

describe.skipIf(!hasFfmpeg)('offsets and mute ranges, rendered by ffmpeg', () => {
  const late = (over: Partial<Stem>): Stem => ({ ...stem('a', 'a.wav'), duration: 3, ...over })

  function shortSine(): void {
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', `sine=frequency=440:sample_rate=${SR}:duration=3`, join(dir, 'a.wav')])
  }

  it('starts a stem at its offset', () => {
    shortSine()
    const out = render([track('a')], [late({ startOffset: 2 })])
    expect(rmsAround(out, 1)).toBe(0)
    expect(rmsAround(out, 1.9)).toBe(0)
    expect(rmsAround(out, 2.1)).toBeGreaterThan(0.01)
    expect(rmsAround(out, 4.9)).toBeGreaterThan(0.01)
    expect(out.length / SR).toBeCloseTo(5, 1)
  })

  it('applies automation in timeline time, not stem time', () => {
    shortSine()
    const curve: Keyframe[] = [
      { t: 2.9, value: 1 },
      { t: 3, value: 0 },
    ]
    const out = render([track('a', { automation: curve })], [late({ startOffset: 2 })])
    expect(rmsAround(out, 2.5)).toBeGreaterThan(0.01)
    expect(rmsAround(out, 4)).toBe(0)
  })

  it('silences a mute range in timeline time', () => {
    shortSine()
    const ranges: MuteRange[] = [[3, 4]]
    const out = render([track('a')], [late({ startOffset: 2, muteRanges: ranges })])
    expect(rmsAround(out, 2.5)).toBeGreaterThan(0.01)
    expect(rmsAround(out, 3.5)).toBe(0)
    expect(rmsAround(out, 4.5)).toBeGreaterThan(0.01)
  })

  it('combines a mute range with automation', () => {
    sine('a.wav', 440)
    const curve: Keyframe[] = [
      { t: 0, value: 0.5 },
      { t: 6, value: 0.5 },
    ]
    const stems = [stem('a', 'a.wav')]
    const base = rmsAround(render([track('a')], stems), 3)
    const out = render([track('a', { automation: curve })], [{ ...stems[0], muteRanges: [[2, 3]] }])
    expect(rmsAround(out, 2.5)).toBe(0)
    expect(rmsAround(out, 4)).toBeCloseTo(base * 0.5, 2)
  })

  it('matches the filter order of app/media.py on the same input', () => {
    shortSine()
    const reference = render([track('a')], [late({ startOffset: 2, muteRanges: [[3, 4]] })])
    const out = join(dir, 'media-offset.f32')
    const expr = "(1.0)*if(between(t,3.0,4.0),0,1)"
    execFileSync('ffmpeg', [
      '-v', 'error', '-y', '-i', join(dir, 'a.wav'),
      '-filter_complex', `[0:a]adelay=2000:all=1,volume='${expr}':eval=frame[a]`,
      '-map', '[a]', '-ac', '1', '-f', 'f32le', out,
    ])
    const bytes = readFileSync(out)
    const shaped = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.length / 4))
    for (const t of [1, 2.5, 3.5, 4.5]) {
      expect(Math.abs(rmsAround(shaped, t) - rmsAround(reference, t)), `t=${t}`).toBeLessThan(0.005)
    }
  })
})
