import { describe, expect, it } from 'vitest'
import type { TrackState } from '../types'
import { activity, boostersFor, roomAt } from './room'

const RATE = 1000 // 10 samples per 10 ms hop
const track = (stemId: string, gain = 1, extra: Partial<TrackState> = {}): TrackState => ({ stemId, gain, muted: false, solo: false, automation: [], ...extra })

/** One second of audio with a tone between `from` and `to` seconds. */
function burst(from: number, to: number): Float32Array {
  const x = new Float32Array(RATE)
  for (let i = Math.round(from * RATE); i < Math.round(to * RATE); i++) x[i] = 0.3 * Math.sin(i)
  return x
}

describe('activity', () => {
  it('is zero for silence', () => {
    expect(Math.max(...activity([new Float32Array(RATE)], RATE))).toBe(0)
  })

  it('rises in 20 ms, holds 250 ms after the sound, then lets go over 250 ms', () => {
    const a = activity([burst(0.2, 0.4)], RATE)
    expect(a[10]).toBe(0)
    expect(a[20]).toBe(0.5)
    expect(a[21]).toBe(1)
    expect(a[64]).toBe(1) // held
    expect(a[77]).toBeCloseTo(0.48)
    expect(a[89]).toBe(0)
  })

  it("ignores faint bleed far under the layer's loud parts", () => {
    const x = burst(0.1, 0.3)
    for (let i = 600; i < 900; i++) x[i] = 0.003 * Math.sin(i) // about 40 dB down
    expect(Math.max(...activity([x], RATE).slice(85))).toBe(0)
  })
})

describe('roomAt', () => {
  const a = activity([burst(0.2, 0.4)], RATE)
  const source = () => ({ start: 0, activity: a })
  it('steps back by the boosted level while the booster sounds, and not otherwise', () => {
    const boosters = boostersFor([track('mclovin', 5), track('rain')], 'rain')
    expect(roomAt(boosters, source, 0.3)).toBeCloseTo(0.2)
    expect(roomAt(boosters, source, 0.1)).toBe(1)
    expect(roomAt(boosters, source, 0.9)).toBe(1)
  })
  it('only boosts above 100% count, and solo or mute silence a booster', () => {
    expect(boostersFor([track('a', 1), track('b')], 'b')).toEqual([])
    expect(boostersFor([track('a', 3, { muted: true }), track('b')], 'b')).toEqual([])
    expect(boostersFor([track('a', 3), track('b', 1, { solo: true }), track('c')], 'c')).toEqual([])
  })
  it("follows the stem's offset", () => {
    const boosters = [track('a', 3)]
    expect(roomAt(boosters, () => ({ start: 1, activity: a }), 1.3)).toBeCloseTo(1 / 3)
    expect(roomAt(boosters, () => ({ start: 1, activity: a }), 0.3)).toBe(1)
  })
})
