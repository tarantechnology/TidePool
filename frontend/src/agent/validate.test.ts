import { describe, expect, it } from 'vitest'
import { MAX_GAIN } from '../edits'
import type { Stem } from '../types'
import { MAX_ACTIONS, MAX_KEYFRAMES, validateActions } from './validate'

const stem = (id: string): Stem => ({ id, label: id, kind: 'sfx', url: 'blob:x', duration: 20 })
const STEMS = [stem('a'), stem('b')]
const check = (actions: unknown, duration = 20) => validateActions(actions, STEMS, duration)
const rejects = (action: unknown) => expect(check([action]).ok).toBe(false)
const accepts = (action: unknown) => expect(check([action]).ok).toBe(true)

describe('validateActions', () => {
  it('rejects a non-array', () => {
    for (const v of [undefined, null, 'x', 3, {}]) expect(check(v).ok).toBe(false)
  })

  it('accepts an empty list', () => {
    expect(check([])).toEqual({ ok: true, actions: [] })
  })

  it('caps the action count', () => {
    const one = { type: 'clear_automation', stemId: 'a' }
    expect(check(new Array(MAX_ACTIONS).fill(one)).ok).toBe(true)
    expect(check(new Array(MAX_ACTIONS + 1).fill(one)).ok).toBe(false)
  })

  it('rejects unknown types and non-object entries', () => {
    rejects({ type: 'delete_everything', stemId: 'a' })
    rejects({ stemId: 'a' })
    rejects(null)
    rejects('set_gain')
    rejects([])
  })

  it('rejects unknown or non-string stem ids', () => {
    rejects({ type: 'set_gain', stemId: 'nope', value: 1 })
    rejects({ type: 'set_gain', value: 1 })
    rejects({ type: 'set_gain', stemId: 7, value: 1 })
    rejects({ type: 'clear_automation', stemId: 'nope' })
  })

  it('checks set_gain range and numeric type', () => {
    accepts({ type: 'set_gain', stemId: 'a', value: 0 })
    accepts({ type: 'set_gain', stemId: 'a', value: MAX_GAIN })
    for (const value of [-0.1, MAX_GAIN + 0.1, NaN, Infinity, -Infinity, '1', null, undefined]) {
      rejects({ type: 'set_gain', stemId: 'a', value })
    }
  })

  it('checks booleans on mute and solo', () => {
    accepts({ type: 'set_mute', stemId: 'a', muted: true })
    accepts({ type: 'set_solo', stemId: 'a', solo: false })
    rejects({ type: 'set_mute', stemId: 'a', muted: 'true' })
    rejects({ type: 'set_solo', stemId: 'a', solo: 1 })
  })

  it('checks keyframe count', () => {
    const kf = { t: 1, value: 1 }
    rejects({ type: 'set_automation', stemId: 'a', keyframes: [] })
    rejects({ type: 'set_automation', stemId: 'a', keyframes: 'x' })
    rejects({ type: 'set_automation', stemId: 'a' })
    accepts({ type: 'set_automation', stemId: 'a', keyframes: new Array(MAX_KEYFRAMES).fill(kf) })
    rejects({ type: 'set_automation', stemId: 'a', keyframes: new Array(MAX_KEYFRAMES + 1).fill(kf) })
  })

  it('checks keyframe t and value', () => {
    const kfs = (keyframes: unknown[]) => ({ type: 'set_automation', stemId: 'a', keyframes })
    accepts(kfs([{ t: 0, value: 0 }, { t: 20, value: 2 }]))
    rejects(kfs([{ t: 20.01, value: 1 }]))
    rejects(kfs([{ t: -1, value: 1 }]))
    rejects(kfs([{ t: NaN, value: 1 }]))
    rejects(kfs([{ t: Infinity, value: 1 }]))
    rejects(kfs([{ t: 1, value: 3 }]))
    rejects(kfs([{ t: 1, value: NaN }]))
    rejects(kfs([{ t: '1', value: 1 }]))
    rejects(kfs([null]))
  })

  it('checks follow_intensity sources and bounds', () => {
    const base = { type: 'follow_intensity', stemId: 'a', sourceStemIds: ['b'] }
    accepts(base)
    accepts({ ...base, minValue: 1, maxValue: 0.3 })
    rejects({ ...base, sourceStemIds: ['nope'] })
    rejects({ ...base, sourceStemIds: [] })
    rejects({ ...base, sourceStemIds: 'b' })
    rejects({ ...base, sourceStemIds: [1] })
    rejects({ ...base, minValue: 5 })
    rejects({ ...base, maxValue: NaN })
  })

  it('drops unexpected fields from accepted actions', () => {
    const res = check([{ type: 'set_gain', stemId: 'a', value: 1, evil: true }])
    expect(res).toEqual({ ok: true, actions: [{ type: 'set_gain', stemId: 'a', value: 1 }] })
  })

  it('omits unset optional fields', () => {
    const res = check([{ type: 'follow_intensity', stemId: 'a', sourceStemIds: ['b'] }])
    expect(res).toEqual({ ok: true, actions: [{ type: 'follow_intensity', stemId: 'a', sourceStemIds: ['b'] }] })
  })

  it('reports every bad entry and rejects the whole batch', () => {
    const res = check([
      { type: 'set_gain', stemId: 'a', value: 1 },
      { type: 'set_gain', stemId: 'a', value: 9 },
      { type: 'nope', stemId: 'a' },
    ])
    expect(res.ok).toBe(false)
    if (!res.ok) expect(res.errors).toHaveLength(2)
  })
})
