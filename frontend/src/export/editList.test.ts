import { describe, expect, it } from 'vitest'
import type { TrackState } from '../types'
import { parseEditList, serializeEditList } from './editList'

const TRACKS: TrackState[] = [
  { stemId: 'a', gain: 1.25, muted: true, solo: false, automation: [] },
  {
    stemId: 'b',
    gain: 0.5,
    muted: false,
    solo: true,
    automation: [
      { t: 0, value: 1 },
      { t: 12.5, value: 0.25 },
    ],
  },
]

describe('serializeEditList', () => {
  it('writes version 1 JSON', () => {
    expect(JSON.parse(serializeEditList(TRACKS))).toEqual({ version: 1, tracks: TRACKS })
  })
})

describe('parseEditList', () => {
  it('round-trips from text and from an object', () => {
    const text = serializeEditList(TRACKS)
    expect(parseEditList(text)).toEqual({ ok: true, editList: { version: 1, tracks: TRACKS } })
    expect(parseEditList(JSON.parse(text))).toEqual({ ok: true, editList: { version: 1, tracks: TRACKS } })
  })

  it('round-trips no tracks', () => {
    expect(parseEditList(serializeEditList([]))).toEqual({ ok: true, editList: { version: 1, tracks: [] } })
  })

  it('rejects bad JSON and wrong top-level shapes', () => {
    for (const v of ['{oops', null, 5, [], {}, { version: 2, tracks: [] }, { version: 1 }]) {
      expect(parseEditList(v).ok).toBe(false)
    }
  })

  const track = (over: Record<string, unknown>) => ({ version: 1, tracks: [{ ...TRACKS[0], ...over }] })

  it.each([
    ['empty stemId', { stemId: '' }],
    ['numeric stemId', { stemId: 1 }],
    ['gain too high', { gain: 6 }],
    ['negative gain', { gain: -1 }],
    ['non-finite gain', { gain: Infinity }],
    ['string muted', { muted: 'no' }],
    ['numeric solo', { solo: 0 }],
    ['non-array automation', { automation: {} }],
    ['negative keyframe time', { automation: [{ t: -1, value: 1 }] }],
    ['keyframe value too high', { automation: [{ t: 1, value: 9 }] }],
    ['null keyframe', { automation: [null] }],
  ])('rejects %s', (_, over) => {
    expect(parseEditList(track(over)).ok).toBe(false)
  })

  it('sorts keyframes, since envelopeAt assumes sorted input', () => {
    const res = parseEditList(track({ automation: [{ t: 10, value: 0 }, { t: 0, value: 1 }] }))
    expect(res.ok && res.editList.tracks[0].automation.map((k) => k.t)).toEqual([0, 10])
  })

  it('drops unknown fields', () => {
    const res = parseEditList({ version: 1, tracks: [{ ...TRACKS[0], extra: 1 }], other: 2 })
    expect(res).toEqual({ ok: true, editList: { version: 1, tracks: [TRACKS[0]] } })
  })
})
