import { describe, expect, it } from 'vitest'
import { applyAction, automationFrom, envelopeAt, initialTracks, isAudible, MAX_GAIN, parsePrompt } from './edits'
import type { Stem, TrackState } from './types'

const stem = (id: string, label: string, kind: Stem['kind'] = 'sfx'): Stem => ({
  id,
  label,
  kind,
  url: `blob:${id}`,
  duration: 20,
})

const STEMS = [stem('a', 'Speaker A', 'dialogue'), stem('b', 'Speaker B', 'dialogue'), stem('s', 'Siren')]
const TRACKS = initialTracks(STEMS)

const track = (over: Partial<TrackState> = {}): TrackState => ({
  stemId: 'x',
  gain: 1,
  muted: false,
  solo: false,
  automation: [],
  ...over,
})

describe('envelopeAt', () => {
  const kfs = [
    { t: 2, value: 0.5 },
    { t: 4, value: 1.5 },
  ]

  it('is unity with no keyframes', () => {
    expect(envelopeAt([], 3)).toBe(1)
  })

  it('holds the first value before the first keyframe', () => {
    expect(envelopeAt(kfs, 0)).toBe(0.5)
    expect(envelopeAt(kfs, 2)).toBe(0.5)
  })

  it('holds the last value after the last keyframe', () => {
    expect(envelopeAt(kfs, 4)).toBe(1.5)
    expect(envelopeAt(kfs, 99)).toBe(1.5)
  })

  it('interpolates linearly between keyframes', () => {
    expect(envelopeAt(kfs, 3)).toBeCloseTo(1)
    expect(envelopeAt(kfs, 2.5)).toBeCloseTo(0.75)
  })

  it('does not divide by zero on equal-time keyframes', () => {
    const step = [
      { t: 1, value: 0 },
      { t: 2, value: 1 },
      { t: 2, value: 2 },
      { t: 3, value: 2 },
    ]
    expect(envelopeAt(step, 2)).toBe(1)
    expect(envelopeAt(step, 2.5)).toBe(2)
    expect(Number.isFinite(envelopeAt(step, 2))).toBe(true)
  })
})

describe('isAudible', () => {
  it('is audible by default', () => {
    expect(isAudible([track()], track())).toBe(true)
  })

  it('a muted track is silent', () => {
    const t = track({ muted: true })
    expect(isAudible([t], t)).toBe(false)
  })

  it('solo silences the other tracks', () => {
    const a = track({ stemId: 'a', solo: true })
    const b = track({ stemId: 'b' })
    expect(isAudible([a, b], a)).toBe(true)
    expect(isAudible([a, b], b)).toBe(false)
  })

  it('two solos are both audible', () => {
    const a = track({ stemId: 'a', solo: true })
    const b = track({ stemId: 'b', solo: true })
    expect(isAudible([a, b], a)).toBe(true)
    expect(isAudible([a, b], b)).toBe(true)
  })

  it('mute beats solo on the same track', () => {
    const a = track({ stemId: 'a', solo: true, muted: true })
    const b = track({ stemId: 'b' })
    expect(isAudible([a, b], a)).toBe(false)
    expect(isAudible([a, b], b)).toBe(false)
  })
})

describe('applyAction', () => {
  const base = initialTracks([stem('a', 'A'), stem('b', 'B')])

  it('set_gain changes only the target track', () => {
    const next = applyAction(base, { type: 'set_gain', stemId: 'a', value: 1.5 })
    expect(next[0].gain).toBe(1.5)
    expect(next[1]).toBe(base[1])
  })

  it('set_gain clamps to [0, MAX_GAIN]', () => {
    expect(applyAction(base, { type: 'set_gain', stemId: 'a', value: 9 })[0].gain).toBe(MAX_GAIN)
    expect(applyAction(base, { type: 'set_gain', stemId: 'a', value: -1 })[0].gain).toBe(0)
  })

  it('set_mute and set_solo', () => {
    expect(applyAction(base, { type: 'set_mute', stemId: 'a', muted: true })[0].muted).toBe(true)
    expect(applyAction(base, { type: 'set_solo', stemId: 'b', solo: true })[1].solo).toBe(true)
  })

  it('set_automation sorts keyframes', () => {
    const next = applyAction(base, {
      type: 'set_automation',
      stemId: 'a',
      keyframes: [
        { t: 3, value: 1 },
        { t: 1, value: 0 },
      ],
    })
    expect(next[0].automation.map((k) => k.t)).toEqual([1, 3])
  })

  it('clear_automation empties the lane', () => {
    const withKf = applyAction(base, { type: 'set_automation', stemId: 'a', keyframes: [{ t: 1, value: 1 }] })
    expect(applyAction(withKf, { type: 'clear_automation', stemId: 'a' })[0].automation).toEqual([])
  })

  it('leaves tracks untouched for an unknown stemId', () => {
    const next = applyAction(base, { type: 'set_gain', stemId: 'nope', value: 0 })
    expect(next).toEqual(base)
  })

  it('does not mutate its input', () => {
    const snapshot = structuredClone(base)
    applyAction(base, { type: 'set_gain', stemId: 'a', value: 0 })
    expect(base).toEqual(snapshot)
  })
})

describe('parsePrompt', () => {
  it('louder with a range makes a four point ramp', () => {
    const res = parsePrompt('make the siren louder from 8 to 16 seconds', STEMS, TRACKS)
    expect(res.actions).toEqual([
      {
        type: 'set_automation',
        stemId: 's',
        keyframes: [
          { t: 8, value: 1 },
          { t: 8.5, value: 2 },
          { t: 15.5, value: 2 },
          { t: 16, value: 1 },
        ],
      },
    ])
  })

  it('quieter with a range uses 0.3', () => {
    const res = parsePrompt('make the siren quieter from 8 to 16', STEMS, TRACKS)
    const action = res.actions[0]
    expect(action.type === 'set_automation' && action.keyframes[1].value).toBe(0.3)
  })

  it('louder with no range moves the fader by 1.5x', () => {
    const res = parsePrompt('make the siren louder', STEMS, TRACKS)
    expect(res.actions).toEqual([{ type: 'set_gain', stemId: 's', value: 1.5 }])
  })

  it('quieter with no range halves the fader', () => {
    const res = parsePrompt('make the siren quieter', STEMS, TRACKS)
    expect(res.actions).toEqual([{ type: 'set_gain', stemId: 's', value: 0.5 }])
  })

  it('louder clamps the fader to MAX_GAIN', () => {
    const loud = TRACKS.map((t) => ({ ...t, gain: MAX_GAIN }))
    const res = parsePrompt('make the siren louder', STEMS, loud)
    expect(res.actions).toEqual([{ type: 'set_gain', stemId: 's', value: MAX_GAIN }])
  })

  it('mute with "at N" goes silent until the end', () => {
    const res = parsePrompt('Mute Speaker B at 12 seconds', STEMS, TRACKS)
    expect(res.actions).toEqual([
      {
        type: 'set_automation',
        stemId: 'b',
        keyframes: [
          { t: 11.95, value: 1 },
          { t: 12, value: 0 },
          { t: 20, value: 0 },
          { t: 20, value: 1 },
        ],
      },
    ])
  })

  it('plain mute and unmute use the mute flag', () => {
    expect(parsePrompt('mute the siren', STEMS, TRACKS).actions).toEqual([
      { type: 'set_mute', stemId: 's', muted: true },
    ])
    expect(parsePrompt('unmute the siren', STEMS, TRACKS).actions).toEqual([
      { type: 'set_mute', stemId: 's', muted: false },
    ])
  })

  it('solo and unsolo', () => {
    expect(parsePrompt('Solo Speaker A', STEMS, TRACKS).actions).toEqual([
      { type: 'set_solo', stemId: 'a', solo: true },
    ])
    expect(parsePrompt('unsolo Speaker A', STEMS, TRACKS).actions).toEqual([
      { type: 'set_solo', stemId: 'a', solo: false },
    ])
  })

  it('reports no stem match', () => {
    const res = parsePrompt('make the trombone louder', STEMS, TRACKS)
    expect(res.actions).toEqual([])
    expect(res.message).toContain('Speaker A')
  })

  it('rejects end <= start for louder', () => {
    const res = parsePrompt('make the siren louder from 16 to 8', STEMS, TRACKS)
    expect(res.actions).toEqual([])
    expect(res.message).toMatch(/after start/)
  })

  it('rejects end <= start for mute', () => {
    const res = parsePrompt('mute the siren from 16 to 8', STEMS, TRACKS)
    expect(res.actions).toEqual([])
    expect(res.message).toMatch(/after start/)
  })

  it('rejects mute at or past the end of the stem', () => {
    expect(parsePrompt('mute the siren at 25', STEMS, TRACKS).actions).toEqual([])
  })

  it('explains itself when no verb matches', () => {
    const res = parsePrompt('do something to the siren', STEMS, TRACKS)
    expect(res.actions).toEqual([])
    expect(res.message).toMatch(/Stub parser/)
  })
})

describe('applyAction follow_intensity', () => {
  it('is a no-op until resolved', () => {
    const tracks = initialTracks(STEMS)
    const next = applyAction(tracks, { type: 'follow_intensity', stemId: 's', sourceStemIds: ['a'] })
    expect(next).toEqual(tracks)
  })
})

describe('parsePrompt intensity', () => {
  const tracks = initialTracks(STEMS)
  const run = (text: string, stems = STEMS) => parsePrompt(text, stems, tracks)

  it('follows dialogue loudness for the demo prompt', () => {
    const res = run('make the siren louder whenever the argument gets intense')
    expect(res.actions).toEqual([{ type: 'follow_intensity', stemId: 's', sourceStemIds: ['a', 'b'] }])
    expect(res.message).toContain('dialogue loudness')
    expect(res.message).toContain('not emotion detection')
  })

  it.each([
    'boost the siren when things get heated',
    'raise the siren as the argument builds',
    'siren louder while they are angry',
    'make the siren louder when they get louder',
  ])('recognises "%s"', (text) => {
    expect(run(text).actions[0].type).toBe('follow_intensity')
  })

  it('ducks for quieter', () => {
    const res = run('make the siren quieter when the argument gets heated up')
    expect(res.actions[0]).toMatchObject({ type: 'follow_intensity', minValue: 1, maxValue: 0.3 })
    expect(res.message).toContain('dialogue loudness')
  })

  it('reads the edit from a main clause after the condition', () => {
    const res = run('when they get louder, make the siren quieter')
    expect(res.actions[0]).toMatchObject({ stemId: 's', maxValue: 0.3 })
  })

  it('targets the stem in the main clause, not one named in the condition', () => {
    const res = run('make the siren louder when speaker a gets angry')
    expect(res.actions[0].stemId).toBe('s')
  })

  it('says so when there is no dialogue to follow', () => {
    const noDialogue = STEMS.filter((s) => s.kind !== 'dialogue')
    const res = parsePrompt('siren louder when it gets intense', noDialogue, initialTracks(noDialogue))
    expect(res.actions).toEqual([])
    expect(res.message).toContain('No dialogue stems')
  })

  it('keeps explicit ranges as range edits', () => {
    const res = run('make the siren louder from 8 to 16 seconds when it gets intense')
    expect(res.actions[0].type).toBe('set_automation')
  })

  it('does not treat a plain louder as intensity', () => {
    expect(run('make the siren louder').actions[0].type).toBe('set_gain')
  })
})

describe('automationFrom', () => {
  const step = [
    { t: 0, value: 1 },
    { t: 0, value: 0 },
    { t: 5, value: 0 },
    { t: 5.05, value: 1 },
  ]

  it('starts after an equal-time step at the start position', () => {
    expect(automationFrom(step, 0)).toEqual({
      start: 0,
      ramps: [
        { t: 5, value: 0 },
        { t: 5.05, value: 1 },
      ],
    })
  })

  it('interpolates when starting between keyframes', () => {
    expect(automationFrom(step, 5.025).start).toBeCloseTo(0.5)
    expect(automationFrom(step, 5.025).ramps).toEqual([{ t: 5.05, value: 1 }])
  })

  it('holds at unity without automation', () => {
    expect(automationFrom([], 3)).toEqual({ start: 1, ramps: [] })
  })

  it('keeps every later keyframe, including later steps', () => {
    const later = [
      { t: 2, value: 1 },
      { t: 2, value: 0 },
    ]
    expect(automationFrom(later, 0)).toEqual({ start: 1, ramps: later })
  })
})

describe('parsePrompt duck', () => {
  const music = stem('m', 'Music')
  const stems = [...STEMS, music]
  const tracks = initialTracks(stems)
  const run = (text: string) => parsePrompt(text, stems, tracks).actions[0]

  it.each([
    'when they get louder, duck the music',
    'duck the music when they get louder',
    'duck the music whenever the argument gets intense',
  ])('ducks for "%s"', (text) => {
    expect(run(text)).toMatchObject({ type: 'follow_intensity', stemId: 'm', minValue: 1, maxValue: 0.3 })
  })

  it('lowers the fader for a plain duck', () => {
    expect(run('duck the music')).toEqual({ type: 'set_gain', stemId: 'm', value: 0.5 })
  })
})
