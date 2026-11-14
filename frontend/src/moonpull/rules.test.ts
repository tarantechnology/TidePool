import { describe as group, expect, it } from 'vitest'
import { envelopeAt, MAX_KEYFRAMES } from '../edits'
import type { Stem } from '../types'
import {
  applyGainPull, cleanMoments, describe, echoGain, echoTimes, findPhrase, gainWindows, loudMoments, newRule, pickLayer,
  RECIPES, responseFor, ruleState, signature,
} from './rules'
import type { Rule } from './rules'

const stems: Stem[] = [
  { id: 'd', label: 'People talking', kind: 'dialogue', url: '', duration: 49 },
  { id: 'r', label: 'Heavy rain', kind: 'ambience', url: '', duration: 49 },
  { id: 't', label: 'Thunder rumbling', kind: 'sfx', url: '', duration: 49 },
  { id: 'x', label: 'residual', kind: 'residual', url: '', duration: 49 },
]

group('moments', () => {
  it('sorts, clips, drops slivers and joins overlaps', () => {
    expect(cleanMoments([{ start: 5, end: 6, note: 'b' }, { start: 4.5, end: 5.2, note: 'a' }, { start: 48, end: 60 }, { start: 10, end: 10.01 }, { start: 3, end: 2 }], 49))
      .toEqual([{ start: 2, end: 3 }, { start: 4.5, end: 6, note: 'a / b' }, { start: 48, end: 49 }])
  })

  it('widens swell windows by lead and tail and joins neighbours', () => {
    expect(gainWindows([{ start: 3, end: 4 }, { start: 5, end: 5.5 }], 'swell', 49)).toEqual([[2.7, 6.7]])
    expect(gainWindows([{ start: 0.1, end: 1 }], 'hush', 49)).toEqual([[0, 1.4]])
  })
})

group('swell and hush', () => {
  it('scales only inside the windows and leaves the rest of the curve alone', () => {
    const res = applyGainPull([], [{ start: 16, end: 24 }, { start: 38, end: 43 }], { kind: 'hush', amount: 0.6 }, 49)
    expect(res.ok).toBe(true)
    if (!res.ok) return
    const k = res.keyframes
    expect(envelopeAt(k, 10)).toBeCloseTo(1)
    expect(envelopeAt(k, 20)).toBeCloseTo(0.46)
    expect(envelopeAt(k, 30)).toBeCloseTo(1)
    expect(envelopeAt(k, 40)).toBeCloseTo(0.46)
    expect(envelopeAt(k, 48)).toBeCloseTo(1)
    expect(k.length).toBeLessThanOrEqual(MAX_KEYFRAMES)
  })

  it('keeps the shape of existing automation under a swell', () => {
    const curve = [{ t: 0, value: 0.5 }, { t: 49, value: 1 }]
    const res = applyGainPull(curve, [{ start: 20, end: 22 }], { kind: 'swell', amount: 0.8 }, 49)
    if (!res.ok) throw new Error(res.error)
    expect(envelopeAt(res.keyframes, 21)).toBeCloseTo((0.5 + (0.5 * 21) / 49) * 1.8, 2)
    expect(envelopeAt(res.keyframes, 5)).toBeCloseTo(0.5 + (0.5 * 5) / 49, 3)
  })

  it('holds through short gaps when a layer would need too many points', () => {
    const many = Array.from({ length: 30 }, (_, i) => ({ start: i * 1.6, end: i * 1.6 + 0.2 }))
    const res = applyGainPull([], many, { kind: 'hush', amount: 0.5 }, 49)
    if (!res.ok) throw new Error(res.error)
    expect(res.keyframes.length).toBeLessThanOrEqual(MAX_KEYFRAMES)
    for (const m of many) expect(envelopeAt(res.keyframes, m.start + 0.1)).toBeCloseTo(0.55)
    // Few moments keep their own windows: the gain comes back between them.
    const few = applyGainPull([], [{ start: 5, end: 6 }, { start: 10, end: 11 }], { kind: 'hush', amount: 0.5 }, 49)
    if (!few.ok) throw new Error(few.error)
    expect(envelopeAt(few.keyframes, 8)).toBeCloseTo(1)
  })
})

group('echo', () => {
  it('lands at the start or right after each moment', () => {
    const ms = [{ start: 13.6, end: 15.9 }, { start: 44.8, end: 46.1 }]
    expect(echoTimes(ms, 'start')).toEqual([13.6, 44.8])
    expect(echoTimes(ms, 'end')).toEqual([15.9, 46.1])
    expect(echoGain(0.7)).toBe(1)
  })
})

group('finding moments', () => {
  it('finds where a layer is loud, with hysteresis and joining short gaps', () => {
    const sr = 8000
    const x = new Float32Array(10 * sr)
    const burst = (a: number, b: number) => { for (let i = a * sr; i < b * sr; i++) x[i] = Math.sin(i / 3) * 0.5 }
    burst(1, 2); burst(2.3, 3) // a short gap joins
    burst(6, 7.5)
    const ms = loudMoments(x, sr, 0, 10)
    expect(ms).toHaveLength(2)
    expect(ms[0].start).toBeCloseTo(1, 0); expect(ms[0].end).toBeCloseTo(3, 0)
    expect(ms[1].start).toBeCloseTo(6, 0); expect(ms[1].end).toBeCloseTo(7.5, 0)
    expect(loudMoments(x, sr, 2, 12)[0].start).toBeCloseTo(3, 0)
    expect(loudMoments(new Float32Array(sr), sr, 0, 1)).toEqual([])
  })

  it('finds every occurrence of a phrase in timed words', () => {
    const words = [
      { start: 13.6, end: 13.9, text: 'Here' }, { start: 14.0, end: 14.1, text: 'I' }, { start: 14.2, end: 15.9, text: 'am.' },
      { start: 30, end: 30.2, text: 'here,' }, { start: 30.3, end: 30.4, text: 'I' }, { start: 30.5, end: 30.8, text: 'AM' },
    ]
    expect(findPhrase(words, 'here I am')).toEqual([
      { start: 13.6, end: 15.9, note: '“here I am”' },
      { start: 30, end: 30.8, note: '“here I am”' },
    ])
    expect(findPhrase(words, '  ')).toEqual([])
  })
})

group('rules', () => {
  it('pick layers by name for each recipe', () => {
    expect(pickLayer(stems, ['thunder'])).toBe('t')
    expect(pickLayer(stems, ['nothing'])).toBe('d')
    const lie = newRule(stems, 0, RECIPES.find((r) => r.id === 'lie'))
    expect(lie.targetStemId).toBe('t')
    expect(lie.response).toMatchObject({ kind: 'echo', sourceStemId: 't', anchor: 'end' })
    expect(newRule(stems, 0, RECIPES.find((r) => r.id === 'words')).targetStemId).toBe('r')
    expect(newRule(stems, 0, RECIPES.find((r) => r.id === 'god')).targetStemId).toBe('d')
    expect(newRule(stems, 0, RECIPES.find((r) => r.id === 'dread')).targetStemId).toBe('x')
  })

  it('reads as one sentence', () => {
    const rule: Rule = { ...newRule(stems, 0, RECIPES[0]), meaning: 'a character lies' }
    expect(describe(rule, stems)).toBe('Whenever “a character lies”, Thunder rumbling answers with its loudest sound')
    expect(describe({ ...rule, targetStemId: 'r', response: { kind: 'hush', amount: 0.5 } }, stems)).toBe('Whenever “a character lies”, Heavy rain hushes')
    expect(describe({ ...rule, response: { kind: 'effect', effect: 'deepen', amount: 0.5 } }, stems)).toBe('Whenever “a character lies”, Thunder rumbling deepens')
  })

  it('knows when an applied rule was edited', () => {
    const rule: Rule = { ...newRule(stems, 0), moments: [{ start: 1, end: 2 }] }
    expect(ruleState(rule)).toBe('draft')
    const applied: Rule = { ...rule, applied: { kind: 'gain', stemId: 'd', before: [], after: [], signature: signature(rule) } }
    expect(ruleState(applied)).toBe('applied')
    expect(ruleState({ ...applied, meaning: 'renamed' })).toBe('applied')
    expect(ruleState({ ...applied, moments: [{ start: 1, end: 3 }] })).toBe('changed')
  })

  it('switches responses and keeps the strength', () => {
    const r = responseFor('deepen', { kind: 'swell', amount: 0.3 }, 't')
    expect(r).toEqual({ kind: 'effect', effect: 'deepen', amount: 0.3 })
    expect(responseFor('echo', r, 't')).toEqual({ kind: 'echo', sourceStemId: 't', source: null, anchor: 'start', amount: 0.3 })
    expect(responseFor('hush', r, 't')).toEqual({ kind: 'hush', amount: 0.3 })
  })
})
