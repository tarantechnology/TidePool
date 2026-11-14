import { describe, expect, it } from 'vitest'
import { resolveActions } from '../audio/resolve'
import { initialTracks, parsePrompt } from '../edits'
import type { Stem } from '../types'
import { createEditorState, mergeKinds, setKindOverride } from './kinds'

const stems: Stem[] = [
  { id: 'a', label: 'Crowd', kind: 'sfx', url: 'blob:a', duration: 4 },
  { id: 'b', label: 'Siren', kind: 'sfx', url: 'blob:b', duration: 4 },
]

describe('mergeKinds', () => {
  it('lets an override win over the stem kind', () => {
    const merged = mergeKinds(stems, new Map([['a', 'dialogue']]))
    expect(merged.map((s) => s.kind)).toEqual(['dialogue', 'sfx'])
  })

  it('leaves everything else on the stem alone', () => {
    const [merged] = mergeKinds(stems, new Map([['a', 'music']]))
    expect(merged).toEqual({ ...stems[0], kind: 'music' })
  })

  it('ignores overrides for unknown stems', () => {
    expect(mergeKinds(stems, new Map([['gone', 'dialogue']]))).toEqual(stems)
  })

  it('does not mutate the original stems', () => {
    const before = structuredClone(stems)
    mergeKinds(stems, new Map([['a', 'dialogue']]))
    expect(stems).toEqual(before)
  })
})

describe('setKindOverride', () => {
  it('returns a new map, so older undo steps keep their kinds', () => {
    const first = createEditorState([]).kinds
    const second = setKindOverride(first, 'a', 'dialogue')
    const third = setKindOverride(second, 'a', 'music')
    expect(first.size).toBe(0)
    expect(second.get('a')).toBe('dialogue')
    expect(third.get('a')).toBe('music')
  })
})

describe('intensity on stems with a picked kind', () => {
  const PROMPT = 'make the siren louder whenever the argument gets intense'
  const tracks = initialTracks(stems)
  const source = {
    getSamples: (id: string) => (id === 'a' ? { data: new Float32Array(16000).fill(0.5), sampleRate: 8000 } : null),
  }

  it('has nothing to follow while every stem is the default sfx', () => {
    const { actions, message } = parsePrompt(PROMPT, stems, tracks)
    expect(actions).toEqual([])
    expect(message).toContain('No dialogue stems')
  })

  it('follows a stem once its kind is changed to dialogue', () => {
    const merged = mergeKinds(stems, setKindOverride(new Map(), 'a', 'dialogue'))
    const { actions } = parsePrompt(PROMPT, merged, tracks)
    expect(actions).toEqual([{ type: 'follow_intensity', stemId: 'b', sourceStemIds: ['a'] }])

    const resolved = resolveActions(actions, source, 4)
    expect(resolved.warnings).toEqual([])
    expect(resolved.actions).toHaveLength(1)
    expect(resolved.actions[0]).toMatchObject({ type: 'set_automation', stemId: 'b' })
  })
})
