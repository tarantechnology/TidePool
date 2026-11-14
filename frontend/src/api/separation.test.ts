import { describe, expect, it } from 'vitest'
import type { SavedStem } from './backend'
import { buildVisualPrompt, canRemove, findRemoveTargets, missingResidualMessage, musicStemName, normalizeStemName, NO_VISUAL_PROMPT } from './separation'

const stem = (over: Partial<SavedStem> & { id: string; name: string }): SavedStem => ({
  duration: 4,
  audio_url: `/audio/${over.id}`,
  revision: 0,
  volume: 1,
  muted: false,
  solo: false,
  automation: [],
  start_offset: 0,
  mute_ranges: [],
  ...over,
})

describe('buildVisualPrompt', () => {
  it('is an empty object with no selection, never omitted', () => {
    expect(buildVisualPrompt(null, '', '')).toEqual({})
    expect(buildVisualPrompt(null, '1', '2')).toBe(NO_VISUAL_PROMPT)
    // The field has to be present in the body, so it must serialize.
    expect(JSON.stringify({ visual_prompt: buildVisualPrompt(null, '', '') })).toBe('{"visual_prompt":{}}')
  })

  it('carries the selection with the interval from the range inputs', () => {
    const selection = { detection_id: 'd1', label: 'dog', interval: { start_seconds: 0, end_seconds: 1 } }
    expect(buildVisualPrompt(selection, '2', '4.5')).toEqual({
      detection_id: 'd1',
      label: 'dog',
      interval: { start_seconds: 2, end_seconds: 4.5 },
    })
  })

  it('refuses an interval that is empty, backwards or not a number', () => {
    const selection = { label: 'dog' }
    for (const [start, end] of [['', '4'], ['2', ''], ['x', '4'], ['4', '2'], ['2', '2']]) {
      expect(() => buildVisualPrompt(selection, start, end)).toThrow('valid visual selection interval')
    }
  })
})

describe('normalizeStemName', () => {
  it('trims, collapses whitespace and ignores case', () => {
    expect(normalizeStemName('  Dog   Barking \n')).toBe('dog barking')
  })
})

describe('findRemoveTargets', () => {
  it('matches on role, and on the prompt the separation was asked for', () => {
    const stems = [
      stem({ id: 't', name: 'dog barking', role: 'target', prompt: 'dog   barking' }),
      stem({ id: 'r', name: 'rest of the mix', role: 'residual' }),
    ]
    const found = findRemoveTargets(stems, '  Dog Barking ')
    expect(found.target?.id).toBe('t')
    expect(found.residual?.id).toBe('r')
  })

  it('takes a lone target even when the service reworded the prompt', () => {
    const stems = [
      stem({ id: 't', name: 'barking dog', role: 'target', prompt: 'barking dog' }),
      stem({ id: 'r', name: 'residual', role: 'residual' }),
    ]
    expect(findRemoveTargets(stems, 'dog barking').target?.id).toBe('t')
  })

  it('needs a name match when several targets came back', () => {
    const stems = [
      stem({ id: 'a', name: 'dog barking', role: 'target' }),
      stem({ id: 'b', name: 'siren', role: 'target' }),
      stem({ id: 'r', name: 'residual', role: 'residual' }),
    ]
    expect(findRemoveTargets(stems, 'siren').target?.id).toBe('b')
    expect(findRemoveTargets(stems, 'cat').target).toBeNull()
  })

  it('falls back to whitespace-collapsed, case-insensitive names without roles', () => {
    const stems = [stem({ id: 't', name: 'Dog   Barking' }), stem({ id: 'r', name: ' Residual ' })]
    const found = findRemoveTargets(stems, 'dog barking')
    expect(found.target?.id).toBe('t')
    expect(found.residual?.id).toBe('r')
  })

  it('reports a missing residual rather than guessing one, as the fake adapter returns', () => {
    const stems = [stem({ id: 't', name: 'dog barking', role: 'target' })]
    const found = findRemoveTargets(stems, 'dog barking')
    expect(found.target?.id).toBe('t')
    expect(found.residual).toBeNull()
    expect(missingResidualMessage(found, 'dog barking')).toContain('no residual audio')
    expect(missingResidualMessage(found, 'dog barking')).toContain('nothing was muted')
  })

  it('reports a missing target, and both when neither is there', () => {
    const named = [stem({ id: 'r', name: 'residual' })]
    expect(missingResidualMessage(findRemoveTargets(named, 'siren'), 'siren')).toContain('no "siren" stem')
    const neither = [stem({ id: 'x', name: 'vocals' })]
    expect(missingResidualMessage(findRemoveTargets(neither, 'siren'), 'siren')).toContain('no "siren" stem and no residual')
  })

  it('will not pick a target when two stems share the name', () => {
    const stems = [stem({ id: 'a', name: 'siren' }), stem({ id: 'b', name: 'siren' }), stem({ id: 'r', name: 'residual' })]
    expect(findRemoveTargets(stems, 'siren').target).toBeNull()
  })
})

describe('music splits', () => {
  const split = ['vocals', 'drums', 'bass', 'other'].map((label) => stem({ id: label, name: label[0].toUpperCase() + label.slice(1), role: 'target', label }))

  it('maps everyday words onto stem names', () => {
    expect(musicStemName('The Singer')).toBe('vocals')
    expect(musicStemName('vocals')).toBe('vocals')
    expect(musicStemName('percussion')).toBe('drums')
    expect(musicStemName('dog barking')).toBeNull()
  })

  it('mutes a split track without a residual, since its siblings hold the rest of the song', () => {
    for (const words of ['vocals', 'Vocals', 'the singer', 'voice']) {
      const found = findRemoveTargets(split, words)
      expect(found.target?.id, words).toBe('vocals')
      expect(found.residual).toBeNull()
      expect(found.split).toBe(true)
      expect(canRemove(found)).toBe(true)
    }
    expect(findRemoveTargets(split, 'siren').target).toBeNull()
  })

  it('still needs a residual for a lone target, as the fake adapter returns', () => {
    const found = findRemoveTargets([stem({ id: 'v', name: 'Vocals', role: 'target', label: 'vocals' })], 'vocals')
    expect(found.split).toBe(false)
    expect(canRemove(found)).toBe(false)
  })

  it('mutes a subset split against its residual', () => {
    const found = findRemoveTargets([split[0], stem({ id: 'r', name: 'residual', role: 'residual' })], 'vocals')
    expect(canRemove(found)).toBe(true)
  })
})

describe('instrument tags', () => {
  const tags = (...labels: string[]) => labels.map((label, i) => ({ label, score: 0.9 - i / 10 }))
  const split = [
    stem({ id: 'v', name: 'Vocals', role: 'target', label: 'vocals', tags: tags('Female singing') }),
    stem({ id: 'd', name: 'Drums', role: 'target', label: 'drums', tags: tags('Drum kit') }),
    stem({ id: 'o', name: 'Synth & Strings', role: 'target', label: 'other', tags: tags('Synthesizer', 'Strings', 'Piano') }),
  ]

  it('finds the descriptive "other" track by its tags and by its canonical label', () => {
    for (const words of ['piano', 'the strings', 'synth', 'Synth & Strings', 'other']) {
      const found = findRemoveTargets(split, words)
      expect(found.target?.id, words).toBe('o')
      expect(canRemove(found), words).toBe(true)
    }
    expect(findRemoveTargets(split, 'kit').target).toBeNull()  // too short to shorten a tag word
    expect(findRemoveTargets(split, 'trumpet').target).toBeNull()
  })
})
