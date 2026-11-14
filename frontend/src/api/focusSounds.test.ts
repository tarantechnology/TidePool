import { describe, expect, it } from 'vitest'
import { focusSounds } from './separation'

describe('focusSounds', () => {
  const heard = [
    { prompt: 'heavy rain', tier: 'background', score: 0.98 },
    { prompt: 'high-pitched screech', tier: 'hits', score: 0.55 },
    { prompt: 'punches and fight impacts', tier: 'hits', score: 0.43 },
    { prompt: 'deep male voice speaking', tier: 'voices', score: 0.02 },
    { prompt: 'tense cinematic music', tier: 'music', score: 0.06 },
  ]
  it('keeps voices, weather with its thunder, and the one most distinctive hit', () => {
    expect(focusSounds(heard)).toEqual(['deep male voice speaking', 'heavy rain', 'thunder rumbling', 'high-pitched screech'])
  })
  it('leaves music and lesser hits in the background', () => {
    expect(focusSounds(heard)).not.toContain('tense cinematic music')
    expect(focusSounds(heard)).not.toContain('punches and fight impacts')
  })
  it('adds no thunder without rain, and nothing when nothing was heard', () => {
    expect(focusSounds([{ prompt: 'dog barking', tier: 'hits', score: 0.6 }])).toEqual(['dog barking'])
    expect(focusSounds([])).toEqual([])
  })
})
