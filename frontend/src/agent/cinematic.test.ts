import { describe, expect, it } from 'vitest'
import { cinematicCommand } from './cinematic'

describe('cinematic command fallback', () => {
  it('recognizes an unconstrained whole-mix request', () => {
    expect(cinematicCommand('Make the whole video more cinematic', false)).toBe('cinematic')
    expect(cinematicCommand('Please make it sound cinematic.', false)).toBe('cinematic')
    expect(cinematicCommand('Can you make the whole video more cinematic?', false)).toBe('cinematic')
  })
  it('understands refinements after a cinematic preview', () => {
    expect(cinematicCommand('make it more dramatic', true)).toBe('dramatic')
    expect(cinematicCommand('less dramatic', true)).toBe('subtle')
    expect(cinematicCommand('less dramatic', false)).toBeNull()
  })
  it('does not silently ignore additional user constraints', () => {
    expect(cinematicCommand('Make the video cinematic but keep the music', false)).toBeNull()
    expect(cinematicCommand('Do not make it cinematic', false)).toBeNull()
    expect(cinematicCommand('Make the first five seconds cinematic', false)).toBeNull()
  })
})
