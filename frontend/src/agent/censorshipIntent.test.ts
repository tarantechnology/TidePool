import { describe, expect, it } from 'vitest'
import { requestsCensorship } from './censorshipIntent'
describe('censorship requests', () => {
  it('recognizes explicit profanity censorship', () => {
    expect(requestsCensorship('beep out any cuss words')).toBe(true)
    expect(requestsCensorship('censor profanity')).toBe(true)
    expect(requestsCensorship('remove swearing')).toBe(true)
  })
  it('does not infer consent from unrelated or negated requests', () => {
    expect(requestsCensorship('make the beep louder')).toBe(false)
    expect(requestsCensorship("don't censor profanity")).toBe(false)
    expect(requestsCensorship('Replace all cuss words with a meow sound')).toBe(false)
  })
})
