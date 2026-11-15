import { describe, expect, it } from 'vitest'
import { requestsWordEffect } from './wordEffectIntent'
import { chatRoute } from './chatRoute'
import { fxRequest } from '../api/backend'

describe('word effects', () => {
  it.each(['bass boost the bad words', 'add echo whenever he says McLovin', 'muffle every time someone swears', 'put reverb on the word "fake ID"', 'deepen person 2 whenever he says McLovin', "make every bad word and instance of the word 'mohammed' bass boosted", 'bass boost every instance of mohammed', 'bass boost the word mohammed', 'make every Mohammed squeaky', 'make the bad words and every Muhammad sound like a giant'])('%s goes to the planner', text => {
    expect(requestsWordEffect(text)).toBe(true)
    expect(chatRoute(text)).toEqual({ fx: null, sfx: false, replacement: null, generate: false })
  })
  it.each(['add a vine boom when he says what', 'add reverb to the vocals', 'bass boost from 2 to 4 seconds'])('%s is not one', text => {
    expect(requestsWordEffect(text)).toBe(false)
  })
  it('keeps bleeping in the censorship flow', () => {
    expect(chatRoute('bleep the swearing').censorship).toBe(true)
  })
  it('sends the words with the effect preview', () => {
    const plan = { targets: ['mix'], preset: 'bass_boost', chain: null, interval: null, adjust: null, speed: null, duck_source: null, duck_depth_db: null, loudness_lufs: null, on_words: { profanity: true, words: [] } }
    expect(fxRequest('fx', plan)).toMatchObject({ op: 'chain', preset: 'bass_boost', on_words: { profanity: true, words: [] } })
    expect(fxRequest('fx', { ...plan, on_words: null })).not.toHaveProperty('on_words')
  })
})
