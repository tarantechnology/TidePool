import { describe, expect, it } from 'vitest'
import { placementLines, requestsSfxInsert, sfxFollowup } from './sfxIntent'
import { requestsSoundReplacement } from './replacementIntent'

describe('requestsSfxInsert', () => {
  it.each([
    "add a vine boom when he says 'what'",
    'record scratch at 0:12',
    'sad trombone at the end',
    'laugh track after the joke',
    'crickets during the awkward silence',
    'add an airhorn at 1:05',
    'put a quack at 0:03',
    'play something funny on the beat',
    'bruh',
  ])('routes %s to the SFX planner', (text) => {
    expect(requestsSfxInsert(text)).toBe(true)
  })

  it.each([
    'replace the punch with a quack',
    'make the dog sound like a duck',
    'isolate the vocals',
    'add some reverb to the vocals',
    'mute the music from 0:10 to 0:20',
    '',
  ])('leaves %s to the other flows', (text) => {
    expect(requestsSfxInsert(text)).toBe(false)
  })

  it('takes "add X at 0:12" before the replacement flow, which also matches it', () => {
    // ProjectPanel checks requestsSfxInsert first; an unmatched sound falls back to replacement.
    expect(requestsSoundReplacement('add a quack at 3 seconds')).toBe(true)
    expect(requestsSfxInsert('add a record scratch at 0:12')).toBe(true)
  })
})

describe('placementLines', () => {
  it('shows the start and, when shifted, the anchor', () => {
    expect(placementLines([{ title: 'Whoosh', start: 29.8, anchor_time: 30 }, { title: 'Ding', start: 1.5, anchor_time: 1.5 }])).toEqual([
      'Whoosh at 29.80s (anchor 30.00s)',
      'Ding at 1.50s',
    ])
  })
})

describe('sfxFollowup', () => {
  it('completes a request that was missing a time', () => {
    expect(sfxFollowup('laugh track', '0:42')).toBe('add laugh track at 0:42')
    expect(sfxFollowup('vine boom', 'at 12s')).toBe('add vine boom at 12s')
    expect(sfxFollowup('vine boom', 'make it louder')).toBeNull()
    expect(sfxFollowup('', '0:42')).toBeNull()
  })
})
