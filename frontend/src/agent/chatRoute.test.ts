import { describe, expect, it } from 'vitest'
import { chatRoute } from './chatRoute'

const stems = [{ id: 'v', label: 'Vocals' }, { id: 'd', label: 'Drums' }, { id: 'o', label: 'Other' }]

describe('chatRoute: effects vs library sound inserts', () => {
  it('sends layer effects to fx', () => {
    for (const ask of ['add reverb to the vocals', 'make the vocals sound like a phone', 'put an echo on the drums', 'add reverb to the vocals at 0:12', 'duck the music under the voice', 'normalize for youtube']) {
      const r = chatRoute(ask, stems)
      expect(r.sfx, ask).toBe(false)
      expect(r.fx?.kind, ask).not.toBeUndefined()
    }
  })
  it('sends library sounds to inserts', () => {
    for (const ask of ['add a vine boom at 0:12', 'add a riser before the drop', 'record scratch at 0:05', 'add a laugh track at the end', "add a phone ring at 0:12", 'play a radio static sting at the end']) {
      const r = chatRoute(ask, stems)
      expect(r.sfx, ask).toBe(true)
      expect(r.fx, ask).toBeNull()
    }
  })
  it('leaves unrelated asks to the planner', () => {
    for (const ask of ['isolate the dog barking', 'isolate the radio chatter', 'extract the robot voice', 'remove the radio chatter']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: null })
    }
  })
})

describe('chatRoute: replacement precedence', () => {
  it('replace / swap / substitute are always replacements, never inserts or effects', () => {
    for (const ask of ['replace the phone ringing with a doorbell', 'replace the dog bark with a monster roar', 'swap the radio chatter for birdsong',
      'substitute the robot beeps with a cat meow', 'replace the vine boom with a record scratch', 'replace the echo with silence at 0:12']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: ask })
    }
  })
  it('answers to a pending replacement question stay replacements', () => {
    for (const pending of ['replace the phone ringing with a doorbell', 'replace the dog bark with a monster roar']) {
      for (const answer of ['12', 'at 12', 'around 4.5 seconds', '7s']) {
        const r = chatRoute(answer, stems, pending)
        expect(r.fx, answer).toBeNull(); expect(r.sfx, answer).toBe(false)
        expect(r.replacement, answer).toMatch(new RegExp(`^${pending} at \\d`))
      }
    }
    expect(chatRoute('12', stems)).toEqual({ fx: null, sfx: false, replacement: null })
  })
  it('"add a <sound> at N" is never an effect: an insert when it reads as one, else (or as fallback) a replacement', () => {
    for (const ask of ['add a monster roar at 12', 'add a phone ringing at 3 seconds', 'add a phone ring at 0:12']) {
      const r = chatRoute(ask, stems)
      expect(r.fx, ask).toBeNull(); expect(r.replacement, ask).toBe(ask)
    }
  })
  it('"turn X into Y" without an effect target is a replacement; with a preset it is an effect', () => {
    for (const ask of ['turn the chorus into a verse', 'make the phone ringing sound like a doorbell', 'make the dog bark sound like a cat']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: ask })
    }
    for (const ask of ['make the drums sound like a phone', 'turn the vocals into a robot']) {
      const r = chatRoute(ask, stems)
      expect(r.fx?.kind, ask).toBe('fx'); expect(r.replacement, ask).toBeNull()
    }
  })
})
