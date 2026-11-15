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
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: null, generate: false })
    }
  })
})

describe('chatRoute: replacement precedence', () => {
  it('replace / swap / substitute are always replacements, never inserts or effects', () => {
    for (const ask of ['replace the phone ringing with a doorbell', 'replace the dog bark with a monster roar', 'swap the radio chatter for birdsong',
      'substitute the robot beeps with a cat meow', 'replace the vine boom with a record scratch', 'replace the echo with silence at 0:12']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: ask, generate: false })
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
    expect(chatRoute('12', stems)).toEqual({ fx: null, sfx: false, replacement: null, generate: false })
  })
  it('"add a <sound> at N" is never an effect: an insert when it reads as one, else (or as fallback) a replacement', () => {
    for (const ask of ['add a monster roar at 12', 'add a phone ringing at 3 seconds', 'add a phone ring at 0:12']) {
      const r = chatRoute(ask, stems)
      expect(r.fx, ask).toBeNull(); expect(r.replacement, ask).toBe(ask)
    }
  })
  it('"turn X into Y" without an effect target is a replacement; with a preset it is an effect', () => {
    for (const ask of ['turn the chorus into a verse', 'make the phone ringing sound like a doorbell', 'make the dog bark sound like a cat']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: ask, generate: false })
    }
    for (const ask of ['make the drums sound like a phone', 'turn the vocals into a robot']) {
      const r = chatRoute(ask, stems)
      expect(r.fx?.kind, ask).toBe('fx'); expect(r.replacement, ask).toBeNull()
    }
  })
  it('precedence: replacement → insert (named library sound) → generate (instrument part) → fx', () => {
    for (const ask of ['add a vine boom at 0:12', 'add a sad trombone at the end', 'put a riser before the drop']) {
      const r = chatRoute(ask, stems)
      expect(r.sfx, ask).toBe(true); expect(r.generate, ask).toBe(false); expect(r.fx, ask).toBeNull()
    }
    for (const ask of ['add a piano part', 'add violins', 'add a funky bassline', 'add strings in the chorus', 'add a choir', 'add an 80s synth arpeggio', 'add a piano at 0:12']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: null, generate: true })
    }
    for (const ask of ['add reverb to the vocals', 'add chorus to the guitar']) {
      const r = chatRoute(ask, stems)
      expect(r.generate, ask).toBe(false); expect(r.sfx, ask).toBe(false); expect(r.fx, ask).not.toBeNull()
    }
    for (const ask of ['add a fade', 'add more bass', 'add compression to the bass']) expect(chatRoute(ask, stems).generate, ask).toBe(false)
    for (const ask of ['put the keys on the table', 'bring in the horns of the car']) expect(chatRoute(ask, stems).generate, ask).toBe(false)
    for (const ask of ['put a string pad under the chorus', 'give me a funky bassline', 'layer some strings']) expect(chatRoute(ask, stems).generate, ask).toBe(true)
    expect(chatRoute('replace the piano with violins', stems).generate).toBe(false)
    expect(chatRoute('replace the piano with violins', stems).replacement).toBe('replace the piano with violins')
  })
})

describe('chatRoute: censorship and layer scope come first', () => {
  it('profanity beeps always go to the censorship flow, even with a pending replacement question', () => {
    for (const ask of ['bleep the swearing', 'censor the cuss words', 'beep out curse words', 'put a bleep over the swear words', 'mute the profanity']) {
      expect(chatRoute(ask, stems), ask).toEqual({ fx: null, sfx: false, replacement: null, generate: false, censorship: true })
      expect(chatRoute(ask, stems, 'replace the dog bark with a meow').censorship, ask).toBe(true)
    }
  })
  it('a generate ask never pre-empts an object-appearance placement', () => {
    for (const ask of ['add a gong every time we see person 5', 'add a drum hit when car 2 appears']) {
      const r = chatRoute(ask, stems); expect(r.generate, ask).toBe(false); expect(r.replacement, ask).toBe(ask)
    }
  })
  it('a generate ask never pre-empts censorship', () => {
    for (const ask of ['add violins and bleep the swearing', 'censor the cuss words then add a piano part']) {
      const r = chatRoute(ask, stems); expect(r.censorship, ask).toBe(true); expect(r.generate, ask).toBe(false)
    }
  })
  it('an explicit censor-bleep library insert is still an insert', () => {
    const r = chatRoute('add a censor bleep at 0:12', stems)
    expect(r.censorship).toBeUndefined(); expect(r.sfx).toBe(true)
  })
  it('a separate/remove ask naming one layer goes to layer separation', () => {
    expect(chatRoute('isolate the dog in the Music layer', stems)).toEqual({ fx: null, sfx: false, replacement: null, generate: false, layer: { name: 'Music', text: 'isolate the dog' } })
    expect(chatRoute('remove the cough from the Vocals track', stems).layer).toEqual({ name: 'Vocals', text: 'remove the cough' })
  })
  it('layer-scoped replacements stay replacements; effects on a named layer stay effects', () => {
    const ask = 'replace the dog bark in the Music layer with a cat meow'
    expect(chatRoute(ask, stems)).toEqual({ fx: null, sfx: false, replacement: ask, generate: false })
    const fx = chatRoute('add reverb on the Vocals track', stems)
    expect(fx.layer).toBeUndefined(); expect(fx.fx?.kind).toBe('fx')
  })
  it('a separation naming two layers is ambiguous', () => {
    expect(() => chatRoute('isolate the dog in the Music layer from the Vocals track', stems)).toThrow(/one source layer/)
  })
  it('a named library insert wins over a pending replacement question', () => {
    const r = chatRoute('add a vine boom at 0:12', stems, 'replace the dog bark with a meow')
    expect(r.sfx).toBe(true); expect(r.replacement === null || r.replacement === 'add a vine boom at 0:12').toBe(true)
    expect(chatRoute('at 12', stems, 'replace the dog bark with a meow').replacement).toMatch(/at 12 seconds$/)
  })
  it('pitch asks without an amount are effects', () => {
    const r = chatRoute('make the vocals higher pitch', stems)
    expect(r.fx?.kind).toBe('fx'); expect(r.sfx).toBe(false); expect(r.replacement).toBeNull()
  })
  it("object-appearance sound asks go to main's generation flow, not a library insert or effect", () => {
    for (const ask of ['Add a gong every time we see person 5', 'Every time person 5 appears, play a gong sound.', 'play a whoosh whenever car 2 appears']) {
      const r = chatRoute(ask, stems)
      expect(r.replacement, ask).toBe(ask); expect(r.sfx, ask).toBe(false); expect(r.fx, ask).toBeNull()
    }
  })
})
