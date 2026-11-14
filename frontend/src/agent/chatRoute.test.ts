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
    expect(chatRoute('isolate the dog barking', stems)).toEqual({ fx: null, sfx: false })
  })
})
