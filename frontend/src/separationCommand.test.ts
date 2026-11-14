import { describe, expect, it } from 'vitest'
import { musicCommand, separationCommand } from './separationCommand'

describe('separation commands', () => {
  it('recognizes explicit whole-track requests', () => {
    expect(separationCommand('Remove the narrator.')).toEqual({description:'narrator',remove:true})
    expect(separationCommand('Isolate dog barking')).toEqual({description:'dog barking',remove:false})
  })
  it('does not treat timed or compound edits as whole-track removal', () => {
    expect(separationCommand('remove the narrator from 10 to 15 seconds')).toEqual({description:'narrator',remove:true,editScope:{action:'mute',interval:{start:10,end:15}}})
    expect(separationCommand('remove the narrator and music')).toBeNull()
    expect(separationCommand('mute the narrator')).toEqual({description:'narrator',remove:true})
    expect(separationCommand('mute the dog from 15 to 10 seconds')).toBeNull()
  })
})

describe('music commands', () => {
  it('reads a whole-song split', () => {
    for (const text of ['Split this song into tracks', 'split the song', 'Separate the instruments.', 'split into stems', 'break up my track into separate tracks', 'give me the stems']) {
      expect(musicCommand(text), text).toEqual({ stems: null, remove: false, description: 'all instruments' })
    }
  })
  it('reads karaoke and instrumental as muting the vocals', () => {
    for (const text of ['Make a karaoke version', 'karaoke', 'make it instrumental', 'turn this into an instrumental']) {
      expect(musicCommand(text), text).toEqual({ stems: ['vocals'], remove: true, description: 'vocals' })
    }
  })
  it('leaves single sounds and other separations alone', () => {
    for (const text of ['isolate the drums', 'separate the dog barking', 'separate the dog into its own track', 'split the narrator from 1 to 2 seconds', 'mute the vocals']) {
      expect(musicCommand(text), text).toBeNull()
    }
  })
})
