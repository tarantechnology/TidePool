import { describe, expect, it } from 'vitest'
import { musicCommand, restCommand, separationCommand } from './separationCommand'

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

describe('timestamped separation commands', () => {
  it('recognizes a local timestamp without implying full-clip processing', () => {
    expect(separationCommand('remove the knife sounds at 4s')).toEqual({description:'knife sounds',remove:true,processingTime:4})
    expect(separationCommand('lower the dog barking around 4.3 seconds')).toEqual({description:'dog barking',remove:false,processingTime:4.3,editScope:{action:'lower'}})
  })
  it('does not discard extra conditions', () => {
    expect(separationCommand('remove dog and music at 4s')).toBeNull()
    expect(separationCommand('remove dog at 4s but keep it later')).toBeNull()
  })
})

describe('music and timestamped commands together', () => {
  it('never reads a timestamped request as a whole-song split', () => {
    expect(musicCommand('split this song at 30s')).toBeNull()
    expect(separationCommand('isolate the drums at 30s')).toEqual({description:'drums',remove:false,processingTime:30})
  })
})

describe('rest commands', () => {
  it('reads a request to separate the other track', () => {
    for (const text of ['Split the rest', 'separate the other track', "What's in the other track?", 'what is in the rest', 'separate the rest into instruments', 'break up the other stem']) {
      expect(restCommand(text), text).toBe(true)
    }
  })
  it('leaves everything else alone', () => {
    for (const text of ['split this song into tracks', 'separate the dog', 'mute the other track', 'separate the rest of the song from 1 to 2 seconds']) {
      expect(restCommand(text), text).toBe(false)
      expect(musicCommand('split the rest')).toBeNull()
    }
  })
})
