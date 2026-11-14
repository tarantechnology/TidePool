import { describe, expect, it } from 'vitest'
import { separationCommand } from './separationCommand'

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
