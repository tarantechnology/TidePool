import { describe, expect, it } from 'vitest'
import { separationCommand } from './separationCommand'

describe('separation commands', () => {
  it('recognizes explicit whole-track requests', () => {
    expect(separationCommand('Remove the narrator.')).toEqual({description:'narrator',remove:true})
    expect(separationCommand('Isolate dog barking')).toEqual({description:'dog barking',remove:false})
  })
  it('does not treat timed or compound edits as whole-track removal', () => {
    expect(separationCommand('remove the narrator from 10 to 15 seconds')).toBeNull()
    expect(separationCommand('remove the narrator and music')).toBeNull()
    expect(separationCommand('mute the narrator')).toBeNull()
  })
})
