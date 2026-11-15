import { describe, it, expect } from 'vitest'
import { mediaCommand } from './mediaCommand'

describe('media commands', () => {
  it('understands natural trim ranges and spoken numbers', () => {
    for (const text of ['trim 3 sec to 4 sec','please trim the video from three seconds to four seconds','keep between 3 and 4 seconds']) {
      expect(mediaCommand(text)).toEqual({kind:'trim',start:3,end:4,exportFormat:null,download:false})
    }
    expect(mediaCommand('trim from 1:03 to 1:04 and export as mp4')).toEqual({kind:'trim',start:63,end:64,exportFormat:'mp4',download:true})
  })
  it('handles exports and rejects invalid or ambiguous trims', () => {
    expect(mediaCommand('download the video as mp4')).toEqual({kind:'export',format:'mp4'})
    expect(mediaCommand('trim 4 to 3')?.kind).toBe('invalid')
    expect(mediaCommand('trim -3 to 4')?.kind).toBe('invalid')
    expect(mediaCommand('trim from 1:99 to 2:00')?.kind).toBe('invalid')
    expect(mediaCommand('trim the first 3 seconds')?.kind).toBe('invalid')
    expect(mediaCommand('cut 3 to 4')?.kind).toBe('invalid')
    expect(mediaCommand('mute the dog from 3 to 4 seconds')).toBeNull()
    // Sound edits without a time range still go to the AI planner.
    expect(mediaCommand('cut the background music')).toBeNull()
    expect(mediaCommand('trim the silence at the start')).toBeNull()
    expect(mediaCommand('crop out the dog barking')).toBeNull()
  })
})
