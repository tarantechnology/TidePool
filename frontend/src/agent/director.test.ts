import {describe,it,expect} from 'vitest'
import {creativeCommand,directorIntent} from './director'
describe('creative command routing',()=>{
  it('recognizes broad requests without a model',()=>{
    expect(creativeCommand('Make it cool')).toBe('cool')
    expect(creativeCommand('Make the whole video more dramatic')).toBe('dramatic')
    expect(creativeCommand('Emphasize the funny sounds')).toBe('funny')
    expect(directorIntent('make it more dramatic')).toBe('dramatic')
  })
  it('does not silently discard constraints',()=>{
    expect(creativeCommand('make it dramatic but keep the music')).toBeNull()
    expect(creativeCommand('emphasize funny sounds at 4 seconds')).toBeNull()
  })
})
