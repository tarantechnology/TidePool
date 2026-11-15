import { describe,it,expect } from 'vitest'
import { splitSelection } from './clipSequence'

describe('split a timeline selection',()=>{
  it('splits the full source into left, selection and right without loss',()=>{
    expect(splitSelection([],10,3,4)).toEqual({clips:[{start:0,end:3},{start:3,end:4},{start:4,end:10}],selected:1})
  })
  it('retains existing playback order and selects the new section',()=>{
    expect(splitSelection([{start:6,end:10},{start:0,end:6}],10,2,3)).toEqual({clips:[{start:6,end:10},{start:0,end:2},{start:2,end:3},{start:3,end:6}],selected:2})
  })
  it('refuses ambiguous repeated clips and crossing an existing edit',()=>{
    expect(()=>splitSelection([{start:0,end:5},{start:0,end:5}],10,1,2)).toThrow(/Repeated/)
    expect(()=>splitSelection([{start:0,end:2},{start:2,end:10}],10,1,3)).toThrow(/within one clip/)
    expect(()=>splitSelection([],10,2,11)).toThrow()
  })
})
