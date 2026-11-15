import { describe,it,expect } from 'vitest'
import { cutSelection, removedSourceRanges, splitSelection } from './clipSequence'

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

describe('cut a selection from the whole video',()=>{
  it('removes the section and closes the gap in playback order',()=>{
    expect(cutSelection([],10,3,4)).toEqual([{start:0,end:3},{start:4,end:10}])
    expect(cutSelection([],10,0,3)).toEqual([{start:3,end:10}])
    expect(cutSelection([],10,7,10)).toEqual([{start:0,end:7}])
    expect(cutSelection([{start:6,end:10},{start:0,end:6}],10,2,3)).toEqual([{start:6,end:10},{start:0,end:2},{start:3,end:6}])
  })
  it('refuses to remove all media or guess which repeated clip to cut',()=>{
    expect(()=>cutSelection([],10,0,10)).toThrow(/whole video/)
    expect(()=>cutSelection([{start:0,end:5},{start:0,end:5}],10,1,2)).toThrow(/Repeated/)
  })
  it('cuts across existing adjacent or previously cut clips',()=>{
    expect(cutSelection([{start:0,end:3},{start:3,end:6},{start:8,end:10}],10,2,9)).toEqual([{start:0,end:2},{start:9,end:10}])
  })
  it('shows removed source spans even when clips are reordered or repeated',()=>{
    expect(removedSourceRanges([{start:4,end:10},{start:0,end:3},{start:4,end:6}],10)).toEqual([{start:3,end:4}])
    expect(removedSourceRanges([],10)).toEqual([])
  })
})
