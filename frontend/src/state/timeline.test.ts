import {describe,it,expect} from 'vitest'
import {lengthOf,replaceTimeline,sliceTimeline,timelinePosition} from './timeline'
const source=[{start:0,end:3},{start:7,end:10},{start:0,end:3}]
describe('edited timeline coordinates',()=>{
 it('selects across cuts and repeated clips by timeline position',()=>{
  expect(lengthOf(source)).toBe(9)
  expect(sliceTimeline(source,2,7)).toEqual([{start:2,end:3},{start:7,end:10},{start:0,end:1}])
  expect(timelinePosition(source,3)).toEqual({index:1,source:7})
  expect(timelinePosition(source,7)).toEqual({index:2,source:1})
 })
 it('closes gaps and pastes at the edited playhead',()=>{
  expect(replaceTimeline(source,2,7)).toEqual([{start:0,end:2},{start:1,end:3}])
  const copied=sliceTimeline(source,2,4)
  expect(replaceTimeline(source,3,3,copied)).toEqual([{start:0,end:3},{start:2,end:3},{start:7,end:8},{start:7,end:10},{start:0,end:3}])
  expect(()=>replaceTimeline(source,0,9)).toThrow(/Keep/)
 })
})
