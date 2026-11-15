import {describe,expect,it} from 'vitest'
import {extractLayerScope,resolveSourceLayer} from './layerScope'
import type {SavedStem} from '../api/backend'

describe('explicit source layer',()=>{
  it('removes the source scope without losing the action or timestamp',()=>{
    expect(extractLayerScope('separate the barking from the music layer at 4 seconds')).toEqual({name:'music',text:'separate the barking at 4 seconds'})
    expect(extractLayerScope('replace the bark in the music track with a quack')).toEqual({name:'music',text:'replace the bark with a quack'})
    expect(extractLayerScope('separate the barking')).toBeNull()
  })
  it('refuses missing and ambiguous source names',()=>{
    const rows=[{id:'1',name:'Music',label:'music'},{id:'2',name:'Music',label:'music'}] as SavedStem[]
    expect(()=>resolveSourceLayer('music',rows)).toThrow(/Several layers/)
    expect(()=>resolveSourceLayer('dog',rows)).toThrow(/No layer/)
    expect(resolveSourceLayer('music',rows.slice(0,1)).id).toBe('1')
  })
})
