import { describe,it,expect } from 'vitest'
import { objectSound } from './visual'
import { toCapabilities } from './api/backend'
import { detectorStatus } from './objects'
import { objectTracks } from './objects'
import { asVisualMetadata } from './visual'
import fixture from './fixtures/detections.trimmed.json'

describe('SAM selection with a text-only audio backend',()=>{
  it('offers a ready object detector when AudioSep visual conditioning is unavailable',()=>{
    const caps=toCapabilities({backend:'audiosep',visual:false,suppression:{mode:'conservative'},music_stems:{available:true,stems:['vocals','drums']},vision:{available:true,model:'sam3.1',device:'cuda',weights_cached:true,selection_only:true}})
    expect(caps.visual).toBe(false)
    expect(caps.vision?.selectionOnly).toBe(true)
    expect(caps.backend).toBe('audiosep')
    expect(caps.preserveVoices).toBe(true)
    expect(caps.musicStems).toEqual({available:true,stems:['vocals','drums']})
    expect(detectorStatus(caps.vision,{phase:'idle'},null).canStart).toBe(true)
  })
  it('starts a dog click with the same description as the verified text path',()=>{
    expect(objectSound('dog')).toBe('dog barking')
    expect(objectSound('Dog')).toBe('dog barking')
    expect(objectSound('person')).toBe('')
    expect(objectSound('guy in white hoodie')).toBe('')
    expect(objectSound('woman')).toBe('')
    expect(objectSound('car')).toBe('car')
  })
  it('allows retrying a temporary GPU connection failure',()=>{
    const caps=toCapabilities({vision:{available:false,retryable:true,selection_only:true,reason:'Tunnel unavailable'}})
    expect(detectorStatus(caps.vision,{phase:'idle'},null).canStart).toBe(true)
    expect(detectorStatus(caps.vision,{phase:'failed',status:502,error:'Tunnel unavailable'},null).canStart).toBe(true)
  })
  it('hides automatic one-frame class flicker without dropping brief SAM tracks',()=>{
    const metadata=asVisualMetadata(fixture)
    const object={detection_id:'d',track_id:1,label:'person',bbox_normalized:[0,0,1,1],mask:{}}
    const frames=Array.from({length:3},(_,i)=>({frame_index:i,timestamp_seconds:i/10,valid_until_seconds:(i+1)/10,
      objects:[object,...(i===0?[{...object,track_id:2,label:'bird'}]:[])]}))
    expect(objectTracks({...metadata,frames,discovery_mode:'automatic'}).map(t=>t.label)).toEqual(['person'])
    expect(objectTracks({...metadata,frames,discovery_mode:'prompted'}).map(t=>t.label)).toEqual(['person','bird'])
  })
})
