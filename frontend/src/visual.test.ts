import { describe, expect, it } from 'vitest'
import { asVisualMetadata, frameAt, selectVisual } from './visual'
import type { VisualMetadata, VisualObject } from './api/backend'

describe('visual selection', () => {
  const object: VisualObject={detection_id:'d',track_id:3,label:'dog',bbox_normalized:[.1,.2,.3,.4],mask:{encoding:'rle'}}
  const data: VisualMetadata={schema_version:'1.0',video_id:'p',analysis_id:'a',source_sha256:'a'.repeat(64),source:'x',width:64,height:48,fps:10,processed_frames:10,processed_duration_seconds:1,frames:[{frame_index:0,timestamp_seconds:0,valid_until_seconds:.1,objects:[object]}]}
  it('preserves the clicked observation and sets a frame-aligned range', () => {
    const selected=selectVisual(data,.05,object)
    expect(selected).toMatchObject({video_id:'p',analysis_id:'a',track_id:3,frame_index:0,interval:{start_seconds:0,end_seconds:1}})
    expect(frameAt(data,.2)).toBeUndefined()
  })
  it('does not invent a visual selection without a tracked mask', () => {
    expect(selectVisual(data,.05,{...object,mask:null})).toBeNull()
    expect(() => asVisualMetadata({frames:'bad'})).toThrow('Invalid detection metadata')
  })
})
