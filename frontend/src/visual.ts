import type { VisualFrame, VisualMetadata, VisualObject } from './api/backend'

export function objectSound(label:string):string {
  if (/^(dog|puppy)$/i.test(label.trim())) return 'dog barking'
  if (/^(person|people)$/i.test(label.trim())) return 'person speaking'
  return label.trim()
}

export interface VisualSelection extends VisualObject {
  schema_version:string; kind:string; video_id:string; analysis_id:string; source:string; source_sha256:string
  frame_width:number; frame_height:number; frame_index:number; timestamp_seconds:number; clicked_at_seconds:number
  interval:{start_seconds:number;end_seconds:number}
}
/** Find a native-frame observation and preserve its source identity and exact box. */
export function frameAt(metadata: VisualMetadata, time: number): VisualFrame | undefined {
  let lo=0, hi=metadata.frames.length
  while(lo<hi) { const mid=(lo+hi)>>1; if(metadata.frames[mid].timestamp_seconds<=time) lo=mid+1; else hi=mid }
  const frame=metadata.frames[lo-1]
  return frame && time < frame.valid_until_seconds ? frame : undefined
}
export function selectVisual(metadata: VisualMetadata, time: number, object: VisualObject): VisualSelection | null {
  const frame=frameAt(metadata,time)
  if (!frame || object.track_id === null || !object.mask || !frame.objects.includes(object)) return null
  const half=Math.max(1,Math.floor(metadata.fps+.5))
  const round=(n:number)=>Math.round(n*1e9)/1e9
  return {...object,schema_version:'1.0',kind:'visual_selection',video_id:metadata.video_id,analysis_id:metadata.analysis_id,
    source:metadata.source,source_sha256:metadata.source_sha256,frame_width:metadata.width,frame_height:metadata.height,
    frame_index:frame.frame_index,timestamp_seconds:frame.timestamp_seconds,clicked_at_seconds:time,
    interval:{start_seconds:round(Math.max(0,frame.frame_index-half)/metadata.fps),end_seconds:Math.min(metadata.processed_duration_seconds,round(Math.min(metadata.processed_frames,frame.frame_index+half+1)/metadata.fps))}}
}

export function asVisualMetadata(value: unknown): VisualMetadata {
  const data = value as VisualMetadata
  if (!data || typeof data !== 'object' || !Array.isArray(data.frames) || !Number.isFinite(data.fps) || data.fps <= 0 ||
      !Number.isFinite(data.width) || !Number.isFinite(data.height) || !Number.isFinite(data.processed_duration_seconds) ||
      typeof data.source_sha256 !== 'string' || typeof data.analysis_id !== 'string' || typeof data.video_id !== 'string' ||
      data.frames.some(f => !Array.isArray(f.objects) || typeof f.timestamp_seconds !== 'number' || typeof f.valid_until_seconds !== 'number' ||
        f.objects.some(o => !Array.isArray(o.bbox_normalized) || o.bbox_normalized.length !== 4))) {
    throw new Error('Invalid detection metadata')
  }
  return data
}
