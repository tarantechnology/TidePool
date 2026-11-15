import type { SourceClip } from './clipSequence'
export const lengthOf=(clips:SourceClip[])=>clips.reduce((n,c)=>n+c.end-c.start,0)
export function sliceTimeline(clips:SourceClip[],start:number,end:number):SourceClip[] {
  const result:SourceClip[]=[];let offset=0
  for(const clip of clips){const size=clip.end-clip.start,a=Math.max(start,offset),b=Math.min(end,offset+size);if(b>a)result.push({start:clip.start+a-offset,end:clip.start+b-offset});offset+=size}
  return result
}
export function replaceTimeline(clips:SourceClip[],start:number,end:number,insert:SourceClip[]=[]):SourceClip[]{
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<start||end>lengthOf(clips)+1e-6)throw new Error('Select a range inside the timeline.')
  const result=[...sliceTimeline(clips,0,start),...insert,...sliceTimeline(clips,end,lengthOf(clips))]
  if(!result.length)throw new Error('Keep at least one section of the video.')
  if(result.length>100)throw new Error('The clip limit is 100.')
  return result
}
export function timelinePosition(clips:SourceClip[],time:number):{index:number;source:number}{
  let offset=0
  for(let index=0;index<clips.length;index++){const c=clips[index],size=c.end-c.start;if(time<offset+size||index===clips.length-1)return {index,source:c.start+Math.min(size,Math.max(0,time-offset))};offset+=size}
  return {index:0,source:0}
}
