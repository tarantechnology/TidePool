/** Source intervals in playback order. An empty list represents the full source. */
export interface SourceClip { start:number; end:number }
const EPSILON=0.001

/** Split around a source selection without dropping or duplicating samples. */
export function splitSelection(clips:SourceClip[], duration:number, start:number, end:number): {clips:SourceClip[];selected:number} {
  if (!Number.isFinite(duration) || !Number.isFinite(start) || !Number.isFinite(end) || duration <= 0 || start < 0 || end-start < .01 || end > duration + 1e-6) throw new Error('Select a valid range inside the media.')
  const current=clips.length ? clips : [{start:0,end:duration}]
  const matching=current.map((clip,index)=>({clip,index})).filter(({clip})=>start >= clip.start-EPSILON && end <= clip.end+EPSILON)
  if (matching.length !== 1) throw new Error('Select a range within one clip. Repeated or overlapping clips must be edited from the clip list.')
  const {clip,index}=matching[0]
  const from=Math.max(start,clip.start), to=Math.min(end,clip.end)
  const parts:SourceClip[]=[]
  if (from > clip.start) parts.push({start:clip.start,end:from})
  const selected=index+parts.length
  parts.push({start:from,end:to})
  if (to < clip.end) parts.push({start:to,end:clip.end})
  if (current.length-1+parts.length > 100) throw new Error('The clip limit is 100. Remove clips first.')
  return {clips:[...current.slice(0,index),...parts,...current.slice(index+1)],selected}
}
