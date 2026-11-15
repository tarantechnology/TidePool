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

/** Remove the selected source interval, joining the remaining video and audio. */
export function cutSelection(clips:SourceClip[], duration:number, start:number, end:number): SourceClip[] {
  if (!Number.isFinite(duration) || !Number.isFinite(start) || !Number.isFinite(end) || duration <= 0 || start < 0 || end-start < .01 || end > duration + 1e-6) throw new Error('Select a valid range inside the media.')
  const current=clips.length ? clips : [{start:0,end:duration}]
  const affected=current.map((clip,index)=>({clip,index})).filter(({clip})=>clip.start < end && clip.end > start)
  if (!affected.length) throw new Error('That range has already been removed.')
  if (affected.some((entry,i) => i > 0 && (entry.index !== affected[i-1].index+1 || entry.clip.start < affected[i-1].clip.end-EPSILON))) throw new Error('Repeated or rearranged clips overlap this selection. Edit them from the clip list.')
  const remaining=current.flatMap(clip => {
    if (clip.end <= start || clip.start >= end) return [clip]
    const parts:SourceClip[]=[]
    if (clip.start < start) parts.push({start:clip.start,end:Math.min(start,clip.end)})
    if (clip.end > end) parts.push({start:Math.max(end,clip.start),end:clip.end})
    return parts
  })
  // An empty list means "full source" elsewhere in the editor, so never use it for an empty video.
  if (!remaining.length) throw new Error('The selection covers the whole video. Keep at least one section.')
  if (remaining.length > 100) throw new Error('The clip limit is 100. Remove clips first.')
  return remaining
}

/** Source-time spans absent from every clip, for displaying cuts on the waveform. */
export function removedSourceRanges(clips:SourceClip[], duration:number): SourceClip[] {
  if (!clips.length) return []
  const sorted=clips.map(c => ({start:Math.max(0,c.start),end:Math.min(duration,c.end)})).filter(c => c.end > c.start).sort((a,b) => a.start-b.start)
  const removed:SourceClip[]=[]
  let cursor=0
  for (const clip of sorted) {
    if (clip.start > cursor) removed.push({start:cursor,end:clip.start})
    cursor=Math.max(cursor,clip.end)
  }
  if (cursor < duration) removed.push({start:cursor,end:duration})
  return removed
}
