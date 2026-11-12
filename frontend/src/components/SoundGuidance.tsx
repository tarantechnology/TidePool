import { useRef } from 'react'
import type { Anchor } from '../api/backend'

/** Source-timeline guidance only. The edit interval has its own controls. */
export function SoundGuidance({duration, anchors, onChange}: {duration:number;anchors:Anchor[];onChange:(anchors:Anchor[])=>void}) {
  const drag=useRef<number|null>(null)
  const positive=anchors.find(a=>a.token==='+')
  const negative=anchors.find(a=>a.token==='-')
  const update=(token:'+'|'-',start:number,end:number)=>onChange([...anchors.filter(a=>a.token!==token),{token,start,end}])
  const fields=(token:'+'|'-',a:Anchor)=> <div className="flex gap-2">{(['start','end'] as const).map(key=><label key={key} className="min-w-0">{key}<input aria-label={`${token==='+'?'Sound present':'Sound absent'} ${key}`} className="w-full rounded border border-line bg-black p-1" type="number" min={0} max={duration} step="0.01" value={a[key]} onChange={e=>update(token,key==='start'?Number(e.target.value):a.start,key==='end'?Number(e.target.value):a.end)}/></label>)}<button onClick={()=>onChange(anchors.filter(x=>x.token!==token))}>Clear</button></div>
  return <div className="space-y-2 text-xs">
    <button className="rounded border border-line px-2 py-1" onClick={()=>update('+',0,Math.min(duration,2))}>I hear it here</button>
    {positive && <>
      <div aria-label="Sound guidance timeline" className="relative h-9 touch-none rounded border border-line bg-black" onPointerDown={e=>{const rect=e.currentTarget.getBoundingClientRect();drag.current=Math.max(0,Math.min(duration,(e.clientX-rect.left)/rect.width*duration));e.currentTarget.setPointerCapture(e.pointerId)}} onPointerMove={e=>{if(drag.current===null)return;const rect=e.currentTarget.getBoundingClientRect();const t=Math.max(0,Math.min(duration,(e.clientX-rect.left)/rect.width*duration));update('+',Math.min(t,drag.current),Math.max(t,drag.current))}} onPointerUp={()=>{drag.current=null}} onPointerCancel={()=>{drag.current=null}}>
        <div className="pointer-events-none absolute inset-y-0 bg-accent/40" style={{left:`${positive.start/duration*100}%`,width:`${(positive.end-positive.start)/duration*100}%`}}/><span className="pointer-events-none relative p-1">Drag to mark where you hear it</span>
      </div>{fields('+',positive)}
      <p className="text-zinc-400">Guides overlapping model windows. This does not identify the same speaker throughout a long recording or define what gets muted.</p>
    </>}
    <details><summary>Advanced sound guidance</summary>{negative?fields('-',negative):<button onClick={()=>update('-',0,Math.min(duration,2))}>Mark where this sound is absent</button>}</details>
  </div>
}
