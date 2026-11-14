import { useRef } from 'react'

export type ProcessingRangeValue = {start:number;end:number}

/** A processing range is independent from model anchors and layer-edit automation. */
export function ProcessingRange({duration,value,onChange}:{duration:number;value:ProcessingRangeValue|null;onChange:(value:ProcessingRangeValue|null)=>void}) {
  const drag=useRef<number|null>(null)
  const at=(event:React.PointerEvent<HTMLDivElement>)=>{const r=event.currentTarget.getBoundingClientRect();return Math.max(0,Math.min(duration,(event.clientX-r.left)/r.width*duration))}
  return <div className="space-y-2 text-xs" aria-label="Processing range">
    <div className="flex items-center gap-2"><button type="button" className="rounded border border-line px-2 py-1" onClick={()=>onChange(value ? null : {start:0,end:Math.min(duration,2)})}>{value?'Use whole audio':'Select audio range'}</button>{value && <span className="text-muted">Only this range is edited.</span>}</div>
    {value && <>
      <div aria-label="Processing range timeline" className="relative h-9 touch-none rounded border border-line bg-surface-2" onPointerDown={e=>{drag.current=at(e);e.currentTarget.setPointerCapture(e.pointerId)}} onPointerMove={e=>{if(drag.current!==null){const t=at(e);onChange({start:Math.min(t,drag.current),end:Math.max(t,drag.current)})}}} onPointerUp={()=>{drag.current=null}} onPointerCancel={()=>{drag.current=null}}>
        <div className="pointer-events-none absolute inset-y-0 bg-accent/40" style={{left:`${value.start/duration*100}%`,width:`${Math.max(0,value.end-value.start)/duration*100}%`}}/><span className="pointer-events-none relative p-1">Drag to choose the section to edit</span>
      </div>
      <div className="flex gap-2">{(['start','end'] as const).map(key=><label key={key} className="min-w-0">{key}<input aria-label={`Processing ${key}`} type="number" min={0} max={duration} step="0.01" className="w-full rounded border border-line bg-surface-2 p-1" value={value[key]} onChange={e=>onChange({...value,[key]:Number(e.target.value)})}/></label>)}</div>
      <p className="text-muted">Audio outside this range stays unchanged. Longer ranges are processed in windows.</p>
    </>}
  </div>
}
