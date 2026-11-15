import {useEffect,useMemo,useRef,useState} from 'react'
import type {AudioEngine} from '../audio/engine'
import type {SourceClip} from '../state/clipSequence'
import {lengthOf,replaceTimeline,sliceTimeline,timelinePosition} from '../state/timeline'
import {useTimelineTime} from '../hooks/useTimelineTime'
import type {EditAction,Stem,StemKind,TrackState} from '../types'
import {isAudible,MAX_GAIN,stemStart} from '../edits'
import {groupActions,timelineRows} from '../state/effectGroups'
import {TrackHeader} from './TrackHeader'
import {Waveform} from './Waveform'

interface Props {
 engine:AudioEngine;stems:Stem[];tracks:TrackState[];clips:SourceClip[]
 onExport:()=>void;onChange:(clips:SourceClip[])=>void;onUndo:()=>void;onRedo:()=>void;canUndo:boolean;canRedo:boolean
 onAction:(action:EditAction,key?:string)=>void;onActions:(actions:EditAction[],key?:string)=>void;onKindChange:(id:string,kind:StemKind)=>void
 onClipEdit:(stemId:string,operation:'move'|'delete',startOffset?:number)=>void
}
const btn='rounded border border-line px-2 py-1 text-xs hover:bg-accent-soft disabled:opacity-40'
export function SequenceTimeline({engine,stems,tracks,clips:provided,onChange,onUndo,onRedo,canUndo,canRedo,onAction,onActions,onKindChange,onClipEdit,onExport}:Props){
 const clips=useMemo(()=>provided.length?provided:[{start:0,end:engine.duration}],[provided,engine])
 const duration=lengthOf(clips),time=useTimelineTime(engine)
 const [range,setRange]=useState<{start:number;end:number}|null>(null)
 const [tool,setTool]=useState<'select'|'range'>('range')
 const [error,setError]=useState('')
 const [menu,setMenu]=useState<{x:number;y:number}|null>(null)
 const [expanded,setExpanded]=useState<Set<string>>(()=>new Set())
 const rows=timelineRows(stems,expanded)
 const clipboard=useRef<SourceClip[]>([]),root=useRef<HTMLDivElement>(null)
 const drag=useRef<{start:number;x:number}|null>(null)
 const valid=range && range.end-range.start>=.01
 const change=(next:SourceClip[])=>{if(JSON.stringify(next)===JSON.stringify(clips))return;engine.pause();onChange(next);setRange(null);setMenu(null);setError('')}
 const attempt=(fn:()=>void)=>{try{fn()}catch(e){setError(e instanceof Error?e.message:String(e))}}
 const remove=()=>{if(valid&&range)change(replaceTimeline(clips,range.start,range.end))}
 const copy=()=>{if(valid&&range)clipboard.current=sliceTimeline(clips,range.start,range.end)}
 const paste=()=>{if(clipboard.current.length)change(replaceTimeline(clips,Math.min(time,duration),Math.min(time,duration),clipboard.current))}
 const split=()=>{const at=Math.min(time,duration);if(at<=0||at>=duration)return;change(replaceTimeline(clips,at,at))}
 const keep=()=>{if(valid&&range)change(sliceTimeline(clips,range.start,range.end))}
 const move=(delta:number)=>{if(!range)return;const i=timelinePosition(clips,range.start+.00001).index,j=i+delta;if(j<0||j>=clips.length)return;const next=[...clips];[next[i],next[j]]=[next[j],next[i]];change(next)}
 const command=(name:string)=>attempt(()=>{switch(name){case 'delete':remove();break;case 'copy':copy();setMenu(null);break;case 'cut':copy();remove();break;case 'paste':paste();break;case 'split':split();break;case 'keep':keep();break;case 'undo':engine.pause();onUndo();setRange(null);break;case 'redo':engine.pause();onRedo();setRange(null);break}root.current?.focus()})
 useEffect(()=>{
  const key=(e:KeyboardEvent)=>{
   const target=e.target as HTMLElement|null
   if(target?.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(target?.tagName??''))return
   // Timeline shortcuts only belong to this editor when it has focus.
   if(!root.current?.contains(target))return
   const mod=e.metaKey||e.ctrlKey,k=e.key.toLowerCase()
   let cmd:string|null=null
   if(mod){if(k==='e'){e.preventDefault();e.stopImmediatePropagation();onExport();return}else if(k==='z')cmd=e.shiftKey?'redo':'undo';else if(k==='y')cmd='redo';else if(k==='c')cmd='copy';else if(k==='x')cmd='cut';else if(k==='v')cmd='paste';else if(k==='b')cmd='split';else if(k==='a'){e.preventDefault();e.stopImmediatePropagation();setRange({start:0,end:duration});return}}
   else if(k==='delete'||k==='backspace')cmd='delete'
   else if(k==='r'||k==='a'){setTool(k==='r'?'range':'select');e.preventDefault();return}
   else if(k==='escape'){setRange(null);setMenu(null);return}
   else if(k==='arrowleft'||k==='arrowright'){e.preventDefault();if(e.altKey)attempt(()=>move(k==='arrowleft'?-1:1));else engine.seekTimeline(Math.min(duration,Math.max(0,time+(k==='arrowleft'?-1:1)*(e.shiftKey?1:1/30))));return}
   else if(k==='i'){setRange({start:time,end:range?.end&&range.end>time?range.end:duration});return}
   else if(k==='o'){setRange({start:range?.start&&range.start<time?range.start:0,end:time});return}
   if(cmd){e.preventDefault();e.stopImmediatePropagation();command(cmd)}
  }
  window.addEventListener('keydown',key,true);return()=>window.removeEventListener('keydown',key,true)
 })
 const layout=useMemo(()=>clips.map((c,index)=>({...c,index,offset:lengthOf(clips.slice(0,index)),size:c.end-c.start})),[clips])
 const waveSlices=useMemo(()=>new Map(stems.map(stem=>{const peaks=engine.getPeaks(stem.id);return [stem.id,layout.map(c=>{const a=Math.max(c.start,stemStart(stem)),b=Math.min(c.end,stemStart(stem)+stem.duration);return peaks?.slice(Math.floor(Math.max(0,a-stemStart(stem))/stem.duration*peaks.length),Math.ceil(Math.max(0,b-stemStart(stem))/stem.duration*peaks.length))??new Float32Array()})]})),[stems,layout,engine])
 const point=(e:React.PointerEvent<HTMLDivElement>)=>{const r=e.currentTarget.getBoundingClientRect();return Math.max(0,Math.min(duration,(e.clientX-r.left)/r.width*duration))}
 const selectClip=(at:number)=>{const item=layout[timelinePosition(clips,at).index];setRange({start:item.offset,end:item.offset+item.size})}
 const down=(e:React.PointerEvent<HTMLDivElement>)=>{if(e.button!==0)return;root.current?.focus();setMenu(null);e.currentTarget.setPointerCapture(e.pointerId);const at=point(e);drag.current={start:at,x:e.clientX};engine.seekTimeline(at);if(tool==='select')selectClip(at);else setRange({start:at,end:at})}
 const moving=(e:React.PointerEvent<HTMLDivElement>)=>{if(!drag.current||!(e.buttons&1)||tool!=='range')return;const at=point(e);setRange({start:Math.min(at,drag.current.start),end:Math.max(at,drag.current.start)})}
 const up=(e:React.PointerEvent<HTMLDivElement>)=>{if(drag.current&&Math.abs(e.clientX-drag.current.x)<3)selectClip(point(e));drag.current=null}
 const lane=(id:string,members:Stem[])=><div data-waveform-lane={id} className="relative h-20 cursor-crosshair touch-none" onPointerDown={down} onPointerMove={moving} onPointerUp={up} onPointerCancel={()=>{drag.current=null}}>
  {layout.map(c=><div key={c.index} className="absolute inset-y-0 border-r border-accent/60 bg-accent/5" style={{left:`${c.offset/duration*100}%`,width:`${c.size/duration*100}%`}}>{members.map(stem=>{const track=tracks.find(t=>t.stemId===stem.id);const a=Math.max(c.start,stemStart(stem)),b=Math.min(c.end,stemStart(stem)+stem.duration);const slice=waveSlices.get(stem.id)?.[c.index];if(!track||!slice||b<=a)return null;return <div key={stem.id} className="absolute inset-y-0" style={{left:`${(a-c.start)/c.size*100}%`,width:`${(b-a)/c.size*100}%`}}><Waveform peaks={slice} color="#5b9be3" dimmed={!isAudible(tracks,track)}/></div>})}</div>)}
  {range&&<div className="pointer-events-none absolute inset-y-0 border-x-2 border-amber-500 bg-amber-400/25" style={{left:`${range.start/duration*100}%`,width:`${(range.end-range.start)/duration*100}%`}}/>}<div className="pointer-events-none absolute inset-y-0 border-l-2 border-accent" style={{left:`${time/duration*100}%`}}/>
 </div>
 return <div ref={root} tabIndex={0} aria-label="Video editing timeline" className="flex min-h-0 flex-1 flex-col outline-none" onContextMenu={e=>{e.preventDefault();root.current?.focus();setMenu({x:Math.min(e.clientX,window.innerWidth-190),y:Math.min(e.clientY,window.innerHeight-230)})}}>
  <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-line p-2">
   <button className={btn} aria-pressed={tool==='select'} onClick={()=>{setTool('select');root.current?.focus()}}>Select (A)</button><button className={btn} aria-pressed={tool==='range'} onClick={()=>{setTool('range');root.current?.focus()}}>Range (R)</button>
   <button className={btn} disabled={!valid} onClick={()=>command('delete')}>Delete</button><button className={btn} onClick={()=>command('split')}>Split ⌘B</button><button className={btn} disabled={!valid} onClick={()=>command('keep')}>Trim to selection</button><button className={btn} disabled={!canUndo} onClick={()=>command('undo')}>Undo</button><button className={btn} disabled={!canRedo} onClick={()=>command('redo')}>Redo</button>
   <button className={btn} onClick={onExport}>Export ⌘E</button>
   <span className="ml-auto font-mono text-xs text-muted">{time.toFixed(2)} / {duration.toFixed(2)}s{valid&&range?` · Selected ${range.start.toFixed(2)}–${range.end.toFixed(2)}s`:''}</span>
  </div>
  <div className="px-3 py-1 text-[10px] text-muted">Space play/pause · ⌘/Ctrl C/X/V copy/cut/paste · ⌘B split · ⌘Z undo · ⇧⌘Z redo · I/O range · Alt+←/→ move clip</div>
  {error&&<p role="status" className="px-3 text-xs text-amber-700">{error}</p>}
  <div className="min-h-0 flex-1 overflow-auto">
   <div className="grid grid-cols-[10rem_minmax(0,1fr)]"><div className="border-r border-line p-2 text-xs">Video + audio</div><div className="relative h-12 cursor-crosshair touch-none bg-tint/[0.03]" onPointerDown={down} onPointerMove={moving} onPointerUp={up} onPointerCancel={()=>{drag.current=null}}>
    {Array.from({length:Math.ceil(duration/(duration>60?10:2))},(_,i)=>i*(duration>60?10:2)).map(t=><span key={t} className="pointer-events-none absolute top-0 border-l border-line pl-1 font-mono text-[9px]" style={{left:`${t/duration*100}%`}}>{t}s</span>)}
    {layout.map(c=><div key={c.index} data-timeline-clip={c.index} className="absolute top-4 bottom-0 overflow-hidden border-r border-accent bg-accent/10 px-1 text-[10px]" style={{left:`${c.offset/duration*100}%`,width:`${c.size/duration*100}%`}}>{c.index+1} · {c.size.toFixed(2)}s</div>)}
    {range&&<div className="pointer-events-none absolute inset-y-0 border-x-2 border-amber-500 bg-amber-400/25" style={{left:`${range.start/duration*100}%`,width:`${(range.end-range.start)/duration*100}%`}}/>}<div className="pointer-events-none absolute inset-y-0 border-l-2 border-accent" style={{left:`${time/duration*100}%`}}/>
   </div></div>
   {rows.map(row=>{
    if(row.type==='group'){const members=row.stems.flatMap(s=>tracks.filter(t=>t.stemId===s.id)),ids=row.stems.map(s=>s.id),muted=members.length>0&&members.every(t=>t.muted),solo=members.length>0&&members.every(t=>t.solo),gain=Math.max(0,...members.map(t=>t.gain)),open=expanded.has(row.id)
     return <div key={row.id} className="grid grid-cols-[10rem_minmax(0,1fr)] border-t border-line"><div className="flex min-h-0 flex-col justify-center gap-1 border-r border-line px-2">
      <div className="flex items-center gap-1"><button type="button" className="text-accent" aria-label={`${open?'Collapse':'Expand'} ${row.label}`} aria-expanded={open} onClick={()=>setExpanded(current=>{const next=new Set(current);if(next.has(row.id))next.delete(row.id);else next.add(row.id);return next})}>{open?'▾':'▸'}</button><span className="min-w-0 flex-1 truncate text-sm font-medium">{row.label} · {row.stems.length}</span>
       <button type="button" aria-label={`${muted?'Unmute':'Mute'} all ${row.label}`} aria-pressed={muted} className="rounded border border-line px-1 text-xs" onClick={()=>onActions(groupActions(tracks,ids,{type:'mute',value:!muted}))}>M</button>
       <button type="button" aria-label={`${solo?'Unsolo':'Solo'} all ${row.label}`} aria-pressed={solo} className="rounded border border-line px-1 text-xs" onClick={()=>onActions(groupActions(tracks,ids,{type:'solo',value:!solo}))}>S</button></div>
      <label className="flex items-center gap-2 text-xs text-muted">Volume<input aria-label={`${row.label} group volume`} className="fader min-w-0 flex-1" type="range" min={0} max={MAX_GAIN} step={.01} value={gain} onChange={e=>onActions(groupActions(tracks,ids,{type:'gain',value:Number(e.target.value)}),`group-gain:${row.id}`)}/></label></div>
      {lane(row.id,row.stems)}</div>}
    const stem=row.stem,track=tracks.find(t=>t.stemId===stem.id);if(!track||!engine.getPeaks(stem.id))return null
    return <div key={stem.id} className="grid grid-cols-[10rem_minmax(0,1fr)] border-t border-line"><TrackHeader stem={stem} track={track} audible={isAudible(tracks,track)} onAction={onAction} onKindChange={kind=>onKindChange(stem.id,kind)} unsynced={false} compact
     clipActions={stem.isEffectClip?<span className="flex gap-1"><button type="button" aria-label={`Move ${stem.label} earlier`} title="Move 0.1 seconds earlier" onClick={()=>onClipEdit(stem.id,'move',Math.max(0,stemStart(stem)-.1))}>←</button><button type="button" aria-label={`Move ${stem.label} later`} title="Move 0.1 seconds later" onClick={()=>onClipEdit(stem.id,'move',Math.min(Math.max(0,engine.duration-stem.duration),stemStart(stem)+.1))}>→</button><button type="button" aria-label={`Delete ${stem.label}`} title="Remove this effect clip" onClick={()=>onClipEdit(stem.id,'delete')}>×</button></span>:undefined}/>
     {lane(stem.id,[stem])}</div>})}
  </div>
  {menu&&<div role="menu" className="fixed z-50 rounded-lg border border-line bg-surface p-1 shadow-lg" style={{left:menu.x,top:menu.y}}>{[['copy','Copy'],['cut','Cut'],['paste','Paste at playhead'],['delete','Delete and close gap'],['split','Split at playhead'],['keep','Trim to selection']].map(([cmd,label])=><button key={cmd} role="menuitem" className="block w-full rounded px-3 py-1 text-left text-xs hover:bg-accent-soft" onClick={()=>command(cmd)}>{label}</button>)}<button className="block px-3 py-1 text-xs" onClick={()=>setMenu(null)}>Close</button></div>}
 </div>
}
