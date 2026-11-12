import { useRef, useState } from 'react'

/** One transport, one decoded timeline: switch sources without loudness normalization. */
export function CandidateComparison({original,edited,target}:{original:string;edited:string;target:string}) {
  const originalRef=useRef<HTMLAudioElement>(null), editedRef=useRef<HTMLAudioElement>(null), targetRef=useRef<HTMLAudioElement>(null)
  const [mode,setMode]=useState<'original'|'edited'|'target'>('edited')
  const [playing,setPlaying]=useState(false)
  const [time,setTime]=useState(0)
  const [duration,setDuration]=useState(0)
  const refs={original:originalRef,edited:editedRef,target:targetRef}
  const switchTo=(next:typeof mode)=>{
    const old=refs[mode].current, audio=refs[next].current
    const t=old?.currentTime || 0;old?.pause()
    if(audio){audio.currentTime=t;if(playing)void audio.play().catch(()=>setPlaying(false))}
    setMode(next);setTime(t)
  }
  return <div className="space-y-2 text-xs" aria-label="Separation preview">
    <p>Compare at the same time and gain. Listen for missing background and leakage.</p>
    <div className="flex flex-wrap gap-1">{(['original','edited','target'] as const).map(key=><button key={key} className="rounded border border-line px-2 py-1" aria-pressed={mode===key} onClick={()=>switchTo(key)}>{key==='original'?'Original mix':key==='edited'?'Edited mix':'Isolated target'}</button>)}</div>
    {(['original','edited','target'] as const).map(key=><audio key={key} ref={refs[key]} preload="auto" onLoadedMetadata={e=>setDuration(e.currentTarget.duration)} src={{original,edited,target}[key]} onTimeUpdate={e=>{if(mode===key)setTime(e.currentTarget.currentTime)}} onEnded={()=>setPlaying(false)}/>)}
    <button className="rounded border border-line px-2 py-1" onClick={()=>{const a=refs[mode].current;if(!a)return;if(playing){a.pause();setPlaying(false)}else{void a.play().then(()=>setPlaying(true)).catch(()=>setPlaying(false))}}}>{playing?'Pause preview':'Play preview'}</button>
    <input aria-label="Preview position" type="range" min={0} max={duration} step="0.01" value={time} onChange={e=>{const t=Number(e.target.value);Object.values(refs).forEach(ref=>{if(ref.current)ref.current.currentTime=t});setTime(t)}}/>
  </div>
}
