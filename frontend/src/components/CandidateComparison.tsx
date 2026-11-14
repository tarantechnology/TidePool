import { useEffect, useRef, useState } from 'react'

const LABELS = { original: 'Original', edited: 'Edited mix', target: 'Just the sound' } as const

/** One transport, one decoded timeline: switch sources without loudness normalization. */
export function CandidateComparison({original,edited,target,showTarget=true,seekTo}:{original:string;edited:string;target:string;showTarget?:boolean;seekTo?:{time:number;id:number}|null}) {
  const originalRef=useRef<HTMLAudioElement>(null), editedRef=useRef<HTMLAudioElement>(null), targetRef=useRef<HTMLAudioElement>(null)
  const [mode,setMode]=useState<'original'|'edited'|'target'>('edited')
  const [playing,setPlaying]=useState(false)
  const [time,setTime]=useState(0)
  const [duration,setDuration]=useState(0)
  const refs={original:originalRef,edited:editedRef,target:targetRef}
  useEffect(()=>{if(!seekTo)return;[originalRef,editedRef,targetRef].forEach(ref=>{if(ref.current)ref.current.currentTime=seekTo.time});setTime(seekTo.time)},[seekTo])
  const switchTo=(next:typeof mode)=>{
    const old=refs[mode].current, audio=refs[next].current
    const t=old?.currentTime || 0;old?.pause()
    if(audio){audio.currentTime=t;if(playing)void audio.play().catch(()=>setPlaying(false))}
    setMode(next);setTime(t)
  }
  const toggle=()=>{const a=refs[mode].current;if(!a)return;if(playing){a.pause();setPlaying(false)}else{void a.play().then(()=>setPlaying(true)).catch(()=>setPlaying(false))}}
  const modes: Array<'original'|'edited'|'target'> = showTarget ? ['original','edited','target'] : ['original','edited']
  const pct = duration > 0 ? (time / duration) * 100 : 0
  return <div className="space-y-3 text-xs" aria-label="Separation preview">
    <div className={`grid ${showTarget ? 'grid-cols-3' : 'grid-cols-2'} gap-1 rounded-xl bg-surface-2 p-1`} role="group" aria-label="Listen to">
      {modes.map(key=><button key={key} aria-pressed={mode===key} onClick={()=>switchTo(key)}
        className={`rounded-lg px-2 py-1.5 font-medium transition-all ${mode===key ? 'bg-surface text-accent shadow-sm' : 'text-muted hover:text-fg'}`}>{LABELS[key]}</button>)}
    </div>
    {(['original','edited','target'] as const).map(key=><audio key={key} ref={refs[key]} preload="auto" onLoadedMetadata={e=>setDuration(e.currentTarget.duration)} src={{original,edited,target}[key]} onTimeUpdate={e=>{if(mode===key)setTime(e.currentTarget.currentTime)}} onEnded={()=>setPlaying(false)}/>)}
    <div className="flex items-center gap-3">
      <button aria-label={playing?'Pause preview':'Play preview'} onClick={toggle} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent text-white shadow-sm transition-colors hover:bg-[#1b62b3]">
        {playing
          ? <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>
          : <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z"/></svg>}
      </button>
      <input aria-label="Preview position" className="fader min-w-0 flex-1" style={{['--pos' as string]:`${pct}%`}} type="range" min={0} max={duration || 0} step="0.01" value={time} onChange={e=>{const t=Number(e.target.value);Object.values(refs).forEach(ref=>{if(ref.current)ref.current.currentTime=t});setTime(t)}}/>
      <span className="w-16 shrink-0 text-right font-mono text-[10px] tabular-nums text-subtle">{time.toFixed(1)} / {duration.toFixed(1)}s</span>
    </div>
    <p className="text-[11px] text-subtle">Switch while it plays. {showTarget ? 'Listen for anything missing from the background or leaking into the sound.' : 'Compare at the same volume and listen for unwanted changes.'}</p>
  </div>
}
