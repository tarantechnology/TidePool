import { useEffect, useRef, useState } from 'react'
import type { AudioEngine } from '../audio/engine'
import type { VisualMetadata } from '../api/backend'
import { frameAt, selectVisual } from '../visual'
import type { VisualSelection } from '../visual'
import { useEngineTime } from '../hooks/useEngineTime'
import { MaskOverlay } from './MaskOverlay'
import { useVideoSync } from '../hooks/useVideoSync'

function Corner({ cls }: { cls: string }) {
  return <span className={`absolute h-3 w-3 border-white/25 ${cls}`} />
}

interface Props {
  engine: AudioEngine | null
  src: string | null
  onPick: (file: File) => void
  onDuration: (duration: number) => void
  onError: (message: string) => void
  visual: VisualMetadata | null
  onSelection: (selection: VisualSelection) => void
  selection: VisualSelection | null
  onVisualAction: (selection:VisualSelection, description:string, action:'isolate'|'mute'|'lower') => Promise<void>
}

export function VideoPanel({ engine, src, onPick, onDuration, onError, visual, onSelection, selection, onVisualAction }: Props) {
  const t = useEngineTime(engine)
  const video = useRef<HTMLVideoElement>(null)
  const picker = useRef<HTMLInputElement>(null)
  const [previewTime, setPreviewTime] = useState(0)
  const [sound, setSound] = useState('')
  const [actionPending, setActionPending] = useState(false)
  const [bounds, setBounds] = useState({width:0,height:0,videoWidth:0,videoHeight:0})
  useEffect(() => {
    const el=video.current
    if (!el) return
    const update=()=>setBounds({width:el.clientWidth,height:el.clientHeight,videoWidth:el.videoWidth,videoHeight:el.videoHeight})
    const observer=new ResizeObserver(update); observer.observe(el); el.addEventListener('loadedmetadata',update); update()
    return () => { observer.disconnect(); el.removeEventListener('loadedmetadata',update) }
  }, [src])

  useVideoSync(engine?.duration ? engine : null, video, src, onError)

  const input = (
    <input
      ref={picker}
      type="file"
      accept="video/*"
      hidden
      onChange={(e) => {
        const picked = e.target.files?.[0]
        if (picked) onPick(picked)
        e.target.value = ''
      }}
    />
  )
  const pickButton = (
    <button
      onClick={() => picker.current?.click()}
      className="rounded-full border border-line px-2.5 py-1 font-mono text-[10px] text-zinc-400 transition-colors hover:border-line-strong hover:bg-white/[0.04] hover:text-zinc-100"
    >
      {src ? 'Replace video' : 'Load video'}
    </button>
  )

  if (src) {
    return (
      <div className="group relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-black">
        <video
          ref={video}
          src={src}
          muted
          controls={!engine?.duration}
          onTimeUpdate={e => setPreviewTime(e.currentTarget.currentTime)}
          playsInline
          preload="auto"
          className="h-full w-full object-contain"
          onLoadedMetadata={(e) => onDuration(e.currentTarget.duration)}
          onError={() => onError('This browser could not play the video.')}
        />
        {visual && bounds.width > 0 && bounds.videoWidth > 0 && (() => {
          const w=bounds.width, h=bounds.height
          const scale=Math.min(w/(bounds.videoWidth || visual.width), h/(bounds.videoHeight || visual.height))
          const vw=(bounds.videoWidth || visual.width)*scale, vh=(bounds.videoHeight || visual.height)*scale
          const current=engine?.duration ? t : previewTime
          const frame=frameAt(visual,current)
          return <div className="pointer-events-none absolute inset-0" aria-label="Detected objects">{selection && frame?.objects.filter(o=>o.track_id===selection.track_id && o.mask).map(o=><MaskOverlay key={o.detection_id} mask={o.mask} style={{left:(w-vw)/2,top:(h-vh)/2,width:vw,height:vh}}/>)}{frame?.objects.map(obj => {
            const [x1,y1,x2,y2]=obj.bbox_normalized
            if (obj.track_id === null || !obj.mask) return null
            return <button key={obj.detection_id} type="button" title={`Select ${obj.label}`} aria-label={`Select ${obj.label}`} className="pointer-events-auto absolute border-2 border-accent bg-accent/10 hover:bg-accent/30" style={{left:(w-vw)/2+x1*vw,top:(h-vh)/2+y1*vh,width:(x2-x1)*vw,height:(y2-y1)*vh}} onClick={() => {const selected=selectVisual(visual,current,obj); if(selected) { video.current?.pause(); engine?.pause(); setSound(''); onSelection(selected) }}}><span className="absolute top-0 left-0 bg-black/80 px-1 text-[10px] text-white">{obj.label}</span></button>
          })}</div>
        })()}
        {selection && <div className="absolute bottom-3 left-3 z-10 w-64 space-y-2 rounded-xl border border-line bg-ink-950/95 p-3 text-xs text-zinc-100 shadow-xl" aria-label="Selected object actions">
          <p className="text-amber-300">Experimental visual guidance. Visibility does not prove sound.</p><p>Selected: <strong>{selection.label}</strong> ({selection.interval.start_seconds.toFixed(1)}–{selection.interval.end_seconds.toFixed(1)}s)</p>
          <label className="block">Describe its sound<input aria-label="Selected object sound" className="mt-1 w-full rounded border border-line bg-black px-2 py-1" value={sound} onChange={e=>setSound(e.target.value)} /></label>
          <div className="flex gap-1">{(['isolate','mute','lower'] as const).map(action=><button key={action} disabled={actionPending || !sound.trim()} className="rounded border border-line px-2 py-1 capitalize disabled:opacity-40" onClick={()=>{setActionPending(true);void onVisualAction(selection,sound,action).catch(e=>onError(e instanceof Error ? e.message : String(e))).finally(()=>setActionPending(false))}}>{action === 'lower' ? 'Preview lower' : `Preview ${action}`} </button>)}</div>
        </div>}
        {input}
        <div className="absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100">{pickButton}</div>
      </div>
    )
  }

  return (
    <div className="relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-black">
      <div className="bg-grid fade-edges absolute inset-0 opacity-70" />
      <div className="relative flex aspect-video h-[88%] max-w-[94%] flex-col items-center justify-center rounded-lg border border-line bg-gradient-to-b from-white/[0.03] to-transparent">
        <Corner cls="top-2 left-2 border-t border-l" />
        <Corner cls="top-2 right-2 border-t border-r" />
        <Corner cls="bottom-2 left-2 border-b border-l" />
        <Corner cls="right-2 bottom-2 border-r border-b" />
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" className="text-zinc-600" aria-hidden="true">
          <rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" strokeWidth="1.5" />
          <path d="M10 9.5v5l4-2.5-4-2.5z" fill="currentColor" />
        </svg>
        <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-600">No video loaded</div>
        <div className="mt-1 font-mono text-3xl tabular-nums text-zinc-300">{t.toFixed(2)}</div>
        <div className="mt-2">{pickButton}</div>
        {input}
      </div>
    </div>
  )
}
