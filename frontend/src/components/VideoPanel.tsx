import { useEffect, useMemo, useRef, useState } from 'react'
import type { AudioEngine } from '../audio/engine'
import type { VisualMetadata } from '../api/backend'
import { frameAt, selectVisual, objectSound } from '../visual'
import { objectTracks, overlayBoxes, readShowObjects, trackColor, writeShowObjects } from '../objects'
import type { VisualSelection } from '../visual'
import { useEngineTime } from '../hooks/useEngineTime'
import { MaskOverlay } from './MaskOverlay'
import { useVideoSync } from '../hooks/useVideoSync'
import { FEATURES } from '../features'
import { FeatureTag } from './FeatureTag'
import { fitRotated, rotateBox, turn } from '../rotation'
import type { Rotation } from '../rotation'

const ROTATE_BUTTON = 'flex h-6 w-6 items-center justify-center rounded-full text-muted transition-colors hover:bg-tint/[0.06] hover:text-fg'

function Corner({ cls }: { cls: string }) {
  return <span className={`absolute h-3 w-3 border-tint/25 ${cls}`} />
}

interface Props {
  engine: AudioEngine | null
  src: string | null
  onPick: (file: File) => void
  onDuration: (duration: number) => void
  /** Width over height of the loaded clip, so the layout can fit it. */
  onAspect?: (aspect: number) => void
  onError: (message: string) => void
  visual: VisualMetadata | null
  onSelection: (selection: VisualSelection) => void
  selection: VisualSelection | null
  onClearSelection: () => void
  onVisualAction: (selection:VisualSelection, description:string, action:'isolate'|'mute'|'lower') => Promise<void>
  /** Seek requests from outside, such as the object list. `n` makes a repeat of the same time count. */
  seekTo?: { time: number; n: number } | null
  /** Clockwise quarter turns to show the picture at. Boxes and masks turn with it. */
  rotation?: Rotation
  onRotate?: (rotation: Rotation) => void
  /** The project is an audio file (a song): there is no picture to show. */
  audioOnly?: boolean
}

/** Fixed bar heights for the audio-only placeholder; a shape, not the file's waveform. */
const BARS = Array.from({ length: 48 }, (_, i) => 10 + Math.round(30 * Math.abs(Math.sin(i * 0.55) * Math.cos(i * 0.21))))

export function VideoPanel({ engine, src, onPick, onDuration, onAspect, onError, visual, onSelection, selection, onClearSelection, onVisualAction, seekTo, rotation = 0, onRotate, audioOnly = false }: Props) {
  const t = useEngineTime(engine)
  const video = useRef<HTMLVideoElement>(null)
  const frame = useRef<HTMLDivElement>(null)
  // Boxes fade out while the picture swings round, then come back in their new places.
  const [turning, setTurning] = useState(false)
  // The angle the video is drawn at keeps counting past 360, so 270 to 0 turns one step forward, not three back.
  const [spin, setSpin] = useState<number>(rotation)
  const lastRotation = useRef<Rotation>(rotation)
  useEffect(() => {
    const step = ((rotation - lastRotation.current + 540) % 360) - 180
    lastRotation.current = rotation
    if (step === 0) return
    setSpin(a => a + step)
    setTurning(true)
    const timer = window.setTimeout(() => setTurning(false), 320)
    return () => window.clearTimeout(timer)
  }, [rotation])
  const picker = useRef<HTMLInputElement>(null)
  const [previewTime, setPreviewTime] = useState(0)
  const [sound, setSound] = useState('')
  const [soundSelection,setSoundSelection] = useState('')
  const selectionKey = selection ? `${selection.analysis_id}:${selection.track_id}` : ''
  const soundDescription = soundSelection === selectionKey ? sound : objectSound(selection?.label || '')
  const [actionPending, setActionPending] = useState(false)
  const [bounds, setBounds] = useState({width:0,height:0,videoWidth:0,videoHeight:0})
  const [showObjects, setShowObjects] = useState(readShowObjects)
  const automaticTracks = useMemo(() => visual?.discovery_mode === 'automatic'
    ? new Set(objectTracks(visual).map(track => track.trackId)) : null, [visual])
  useEffect(() => {
    if (!seekTo) return
    // With stems loaded the engine is the clock and the video follows it.
    if (engine?.duration) { engine.pause(); engine.seek(seekTo.time) }
    const el=video.current
    // Without stems the overlay follows the video's own timeupdate.
    if (el) { el.pause(); el.currentTime=seekTo.time }
  }, [seekTo, engine])
  useEffect(() => {
    const el=video.current, box=frame.current
    if (!el || !box) return
    // Measure the frame, not the video: a turned video is sized from these bounds.
    const update=()=>setBounds({width:box.clientWidth,height:box.clientHeight,videoWidth:el.videoWidth,videoHeight:el.videoHeight})
    const observer=new ResizeObserver(update); observer.observe(box); el.addEventListener('loadedmetadata',update); update()
    return () => { observer.disconnect(); el.removeEventListener('loadedmetadata',update) }
  }, [src])

  useVideoSync(engine?.duration ? engine : null, video, src, onError)

  const input = (
    <input
      ref={picker}
      type="file"
      accept="video/*,audio/*"
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
      className={src ? 'rounded-full bg-white/90 px-3 py-1 text-[11px] text-fg shadow-sm backdrop-blur hover:bg-white' : 'rounded-full bg-accent px-4 py-2 text-xs font-medium text-white shadow-sm transition-colors hover:bg-[#1b62b3]'}
    >
      {src ? 'Replace video' : audioOnly ? 'Replace file' : 'Choose a video or song'}
    </button>
  )

  const overlay = (() => {
    if (!visual || bounds.width <= 0 || bounds.videoWidth <= 0) return null
    const w=bounds.width, h=bounds.height
    // vw x vh is the picture as shown (turned); element places the unturned mask under it.
    const fit=fitRotated(w,h,bounds.videoWidth || visual.width,bounds.videoHeight || visual.height,rotation)
    const vw=fit.width, vh=fit.height, left=fit.left, top=fit.top
    const current=engine?.duration ? t : previewTime
    const visualFrame=frameAt(visual,current)
    const selectedTrack=selection?.track_id ?? null
    // With Show objects off, only the selected object stays visible.
    const boxes=overlayBoxes(visualFrame,selectedTrack).filter(b=>(!automaticTracks || automaticTracks.has(b.trackId)) && (showObjects || b.selected)).map(b=>({...b,box:rotateBox(b.box,rotation)}))
    return {w,h,vw,vh,left,top,current,selectedTrack,boxes,element:fit.element}
  })()
  // The action card sits beside the selected object when it is on screen, else bottom left.
  const popoverStyle = (() => {
    const b = overlay?.boxes.find(x=>x.selected)
    if (!overlay || !b) return { left: 12, bottom: 12 }
    const [x1,,x2,y2] = b.box
    const right = overlay.left + x2*overlay.vw
    const leftEdge = overlay.left + x1*overlay.vw
    const x = right + 268 < overlay.w ? right + 8 : Math.max(8, leftEdge - 264)
    const y = Math.min(Math.max(8, overlay.top + y2*overlay.vh - 180), Math.max(8, overlay.h - 200))
    return { left: x, top: y }
  })()

  // The unturned video, placed so that rotating it about its centre fills the frame. Native controls
  // would turn sideways with the picture, so a turned clip plays from the transport instead.
  const placed = bounds.width > 0 && bounds.videoWidth > 0 ? fitRotated(bounds.width,bounds.height,bounds.videoWidth,bounds.videoHeight,rotation).element : null
  const turned = placed ? { ...placed, transform: `rotate(${spin}deg)` } : null

  if (src) {
    return (
      <div ref={frame} className="shadow-card group relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-2xl border border-line bg-[#dce8f5]">
        <video
          ref={video}
          src={src}
          muted
          controls={!engine?.duration && !rotation}
          onTimeUpdate={e => setPreviewTime(e.currentTarget.currentTime)}
          playsInline
          preload="auto"
          className={turned ? 'absolute max-w-none object-fill transition-[transform,width,height,left,top] duration-300 ease-out' : 'h-full w-full object-contain'}
          style={turned ? turned : undefined}
          onLoadedMetadata={(e) => { const v = e.currentTarget; onDuration(v.duration); if (v.videoWidth && v.videoHeight) onAspect?.(v.videoWidth / v.videoHeight) }}
          onError={() => onError('This browser could not play the video.')}
        />
        {overlay && <div className={`pointer-events-none absolute inset-0 transition-opacity duration-150 ${turning ? 'opacity-0' : ''}`} aria-label="Detected objects">
          {overlay.selectedTrack !== null && overlay.boxes.filter(b=>b.selected).map(b=><MaskOverlay key={`mask-${b.object.detection_id}`} mask={b.object.mask} color={trackColor(b.trackId).rgb} style={overlay.element}/>)}
          {overlay.boxes.map(b => {
            const [x1,y1,x2,y2]=b.box
            return <button key={b.object.detection_id} type="button" title={`Select ${b.label}`} aria-label={`Select ${b.label}`} aria-pressed={b.selected}
              className={`pointer-events-auto absolute rounded-md transition-[box-shadow,opacity,background-color] duration-200 ${b.selected ? 'border-[3px] [animation:box-pulse_1.8s_ease-in-out_infinite]' : 'border-2 opacity-75 hover:bg-white/10 hover:opacity-100 hover:shadow-[0_0_0_4px_rgb(140_196_242/0.55),0_0_24px_rgb(140_196_242/0.6)]'}`}
              style={{left:overlay.left+x1*overlay.vw,top:overlay.top+y1*overlay.vh,width:(x2-x1)*overlay.vw,height:(y2-y1)*overlay.vh,borderColor:b.color}}
              onClick={() => {const selected=selectVisual(visual!,overlay.current,b.object); if(selected) { video.current?.pause(); engine?.pause(); setSound(objectSound(b.object.label)); setSoundSelection(`${selected.analysis_id}:${selected.track_id}`); onSelection(selected) }}}>
              <span className={`absolute -top-5 left-[-2px] whitespace-nowrap rounded-md px-1.5 text-[10px] leading-4 font-medium text-white shadow-sm ${b.selected ? '' : 'opacity-90'}`} style={{background:b.color}}>{b.label}</span>
            </button>
          })}
        </div>}
        {visual && <label className="absolute top-2 left-2 z-10 flex cursor-pointer items-center gap-1.5 rounded-full bg-white/90 px-2.5 py-1 text-[11px] text-fg shadow-sm backdrop-blur">
          <input type="checkbox" className="accent-accent" checked={showObjects} onChange={e => { setShowObjects(e.target.checked); writeShowObjects(e.target.checked) }} />Show objects
        </label>}
        {visual && !selection && showObjects && <div className="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2 rounded-full bg-white/90 px-3 py-1 text-[11px] text-fg shadow-sm backdrop-blur">{FEATURES.visual}: click an object, then choose its sound</div>}
        {selection && <div className="rise-in shadow-card absolute z-20 max-h-[90%] w-64 max-w-[calc(100%-1rem)] overflow-y-auto space-y-2 rounded-xl border border-line bg-surface p-3 text-xs text-fg" style={popoverStyle} aria-label="Selected object actions">
          <div className="flex items-center justify-between gap-2"><p className="flex items-center gap-1.5 font-semibold">{selection.label}<FeatureTag name={FEATURES.visual} /></p>{!visual?.selection_only && <span className="text-[10px] text-subtle">{selection.interval.start_seconds.toFixed(1)}–{selection.interval.end_seconds.toFixed(1)}s</span>}<button type="button" aria-label="Close object actions" className="px-1 text-muted" onClick={onClearSelection}>×</button></div>
          <label className="block text-muted">Sound to edit<input aria-label="Selected object sound" className="mt-1 w-full rounded-lg border border-line bg-surface px-2 py-1.5 text-fg focus:border-accent/50 focus:outline-none focus:ring-2 focus:ring-accent/15" value={soundDescription} onChange={e=>{setSound(e.target.value);setSoundSelection(selectionKey)}} placeholder={objectSound(selection.label) ? 'e.g. dog barking' : 'e.g. a man imitating a dog barking'} /></label>
          {!objectSound(selection.label) && <p className="text-[10px] text-subtle">Describe what you hear this person doing. Their outline does not identify their voice.</p>}
          <div className="grid grid-cols-3 gap-1">{(['isolate','mute','lower'] as const).map(action=><button key={action} disabled={actionPending || !soundDescription.trim()} className={`rounded-lg px-2 py-1.5 font-medium capitalize transition-colors disabled:opacity-40 ${action==='isolate' ? 'bg-accent text-white hover:bg-[#1b62b3]' : 'border border-line text-fg hover:bg-surface-2'}`} onClick={()=>{setActionPending(true);void onVisualAction(selection,soundDescription,action).catch(e=>onError(e instanceof Error ? e.message : String(e))).finally(()=>setActionPending(false))}}>{action}</button>)}</div>
          <p className="text-[10px] text-subtle">{visual?.selection_only ? 'Edits matching sounds throughout the clip. Similar sounds may also be affected.' : 'Seeing an object does not guarantee it makes a sound.'}</p>
        </div>}
        {input}
        <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5">
          <div className="opacity-0 transition-opacity group-hover:opacity-100">{pickButton}</div>
          {onRotate && <div className="flex items-center rounded-full bg-white/90 p-0.5 text-fg shadow-sm backdrop-blur" role="group" aria-label="Rotate video">
            <button type="button" aria-label="Rotate left" title="Rotate left" onClick={() => onRotate(turn(rotation, -1))} className={ROTATE_BUTTON}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 4v5h5" /><path d="M4.6 9A8 8 0 1 1 4 12" /></svg>
            </button>
            {rotation !== 0 && <button type="button" title="Reset rotation" aria-label={`Rotated ${rotation} degrees. Reset rotation`} onClick={() => onRotate(0)} className="rounded-full px-1 font-mono text-[10px] text-accent hover:text-fg">{rotation}°</button>}
            <button type="button" aria-label="Rotate right" title="Rotate right" onClick={() => onRotate(turn(rotation, 1))} className={ROTATE_BUTTON}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 4v5h-5" /><path d="M19.4 9A8 8 0 1 0 20 12" /></svg>
            </button>
          </div>}
        </div>
      </div>
    )
  }

  if (audioOnly) {
    const played = engine?.duration ? t / engine.duration : 0
    return (
      <div className="shadow-card group relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-2xl border border-line bg-surface">
        <div className="bg-grid fade-edges absolute inset-0 opacity-70" />
        <div className="relative flex flex-col items-center px-4 text-center">
          <div className="flex h-12 items-center gap-1" aria-hidden="true">
            {BARS.map((h, i) => (
              <span key={i} className={`w-1 rounded-full transition-colors ${i / BARS.length < played ? 'bg-accent' : 'bg-sky'}`} style={{ height: h }} />
            ))}
          </div>
          <div className="mt-3 text-sm font-medium text-fg">Audio only</div>
          <div className="mt-1 max-w-xs text-xs text-muted">There is no picture to show. Every track plays in the timeline below.</div>
        </div>
        {input}
        <div className="absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100">{pickButton}</div>
      </div>
    )
  }

  return (
    <div className="shadow-card relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-2xl border border-line bg-surface">
      <div className="bg-grid fade-edges absolute inset-0 opacity-70" />
      <div className="relative flex aspect-video h-[88%] max-w-[94%] flex-col items-center justify-center rounded-lg border border-line bg-gradient-to-b from-tint/[0.03] to-transparent">
        <Corner cls="top-2 left-2 border-t border-l" />
        <Corner cls="top-2 right-2 border-t border-r" />
        <Corner cls="bottom-2 left-2 border-b border-l" />
        <Corner cls="right-2 bottom-2 border-r border-b" />
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" className="text-accent" aria-hidden="true">
          <rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" strokeWidth="1.5" />
          <path d="M10 9.5v5l4-2.5-4-2.5z" fill="currentColor" />
        </svg>
        <div className="mt-3 text-sm font-medium text-fg">Start with a video or a song</div>
        <div className="mt-1 max-w-xs text-center text-xs text-muted">Drop a clip anywhere, then click what you want to hear. A song can be split into one track per instrument.</div>
        <div className="mt-4">{pickButton}</div>
        {input}
      </div>
    </div>
  )
}
