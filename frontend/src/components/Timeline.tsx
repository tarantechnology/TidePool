import { useEffect, useRef, useState } from 'react'
import type { CSSProperties, MouseEvent, PointerEvent } from 'react'
import { hasStepKeyframes } from '../api/backend'
import type { AudioEngine } from '../audio/engine'
import { isAudible, MAX_GAIN, MAX_KEYFRAMES, spliceAutomation, stemStart } from '../edits'
import { useEngineTime } from '../hooks/useEngineTime'
import { addKeyframe, moveKeyframe, pointToKeyframe, removeKeyframe } from '../keyframes'
import type { EditAction, Keyframe, Stem, StemKind, TrackState } from '../types'
import { TrackHeader } from './TrackHeader'
import { Waveform } from './Waveform'
import { FEATURES } from '../features'
import { removedSourceRanges } from '../state/clipSequence'
import type { SourceClip } from '../state/clipSequence'
import { groupActions, timelineRows } from '../state/effectGroups'

const ACCENT = '#5b9be3'
const COLS = 'grid grid-cols-[var(--hdr)_minmax(0,1fr)]'
const LANE_MIN = 44
const LANE_MAX = 260
const clampLane = (h: number) => Math.min(LANE_MAX, Math.max(LANE_MIN, h))
/** Below this a lane drops its volume row. */
const COMPACT_BELOW = 88

interface Props {
  engine: AudioEngine
  stems: Stem[]
  tracks: TrackState[]
  duration: number
  onAction: (a: EditAction, coalesceKey?: string) => void
  onActions: (actions: EditAction[], coalesceKey?: string) => void
  onClipEdit: (stemId:string,operation:'move'|'delete',startOffset?:number)=>void
  onKindChange: (stemId: string, kind: StemKind) => void
  /** Edits are going to a backend, so a curve it would reject gets a marker. */
  syncing: boolean
  canTrim: boolean
  onKeepRange: (start:number, end:number) => string
  onSplitRange: (start:number, end:number) => string
  onCutRange: (start:number, end:number) => string
  exportClips: SourceClip[]
  canUndoClip: boolean
  canRedoClip: boolean
  onUndoClip: () => void
  onRedoClip: () => void
}

function tickStep(duration: number): number {
  if (duration <= 30) return 2
  if (duration <= 120) return 10
  return 30
}

function Playhead({ engine, duration }: { engine: AudioEngine; duration: number }) {
  const t = useEngineTime(engine)
  const pct = duration > 0 ? (t / duration) * 100 : 0
  return (
    <>
      {/* Dim the part of the timeline that has not played yet. */}
      <div
        className="pointer-events-none absolute right-0 bottom-0 top-8 bg-tint/[0.06]"
        style={{ left: `${pct}%` }}
      />
      <div
        className="pointer-events-none absolute top-0 bottom-0 w-px bg-accent shadow-[0_0_12px_2px_rgb(43_123_214/0.5)]"
        style={{ left: `${pct}%` }}
      >
        <div className="absolute top-1 left-1/2 -translate-x-1/2 rounded bg-accent px-1.5 py-0.5 font-mono text-[10px] font-medium text-white">
          {t.toFixed(1)}
        </div>
      </div>
    </>
  )
}

interface MarkerProps {
  keyframe: Keyframe
  duration: number
  onDrag: (clientX: number, clientY: number, lane: DOMRect) => void
  onDelete: () => void
}

// Sits above the scrub layer so it receives the pointer first. The 16px box is
// only a hit target around the visible 8px dot.
function KeyframeMarker({ keyframe, duration, onDrag, onDelete }: MarkerProps) {
  const lane = useRef<DOMRect | null>(null)
  return (
    <div
      className="absolute z-20 flex h-4 w-4 -translate-x-1/2 -translate-y-1/2 cursor-grab touch-none items-center justify-center active:cursor-grabbing"
      style={{
        left: `${(keyframe.t / duration) * 100}%`,
        top: `${(1 - keyframe.value / MAX_GAIN) * 100}%`,
      }}
      onPointerDown={(e) => {
        e.stopPropagation()
        // Right-click and macOS Ctrl-click open a menu that can swallow pointerup.
        if (e.button !== 0 || e.ctrlKey) return
        if (e.altKey) {
          onDelete()
          return
        }
        e.currentTarget.setPointerCapture(e.pointerId)
        lane.current = e.currentTarget.parentElement?.getBoundingClientRect() ?? null
      }}
      onPointerMove={(e) => {
        // The scrub layer seeks on any move with a button held.
        e.stopPropagation()
        if (lane.current) onDrag(e.clientX, e.clientY, lane.current)
      }}
      onPointerUp={() => {
        lane.current = null
      }}
      onPointerCancel={() => {
        lane.current = null
      }}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <div className="h-2 w-2 rounded-full border border-black bg-white" />
    </div>
  )
}

export function Timeline({ engine, stems, tracks, duration, onAction, onActions, onClipEdit, onKindChange, syncing, canTrim, onKeepRange, onSplitRange, onCutRange, exportClips, canUndoClip, canRedoClip, onUndoClip, onRedoClip }: Props) {
  const [limitMessage, setLimitMessage] = useState('')
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const [selecting, setSelecting] = useState(false)
  const [selection, setSelection] = useState<{stemId:string;start:number;end:number}|null>(null)
  const [menu, setMenu] = useState<{x:number;y:number}|null>(null)
  const draggingRange = useRef<{stemId:string;start:number}|null>(null)
  const previewFrame = useRef<number|null>(null)
  useEffect(() => () => { if (previewFrame.current !== null) cancelAnimationFrame(previewFrame.current) }, [])
  // Lane heights. By default the lanes share the timeline's height (fit). The size slider sets
  // every lane at once, and dragging a lane's bottom edge sizes just that lane.
  const scroller = useRef<HTMLDivElement>(null)
  const [fitHeight, setFitHeight] = useState(96)
  const [allHeight, setAllHeight] = useState<number | null>(null)
  const [laneSizes, setLaneSizes] = useState<Record<string, number>>({})
  const rows = timelineRows(stems, expanded)
  const laneCount = rows.length
  useEffect(() => {
    const el = scroller.current
    if (!el) return
    const fit = () => setFitHeight(Math.round(clampLane((el.clientHeight - 32) / Math.max(1, laneCount))))
    fit()
    const observer = new ResizeObserver(fit); observer.observe(el)
    return () => observer.disconnect()
  }, [laneCount])
  const laneHeight = (id: string) => laneSizes[id] ?? allHeight ?? fitHeight
  const fitted = allHeight === null && Object.keys(laneSizes).length === 0
  const resizeLane = (id: string, e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); e.stopPropagation()
    const startY = e.clientY, startH = laneHeight(id)
    const el = e.currentTarget; el.setPointerCapture(e.pointerId)
    const move = (ev: globalThis.PointerEvent) => setLaneSizes((s) => ({ ...s, [id]: Math.round(clampLane(startH + ev.clientY - startY)) }))
    const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up) }
    el.addEventListener('pointermove', move); el.addEventListener('pointerup', up)
  }
  const step = tickStep(duration)
  const majors: number[] = []
  for (let t = 0; t < duration; t += step) majors.push(t)
  const minors: number[] = []
  for (const t of majors) {
    for (let k = 1; k < 4; k++) {
      const m = t + (step * k) / 4
      if (m < duration) minors.push(m)
    }
  }

  const seekFrom = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    engine.seek(ratio * duration)
  }
  const timeFrom = (e: PointerEvent<HTMLDivElement>) => {
    const rect=e.currentTarget.getBoundingClientRect()
    return rect.width > 0 ? Math.min(duration,Math.max(0,(e.clientX-rect.left)/rect.width*duration)) : 0
  }
  const selectedTrack = selection && tracks.find(t => t.stemId === selection.stemId)
  const selectedStem = selection && stems.find(s => s.id === selection.stemId)
  const removed = removedSourceRanges(exportClips,duration)
  const validSelection = Boolean(selection && selectedTrack && selectedStem && selection.end-selection.start >= 0.01)
  const editSelection = (scale:number) => {
    if (!selection || !selectedTrack || !selectedStem || !validSelection) return
    const next=spliceAutomation(selectedTrack.automation,scale === 0 ? {level:0} : {scale},selection.start,selection.end,duration)
    if (next.length > MAX_KEYFRAMES) { setLimitMessage(`That edit exceeds the ${MAX_KEYFRAMES}-keyframe limit.`); return }
    onAction({type:'set_automation',stemId:selectedTrack.stemId,keyframes:next})
    setLimitMessage(`${scale === 0 ? 'Muted' : 'Lowered'} ${selectedStem.label} in the selected range. Use Undo to restore it.`)
  }
  const splitSelected = () => {
    setMenu(null)
    if (!selection) return
    try { setLimitMessage(onSplitRange(selection.start,selection.end)) } catch (e) { setLimitMessage(e instanceof Error ? e.message : String(e)) }
  }
  const cutSelected = () => {
    setMenu(null)
    if (!selection || !validSelection || !canTrim) return
    try { setLimitMessage(onCutRange(selection.start,selection.end)); setSelection(null) } catch (e) { setLimitMessage(e instanceof Error ? e.message : String(e)) }
  }
  useEffect(() => {
    const key=(event:KeyboardEvent) => {
      if (event.key === 'Escape') { setMenu(null); return }
      if ((event.key !== 'Delete' && event.key !== 'Backspace') || event.metaKey || event.ctrlKey || event.altKey || event.repeat || !selection || !validSelection || !canTrim) return
      const target=event.target as HTMLElement | null
      if (target?.isContentEditable || ['INPUT','TEXTAREA','SELECT'].includes(target?.tagName ?? '')) return
      event.preventDefault()
      cutSelected()
    }
    window.addEventListener('keydown',key)
    return () => window.removeEventListener('keydown',key)
  })
  const previewSelection = async () => {
    if (!selection || !validSelection) return
    if (previewFrame.current !== null) cancelAnimationFrame(previewFrame.current)
    engine.pause(); engine.seek(selection.start)
    try { await engine.play() } catch (e) { setLimitMessage(`Playback failed: ${String(e)}`); return }
    const stopAt=selection.end
    const tick=() => {
      if (!engine.isPlaying()) { previewFrame.current=null; return }
      if (engine.getTime() >= stopAt) { engine.pause(); engine.seek(stopAt); previewFrame.current=null; return }
      previewFrame.current=requestAnimationFrame(tick)
    }
    previewFrame.current=requestAnimationFrame(tick)
  }
  const previewSequence = async () => {
    if (!exportClips.length) return
    if (previewFrame.current !== null) cancelAnimationFrame(previewFrame.current)
    const sequence=exportClips.map(clip => ({...clip}))
    let index=0
    engine.pause(); engine.seek(sequence[0].start)
    try { await engine.play() } catch (e) { setLimitMessage(`Playback failed: ${String(e)}`); return }
    const tick=() => {
      if (!engine.isPlaying()) { previewFrame.current=null; return }
      const current=sequence[index], time=engine.getTime()
      if (time < current.start - .2) { previewFrame.current=null; return }
      if (time >= current.end) {
        index++
        if (index >= sequence.length) { engine.pause(); engine.seek(current.end); previewFrame.current=null; return }
        engine.seek(sequence[index].start)
      }
      previewFrame.current=requestAnimationFrame(tick)
    }
    previewFrame.current=requestAnimationFrame(tick)
  }

  const setAutomation = (stemId: string, keyframes: Keyframe[], coalesceKey?: string) =>
    onAction({ type: 'set_automation', stemId, keyframes }, coalesceKey)

  // The scrub layer captures the pointer, so double-clicks always land on it and
  // the lane has to be found from the pointer position.
  const addKeyframeAt = (e: MouseEvent<HTMLDivElement>) => {
    const laneEls = e.currentTarget.querySelectorAll<HTMLElement>('[data-lane]')
    for (const el of laneEls) {
      const rect = el.getBoundingClientRect()
      if (e.clientY < rect.top || e.clientY > rect.bottom) continue
      const track = tracks.find((t) => t.stemId === el.dataset.lane)
      if (!track) return
      const point = pointToKeyframe(e.clientX, e.clientY, rect, duration)
      const kf = track.automation.length === 0 ? { ...point, value: 1 } : point
      const next = addKeyframe(track.automation, kf, duration)
      if (next === track.automation && track.automation.length >= MAX_KEYFRAMES) {
        setLimitMessage(`Maximum ${MAX_KEYFRAMES} keyframes per stem. Remove one first.`)
        return
      }
      setLimitMessage('')
      setAutomation(track.stemId, next)
      return
    }
  }

  return (
    <div ref={scroller} className="flex min-h-0 flex-1 flex-col [--hdr:15rem] max-md:[--hdr:8.5rem] overflow-x-hidden overflow-y-auto">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-1.5 text-xs">
        <button type="button" aria-pressed={selecting} onClick={() => {setSelecting(!selecting);setLimitMessage('')}} className={`rounded border px-2 py-1 ${selecting ? 'border-accent bg-accent-soft text-accent' : 'border-line'}`}>Select range</button>
        <span className="text-muted">{validSelection && selection && selectedStem ? `${selectedStem.label}: ${selection.start.toFixed(2)}–${selection.end.toFixed(2)}s` : selecting ? 'Drag across a sound wave to select a range.' : 'Select a range to edit part of a sound.'}</span>
        {exportClips.length > 0 && <button type="button" className="rounded border border-line px-2 py-1" onClick={() => void previewSequence()}>Play edited sequence</button>}
        {(canUndoClip || canRedoClip) && <><button type="button" disabled={!canUndoClip} className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={onUndoClip}>Undo clip edit</button><button type="button" disabled={!canRedoClip} className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={onRedoClip}>Redo clip edit</button></>}
        {validSelection && <><button type="button" className="rounded border border-line px-2 py-1" onClick={() => void previewSelection()}>Play selection</button><button type="button" disabled={!canTrim} title="Remove this section from the entire video and close the gap" className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={cutSelected}>Delete selection</button><button type="button" disabled={!canTrim} title="Split the video and its audio at the selection boundaries" className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={splitSelected}>Split selection</button><button type="button" disabled={!canTrim} title={canTrim ? 'Keep this range when exporting' : 'Upload media first'} className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={() => {if (!selection) return; try {setLimitMessage(onKeepRange(selection.start,selection.end))} catch(e) {setLimitMessage(e instanceof Error ? e.message : String(e))}}}>Keep selection</button><button type="button" disabled={!syncing} title={syncing ? 'Mute this layer in the selection' : 'Separate a sound into project layers first so this edit can be exported'} className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={() => editSelection(0)}>Mute selection</button><button type="button" disabled={!syncing} title={syncing ? 'Lower this layer in the selection' : 'Separate a sound into project layers first so this edit can be exported'} className="rounded border border-line px-2 py-1 disabled:opacity-40" onClick={() => editSelection(0.3)}>Lower selection</button><button type="button" className="rounded border border-line px-2 py-1" onClick={() => {setSelection(null);setMenu(null)}}>Clear selection</button></>}
        {validSelection && !syncing && <span className="text-muted">Separate a sound first to mute or lower just that layer.</span>}
      </div>
      {limitMessage && <div role="status" className="px-4 py-1 text-xs text-amber-700">{limitMessage}</div>}
      <div className="relative min-h-full">
        {/* Ruler */}
        <div className={`${COLS} h-8 border-b border-line bg-tint/[0.02]`}>
          <div className="flex items-center gap-2 border-r border-line px-4 max-md:px-2">
            <span className="font-mono max-md:hidden text-[10px] uppercase tracking-wider text-subtle">{FEATURES.timeline}</span>
            <label className="ml-auto flex items-center gap-1.5 max-md:ml-0" title="Layer height">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" className="text-subtle" aria-hidden="true"><path d="M2 5h12M2 11h12" /></svg>
              <input type="range" aria-label="Layer height" min={LANE_MIN} max={LANE_MAX} step={2}
                value={allHeight ?? fitHeight} onChange={(e) => { setLaneSizes({}); setAllHeight(Number(e.target.value)) }}
                className="fader w-16 max-md:w-14" style={{ ['--pos' as string]: `${(((allHeight ?? fitHeight) - LANE_MIN) / (LANE_MAX - LANE_MIN)) * 100}%` } as CSSProperties} />
            </label>
            <button type="button" disabled={fitted} onClick={() => { setLaneSizes({}); setAllHeight(null) }}
              title="Fit every layer to the timeline's height"
              className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-accent transition-colors hover:bg-accent-soft disabled:text-subtle disabled:hover:bg-transparent">Fit</button>
          </div>
          <div className="relative">
            {minors.map((t) => (
              <div
                key={`m${t}`}
                className="absolute bottom-0 h-1.5 border-l border-tint/10"
                style={{ left: `${(t / duration) * 100}%` }}
              />
            ))}
            {majors.map((t) => (
              <div
                key={t}
                className="absolute bottom-0 h-3 border-l border-tint/25 pl-1.5 font-mono text-[10px] leading-3 text-subtle"
                style={{ left: `${(t / duration) * 100}%` }}
              >
                <span className="absolute -top-4">{t}s</span>
              </div>
            ))}
          </div>
        </div>

        {/* Tracks */}
        {rows.map((row) => {
          if (row.type === 'group') {
            const members=row.stems.flatMap(s=>tracks.filter(t=>t.stemId===s.id))
            const ids=row.stems.map(s=>s.id), muted=members.length>0 && members.every(t=>t.muted), solo=members.length>0 && members.every(t=>t.solo)
            const gain=Math.max(0,...members.map(t=>t.gain))
            return <div key={row.id} className={`${COLS} relative border-b border-line`} style={{height:laneHeight(row.id)}}>
              <div className="relative z-30 flex min-h-0 flex-col justify-center gap-1 border-r border-line px-4">
                <div className="flex items-center gap-1"><button type="button" className="text-accent" aria-label={`${expanded.has(row.id)?'Collapse':'Expand'} ${row.label}`} aria-expanded={expanded.has(row.id)} onClick={()=>setExpanded(current=>{const next=new Set(current);if(next.has(row.id))next.delete(row.id);else next.add(row.id);return next})}>{expanded.has(row.id)?'▾':'▸'}</button><span className="min-w-0 flex-1 truncate text-sm font-medium">{row.label} · {row.stems.length}</span>
                  <button type="button" aria-label={`${muted?'Unmute':'Mute'} all ${row.label}`} aria-pressed={muted} className="rounded border border-line px-1 text-xs" onClick={()=>onActions(groupActions(tracks,ids,{type:'mute',value:!muted}))}>M</button>
                  <button type="button" aria-label={`${solo?'Unsolo':'Solo'} all ${row.label}`} aria-pressed={solo} className="rounded border border-line px-1 text-xs" onClick={()=>onActions(groupActions(tracks,ids,{type:'solo',value:!solo}))}>S</button></div>
                <label className="flex items-center gap-2 text-xs text-muted">Volume<input aria-label={`${row.label} group volume`} className="fader min-w-0 flex-1" type="range" min={0} max={MAX_GAIN} step={.01} value={gain} onChange={e=>onActions(groupActions(tracks,ids,{type:'gain',value:Number(e.target.value)}),`group-gain:${row.id}`)}/></label>
              </div>
              <div className="relative overflow-hidden bg-gradient-to-b from-tint/[0.02] to-transparent">{row.stems.map(stem=>{const peaks=engine.getPeaks(stem.id),track=tracks.find(t=>t.stemId===stem.id);if(!peaks||!track)return null;const interval=stem.soundInterval;const clipPath=interval && stem.duration>0 ? `inset(0 ${Math.max(0,100-100*interval.end/stem.duration)}% 0 ${Math.max(0,100*interval.start/stem.duration)}%)` : undefined;return <div key={stem.id} className="absolute inset-y-0" style={{left:`${(stemStart(stem)/duration)*100}%`,width:`${(stem.duration/duration)*100}%`,clipPath}}><Waveform peaks={peaks} color={ACCENT} dimmed={!isAudible(tracks,track)}/></div>})}</div>
            </div>
          }
          const stem = row.stem
          const track = tracks.find((t) => t.stemId === stem.id)
          const peaks = engine.getPeaks(stem.id)
          if (!track || !peaks) return null
          const audible = isAudible(tracks, track)
          return (
            <div key={stem.id} className={`${COLS} relative border-b border-line`} style={{ height: laneHeight(stem.id) }}>
              {/* Drag this layer's bottom edge to make it taller or shorter. */}
              <div role="separator" aria-orientation="horizontal" aria-label={`Resize ${stem.label}`} title="Drag to resize this layer, double-click to fit"
                onPointerDown={(e) => resizeLane(stem.id, e)} onDoubleClick={() => setLaneSizes((s) => { const next = { ...s }; delete next[stem.id]; return next })}
                className="group absolute inset-x-0 -bottom-1 z-20 flex h-2 cursor-row-resize items-center justify-center">
                <span className="h-0.5 w-full bg-accent/0 transition-colors group-hover:bg-accent/50" />
              </div>
              <TrackHeader
                stem={stem}
                track={track}
                audible={audible}
                onAction={onAction}
                onKindChange={(kind) => onKindChange(stem.id, kind)}
                unsynced={syncing && hasStepKeyframes(track.automation)}
                compact={laneHeight(stem.id) < COMPACT_BELOW}
                clipActions={stem.isEffectClip ? <span className="flex gap-1"><button type="button" aria-label={`Move ${stem.label} earlier`} title="Move 0.1 seconds earlier" onClick={()=>onClipEdit(stem.id,'move',Math.max(0,stemStart(stem)-.1))}>←</button><button type="button" aria-label={`Move ${stem.label} later`} title="Move 0.1 seconds later" onClick={()=>onClipEdit(stem.id,'move',Math.min(Math.max(0,duration-stem.duration),stemStart(stem)+.1))}>→</button><button type="button" aria-label={`Delete ${stem.label}`} title="Remove this effect clip" onClick={()=>onClipEdit(stem.id,'delete')}>×</button></span> : undefined}
              />
              <div className="relative overflow-hidden bg-gradient-to-b from-tint/[0.02] to-transparent">
                {majors.map((t) => (
                  <div
                    key={t}
                    className="absolute top-0 bottom-0 border-l border-tint/[0.04]"
                    style={{ left: `${(t / duration) * 100}%` }}
                  />
                ))}
                <div
                  className="absolute inset-y-0"
                  style={{
                    left: `${(stemStart(stem) / duration) * 100}%`,
                    width: `${(stem.duration / duration) * 100}%`,
                  }}
                >
                  <Waveform peaks={peaks} color={ACCENT} dimmed={!audible} />
                </div>
                {/* Read-only: only the backend writes these. */}
                {stem.muteRanges?.map(([a, b]) => (
                  <div
                    key={`${a}-${b}`}
                    data-mute-range
                    className="absolute inset-y-0 border-x border-line-strong bg-tint/[0.08]"
                    style={{
                      left: `${(Math.max(a, 0) / duration) * 100}%`,
                      width: `${((Math.min(b, duration) - Math.max(a, 0)) / duration) * 100}%`,
                    }}
                  />
                ))}
                {track.automation.length > 0 && (
                  <>
                    <div
                      className="absolute right-0 left-0 border-t border-dashed border-tint/20"
                      style={{ top: `${(1 - 1 / MAX_GAIN) * 100}%` }}
                    />
                    <svg
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                    >
                      <polyline
                        fill="none"
                        stroke="#0e2440"
                        strokeWidth={1.5}
                        vectorEffect="non-scaling-stroke"
                        points={track.automation
                          .map((k) => `${(k.t / duration) * 100},${(1 - k.value / MAX_GAIN) * 100}`)
                          .join(' ')}
                      />
                    </svg>
                  </>
                )}
              </div>
            </div>
          )
        })}

        {/* Scrub layer + playhead, aligned to the lane column (15rem header) */}
        <div
          className={`absolute top-0 right-0 bottom-0 left-[var(--hdr)] touch-none ${selecting ? 'cursor-crosshair' : 'cursor-col-resize'}`}
          onContextMenu={e => { const lane=(e.target as HTMLElement).closest<HTMLElement>('[data-lane]'); if (!validSelection || !selection || lane?.dataset.lane !== selection.stemId) return; e.preventDefault(); setMenu({x:e.clientX,y:e.clientY}) }}
          onPointerDown={(e) => {
            setMenu(null)
            if (e.button !== 0 || e.ctrlKey) return
            e.currentTarget.setPointerCapture(e.pointerId)
            const lane=(e.target as HTMLElement).closest<HTMLElement>('[data-lane]')
            if (selecting && lane?.dataset.lane) { const start=timeFrom(e); draggingRange.current={stemId:lane.dataset.lane,start}; setSelection({stemId:lane.dataset.lane,start,end:start}); return }
            seekFrom(e)
          }}
          onPointerMove={(e) => {
            if (!(e.buttons & 1)) return
            if (draggingRange.current) {const {stemId,start}=draggingRange.current, t=timeFrom(e); setSelection({stemId,start:Math.min(start,t),end:Math.max(start,t)}); return}
            seekFrom(e)
          }}
          onPointerUp={() => {draggingRange.current=null}}
          onPointerCancel={() => {draggingRange.current=null; setSelection(null)}}
          onDoubleClick={e => {if (!selecting) addKeyframeAt(e)}}
        >
          <Playhead engine={engine} duration={duration} />
          {/* Keyframe layer: one box per lane, matching the track rows above. */}
          <div className="absolute inset-x-0 top-8">
            {rows.map((row) => {
              if (row.type === 'group') return <div key={row.id} className="relative border-b border-transparent" style={{height:laneHeight(row.id)}} />
              const stem = row.stem
              const track = tracks.find((t) => t.stemId === stem.id)
              if (!track || !engine.getPeaks(stem.id)) return null
              return (
                <div key={stem.id} data-lane={stem.id} className="relative border-b border-transparent" style={{ height: laneHeight(stem.id) }}>
                  {removed.map((range,i) => <div key={`removed:${i}`} aria-hidden="true" className="pointer-events-none absolute inset-y-0 border-x border-red-500/60 bg-red-500/20" style={{left:`${range.start/duration*100}%`,width:`${(range.end-range.start)/duration*100}%`}} />)}
                  {exportClips.flatMap((clip,i) => [clip.start,clip.end].filter(t => t > 0 && t < duration).map((t,j) => <div key={`${i}:${j}`} aria-hidden="true" className="pointer-events-none absolute inset-y-0 border-l border-dashed border-amber-600" style={{left:`${t/duration*100}%`}} />))}
                  {selection?.stemId === stem.id && <div aria-hidden="true" className="pointer-events-none absolute inset-y-0 border-x-2 border-accent bg-accent/25" style={{left:`${selection.start/duration*100}%`,width:`${(selection.end-selection.start)/duration*100}%`}} />}
                  {track.automation.map((k, idx) => (
                    <KeyframeMarker
                      key={idx}
                      keyframe={k}
                      duration={duration}
                      onDrag={(x, y, lane) =>
                        setAutomation(
                          stem.id,
                          moveKeyframe(track.automation, idx, pointToKeyframe(x, y, lane, duration), duration),
                          `automation:${stem.id}`,
                        )
                      }
                      onDelete={() => setAutomation(stem.id, removeKeyframe(track.automation, idx))}
                    />
                  ))}
                </div>
              )
            })}
          </div>
        </div>
      </div>
      {menu && validSelection && <div role="menu" style={{position:'fixed',left:menu.x,top:menu.y,zIndex:100}} className="rounded-lg border border-line bg-surface p-1 shadow-lg text-xs"><button role="menuitem" type="button" disabled={!canTrim} className="block rounded px-3 py-2 hover:bg-accent-soft disabled:opacity-40" onClick={cutSelected}>Delete selection from video</button><button role="menuitem" type="button" disabled={!canTrim} className="block rounded px-3 py-2 hover:bg-accent-soft disabled:opacity-40" onClick={splitSelected}>Split selection</button><button role="menuitem" type="button" className="block rounded px-3 py-2 hover:bg-accent-soft" onClick={() => setMenu(null)}>Close</button></div>}
    </div>
  )
}
