import { useRef, useState } from 'react'
import type { MouseEvent, PointerEvent } from 'react'
import { hasStepKeyframes } from '../api/backend'
import type { AudioEngine } from '../audio/engine'
import { isAudible, MAX_GAIN, MAX_KEYFRAMES, stemStart } from '../edits'
import { useEngineTime } from '../hooks/useEngineTime'
import { addKeyframe, moveKeyframe, pointToKeyframe, removeKeyframe } from '../keyframes'
import type { EditAction, Keyframe, Stem, StemKind, TrackState } from '../types'
import { TrackHeader } from './TrackHeader'
import { Waveform } from './Waveform'

const ACCENT = '#3dff8b'
const COLS = 'grid grid-cols-[15rem_minmax(0,1fr)]'

interface Props {
  engine: AudioEngine
  stems: Stem[]
  tracks: TrackState[]
  duration: number
  onAction: (a: EditAction, coalesceKey?: string) => void
  onKindChange: (stemId: string, kind: StemKind) => void
  /** Edits are going to a backend, so a curve it would reject gets a marker. */
  syncing: boolean
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
        className="pointer-events-none absolute right-0 bottom-0 top-8 bg-black/40"
        style={{ left: `${pct}%` }}
      />
      <div
        className="pointer-events-none absolute top-0 bottom-0 w-px bg-accent shadow-[0_0_12px_2px_rgb(61_255_139/0.5)]"
        style={{ left: `${pct}%` }}
      >
        <div className="absolute top-1 left-1/2 -translate-x-1/2 rounded bg-accent px-1.5 py-0.5 font-mono text-[10px] font-medium text-black">
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

export function Timeline({ engine, stems, tracks, duration, onAction, onKindChange, syncing }: Props) {
  const [limitMessage, setLimitMessage] = useState('')
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
    <div className="flex min-h-0 flex-1 flex-col overflow-x-hidden overflow-y-auto">
      {limitMessage && <div role="status" className="px-4 py-1 text-xs text-amber-300">{limitMessage}</div>}
      <div className="relative min-h-full">
        {/* Ruler */}
        <div className={`${COLS} h-8 border-b border-line bg-white/[0.02]`}>
          <div className="flex items-center border-r border-line px-4 font-mono text-[10px] uppercase tracking-wider text-zinc-500">
            Layers
          </div>
          <div className="relative">
            {minors.map((t) => (
              <div
                key={`m${t}`}
                className="absolute bottom-0 h-1.5 border-l border-white/10"
                style={{ left: `${(t / duration) * 100}%` }}
              />
            ))}
            {majors.map((t) => (
              <div
                key={t}
                className="absolute bottom-0 h-3 border-l border-white/25 pl-1.5 font-mono text-[10px] leading-3 text-zinc-500"
                style={{ left: `${(t / duration) * 100}%` }}
              >
                <span className="absolute -top-4">{t}s</span>
              </div>
            ))}
          </div>
        </div>

        {/* Tracks */}
        {stems.map((stem) => {
          const track = tracks.find((t) => t.stemId === stem.id)
          const peaks = engine.getPeaks(stem.id)
          if (!track || !peaks) return null
          const audible = isAudible(tracks, track)
          return (
            <div key={stem.id} className={`${COLS} h-24 border-b border-line`}>
              <TrackHeader
                stem={stem}
                track={track}
                audible={audible}
                onAction={onAction}
                onKindChange={(kind) => onKindChange(stem.id, kind)}
                unsynced={syncing && hasStepKeyframes(track.automation)}
              />
              <div className="relative overflow-hidden bg-gradient-to-b from-white/[0.02] to-transparent">
                {majors.map((t) => (
                  <div
                    key={t}
                    className="absolute top-0 bottom-0 border-l border-white/[0.04]"
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
                    className="absolute inset-y-0 border-x border-line-strong bg-black/50"
                    style={{
                      left: `${(Math.max(a, 0) / duration) * 100}%`,
                      width: `${((Math.min(b, duration) - Math.max(a, 0)) / duration) * 100}%`,
                    }}
                  />
                ))}
                {track.automation.length > 0 && (
                  <>
                    <div
                      className="absolute right-0 left-0 border-t border-dashed border-white/20"
                      style={{ top: `${(1 - 1 / MAX_GAIN) * 100}%` }}
                    />
                    <svg
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                    >
                      <polyline
                        fill="none"
                        stroke="white"
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
          className="absolute top-0 right-0 bottom-0 left-60 cursor-col-resize touch-none"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId)
            seekFrom(e)
          }}
          onPointerMove={(e) => {
            if (e.buttons & 1) seekFrom(e)
          }}
          onDoubleClick={addKeyframeAt}
        >
          <Playhead engine={engine} duration={duration} />
          {/* Keyframe layer: one box per lane, matching the track rows above. */}
          <div className="absolute inset-x-0 top-8">
            {stems.map((stem) => {
              const track = tracks.find((t) => t.stemId === stem.id)
              if (!track || !engine.getPeaks(stem.id)) return null
              return (
                <div key={stem.id} data-lane={stem.id} className="relative h-24 border-b border-transparent">
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
    </div>
  )
}
