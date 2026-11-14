import { Tip } from './Tip'
import type { CSSProperties, ReactNode } from 'react'
import { refinementBadge } from '../api/refine'
import { tagLine } from '../api/backend'
import { MAX_GAIN, toDb } from '../edits'
import type { EditAction, Stem, StemKind, TrackState } from '../types'

interface Props {
  stem: Stem
  track: TrackState
  audible: boolean
  onAction: (a: EditAction, coalesceKey?: string) => void
  onKindChange: (kind: StemKind) => void
  /** This stem's automation is not on the backend. */
  unsynced: boolean
  /** A short lane: keep the name and M/S, drop the volume row. */
  compact?: boolean
  clipActions?: ReactNode
}

const KIND_LABEL: Record<Stem['kind'], string> = {
  dialogue: 'Dialogue',
  sfx: 'SFX',
  music: 'Music',
  ambience: 'Ambience',
  residual: 'Residual',
}
const KINDS = Object.keys(KIND_LABEL) as StemKind[]

export function TrackHeader({ stem, track, audible, onAction, onKindChange, unsynced, compact = false, clipActions }: Props) {
  const btn = (active: boolean, activeCls: string) =>
    `h-6 w-6 rounded-md border font-mono text-[11px] font-medium transition-colors ${
      active ? activeCls : 'border-line bg-tint/[0.03] text-subtle hover:border-line-strong hover:text-fg'
    }`
  const refined = refinementBadge(stem.refinement)
  const pos = `${(track.gain / MAX_GAIN) * 100}%`
  const sliderStyle = {
    '--pos': pos,
    '--fill': audible ? '#2b7bd6' : '#b5c3d4',
  } as CSSProperties

  return (
    <div className={`relative z-30 flex min-h-0 flex-col justify-center border-r border-line px-4 max-md:px-2.5 ${compact ? 'gap-0' : 'gap-2.5'}`}>
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium tracking-tight" title={stem.tags?.length ? `Heard: ${stem.tags.join(', ')}` : undefined}>
            {stem.label}
            {tagLine(stem.label, stem.tags) && <span className="ml-1.5 text-[11px] font-normal text-subtle">{tagLine(stem.label, stem.tags)}</span>}
          </div>
          <div className="flex items-center gap-1.5">
            <select
              title="Stem kind"
              value={stem.kind}
              onChange={(e) => {
                onKindChange(e.target.value as StemKind)
                // Hand the keyboard back, so Space plays again.
                e.target.blur()
              }}
              className="-ml-0.5 min-w-0 cursor-pointer bg-transparent font-mono text-[10px] uppercase tracking-wider text-subtle outline-none transition-colors hover:text-fg"
            >
              {KINDS.map((kind) => (
                <option key={kind} value={kind} className="bg-surface text-fg">
                  {KIND_LABEL[kind]}
                </option>
              ))}
            </select>
            {unsynced && (
              <span
                title="This curve has a hard step, which the backend cannot store, so its export keeps the previous curve."
                className="shrink-0 font-mono text-[10px] text-amber-700"
              >
                not synced
              </span>
            )}
            {refined && (
              <span title={refined.title} className="shrink-0 font-mono text-[10px] text-accent">
                {refined.text}
              </span>
            )}
          </div>
        </div>
        {clipActions}
        <Tip side="bottom" label={track.muted ? 'Unmute' : 'Mute'} detail={track.muted ? 'Bring this layer back into the mix.' : 'Silence this layer. Everything else keeps playing.'}>
          <button
            aria-label={track.muted ? `Unmute ${stem.label}` : `Mute ${stem.label}`}
            aria-pressed={track.muted}
            className={btn(track.muted, 'border-red-200 bg-red-50 text-red-600')}
            onClick={() => onAction({ type: 'set_mute', stemId: stem.id, muted: !track.muted })}
          >
            M
          </button>
        </Tip>
        <Tip side="bottom" label={track.solo ? 'Unsolo' : 'Solo'} detail={track.solo ? 'Hear the full mix again.' : 'Hear only this layer (and any other soloed ones).'}>
          <button
            aria-label={track.solo ? `Unsolo ${stem.label}` : `Solo ${stem.label}`}
            aria-pressed={track.solo}
            className={btn(track.solo, 'border-amber-300 bg-amber-50 text-amber-700')}
            onClick={() => onAction({ type: 'set_solo', stemId: stem.id, solo: !track.solo })}
          >
            S
          </button>
        </Tip>
      </div>
      <div className={`flex items-center gap-2.5 max-md:hidden ${compact ? 'hidden' : ''}`}>
        <div className="relative flex-1">
          <div className="pointer-events-none absolute top-1/2 left-1/2 h-2 w-px -translate-y-1/2 bg-tint/25" />
          <input
            type="range"
            min={0}
            max={MAX_GAIN}
            step={0.01}
            value={track.gain}
            title="Volume (double-click to reset)"
            style={sliderStyle}
            className="fader w-full"
            onChange={(e) =>
              onAction({ type: 'set_gain', stemId: stem.id, value: Number(e.target.value) }, `gain:${stem.id}`)
            }
            onDoubleClick={() => onAction({ type: 'set_gain', stemId: stem.id, value: 1 })}
          />
        </div>
        <span className="w-14 text-right font-mono text-[10px] tabular-nums text-subtle">
          {toDb(track.gain)}
        </span>
      </div>
    </div>
  )
}
