import type { CSSProperties } from 'react'
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
}

const KIND_LABEL: Record<Stem['kind'], string> = {
  dialogue: 'Dialogue',
  sfx: 'SFX',
  music: 'Music',
  ambience: 'Ambience',
  residual: 'Residual',
}
const KINDS = Object.keys(KIND_LABEL) as StemKind[]

export function TrackHeader({ stem, track, audible, onAction, onKindChange, unsynced }: Props) {
  const btn = (active: boolean, activeCls: string) =>
    `h-6 w-6 rounded-md border font-mono text-[11px] font-medium transition-colors ${
      active ? activeCls : 'border-line bg-white/[0.03] text-zinc-500 hover:border-line-strong hover:text-zinc-200'
    }`
  const pos = `${(track.gain / MAX_GAIN) * 100}%`
  const sliderStyle = {
    '--pos': pos,
    '--fill': audible ? '#3dff8b' : '#52525b',
  } as CSSProperties

  return (
    <div className="flex flex-col justify-center gap-2.5 border-r border-line px-4">
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium tracking-tight">{stem.label}</div>
          <div className="flex items-center gap-1.5">
            <select
              title="Stem kind"
              value={stem.kind}
              onChange={(e) => {
                onKindChange(e.target.value as StemKind)
                // Hand the keyboard back, so Space plays again.
                e.target.blur()
              }}
              className="-ml-0.5 min-w-0 cursor-pointer bg-transparent font-mono text-[10px] uppercase tracking-wider text-zinc-500 outline-none transition-colors hover:text-zinc-200"
            >
              {KINDS.map((kind) => (
                <option key={kind} value={kind} className="bg-ink-900 text-zinc-200">
                  {KIND_LABEL[kind]}
                </option>
              ))}
            </select>
            {unsynced && (
              <span
                title="This curve has a hard step, which the backend cannot store, so its export keeps the previous curve."
                className="shrink-0 font-mono text-[10px] text-amber-300"
              >
                not synced
              </span>
            )}
          </div>
        </div>
        <button
          title="Mute"
          className={btn(track.muted, 'border-red-400/40 bg-red-500/15 text-red-300')}
          onClick={() => onAction({ type: 'set_mute', stemId: stem.id, muted: !track.muted })}
        >
          M
        </button>
        <button
          title="Solo"
          className={btn(track.solo, 'border-amber-300/40 bg-amber-300/15 text-amber-200')}
          onClick={() => onAction({ type: 'set_solo', stemId: stem.id, solo: !track.solo })}
        >
          S
        </button>
      </div>
      <div className="flex items-center gap-2.5">
        <div className="relative flex-1">
          <div className="pointer-events-none absolute top-1/2 left-1/2 h-2 w-px -translate-y-1/2 bg-white/25" />
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
        <span className="w-14 text-right font-mono text-[10px] tabular-nums text-zinc-500">
          {toDb(track.gain)}
        </span>
      </div>
    </div>
  )
}
