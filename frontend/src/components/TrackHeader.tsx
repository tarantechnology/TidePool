import { Tip } from './Tip'
import { useState } from 'react'
import { EditableName } from './EditableName'
import { DeleteLayer } from './DeleteLayer'
import type { CSSProperties, ReactNode } from 'react'
import { refinementBadge } from '../api/refine'
import { tagLine } from '../api/backend'
import { gainToPos, posToGain, toDb } from '../edits'
import type { EditAction, FxChip, Stem, StemKind, TrackState } from '../types'

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
  /** Preview this layer without one of its effects. */
  onRemoveFx?: (chip: FxChip) => void
  /** Rename this layer; absent when it cannot be renamed. */
  onRename?: (name: string) => Promise<void> | void
  /** Delete this layer (fold: its sound goes into the residual); absent when it cannot be deleted. */
  onDelete?: (fold: boolean) => Promise<void>
  /** The residual layer's name, when there is one to fold into. */
  residualLabel?: string | null
  /** The residual itself can't be deleted. */
  residualId?: string | null
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

/** The effects rendered into a layer, as small chips; clicking one offers to preview it removed. */
function FxChips({ chips, label, onRemove }: { chips: FxChip[]; label: string; onRemove?: (chip: FxChip) => void }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <div className="mt-1 flex flex-wrap items-center gap-1" aria-label={`Effects on ${label}`}>
      {chips.map((chip) => (
        <span key={chip.id} className="inline-flex items-center gap-0.5">
          <button
            type="button"
            title={chip.baked ? `${chip.label} (baked in by a later edit; can no longer be removed)` : chip.scope === 'mix' ? `${chip.label} (whole mix)` : chip.label}
            aria-expanded={open === chip.id}
            disabled={!onRemove || chip.baked}
            onClick={() => setOpen(open === chip.id ? null : chip.id)}
            className="max-w-[7.5rem] truncate rounded-full border border-line bg-tint/[0.03] px-1.5 py-px text-[10px] text-muted transition-colors hover:border-line-strong hover:text-fg disabled:hover:border-line"
          >
            {chip.label}{chip.baked ? ' · baked' : ''}
          </button>
          {open === chip.id && onRemove && !chip.baked && (
            <button
              type="button"
              aria-label={`Remove ${chip.label} from ${label}`}
              onClick={() => { setOpen(null); onRemove(chip) }}
              className="rounded-full px-1 text-[10px] text-accent hover:underline"
            >
              Remove
            </button>
          )}
        </span>
      ))}
    </div>
  )
}

export function TrackHeader({ stem, track, audible, onAction, onKindChange, unsynced, compact = false, onRename, onDelete, residualLabel = null, residualId = null, onRemoveFx, clipActions }: Props) {
  const btn = (active: boolean, activeCls: string) =>
    `h-6 w-6 rounded-md border font-mono text-[11px] font-medium transition-colors ${
      active ? activeCls : 'border-line bg-tint/[0.03] text-subtle hover:border-line-strong hover:text-fg'
    }`
  const refined = refinementBadge(stem.refinement)
  const pos = `${gainToPos(track.gain) * 100}%`
  const sliderStyle = {
    '--pos': pos,
    '--fill': audible ? '#2b7bd6' : '#b5c3d4',
  } as CSSProperties

  return (
    <div className={`relative z-30 has-[[role=dialog]]:z-40 flex min-h-0 flex-col justify-center border-r border-line px-4 max-md:px-2.5 max-md:gap-0 ${compact ? 'gap-0' : 'gap-2.5'}`}>
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-baseline truncate text-[13px] font-medium tracking-tight" title={stem.tags?.length ? `Heard: ${stem.tags.join(', ')}` : undefined}>
            <EditableName value={stem.label} onRename={onRename} className="text-[13px] font-medium tracking-tight" />
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
          {!compact && stem.fx?.length ? <FxChips chips={stem.fx} label={stem.label} onRemove={onRemoveFx} /> : null}
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
        {onDelete && <DeleteLayer label={stem.label} residualLabel={residualLabel} isResidual={stem.id === residualId} onDelete={onDelete} />}
      </div>
      <div className={`flex items-center gap-2.5 max-md:gap-1 ${compact ? 'hidden max-md:flex' : ''}`}>
        <div className="relative flex-1">
          <div className="pointer-events-none absolute top-1/2 left-1/2 h-2 w-px -translate-y-1/2 bg-tint/25" />
          <input
            type="range"
            aria-label={`Volume ${stem.label}`}
            min={0}
            max={1}
            step={0.001}
            value={gainToPos(track.gain)}
            title="Volume (double-click to reset)"
            style={sliderStyle}
            className="fader w-full"
            onChange={(e) =>
              onAction({ type: 'set_gain', stemId: stem.id, value: Math.round(posToGain(Number(e.target.value)) * 100) / 100 }, `gain:${stem.id}`)
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
