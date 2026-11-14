import type { CSSProperties } from 'react'
import { gainToPos, isAudible, posToGain, toDb } from '../edits'
import type { EditAction, Stem, TrackState } from '../types'
import { EditableName } from './EditableName'
import { DeleteLayer } from './DeleteLayer'

interface Props {
  stems: Stem[]
  tracks: TrackState[]
  onAction: (a: EditAction, coalesceKey?: string) => void
  onRename?: (stemId: string, name: string) => Promise<void>
  onDelete?: (stemId: string, fold: boolean) => Promise<void>
  residualLabel?: string | null
  /** The residual itself can't be deleted. */
  residualId?: string | null
}

/** One tall slider per layer: how loud each sound is in the mix. Same faders as the Tide chart. */
export function Mixer({ stems, tracks, onAction, onRename, onDelete, residualLabel = null, residualId = null }: Props) {
  return (
    <div className="flex min-h-0 flex-1 gap-3 overflow-auto p-4 pt-12" aria-label="Mixer">
      {stems.map((stem) => {
        const track = tracks.find((t) => t.stemId === stem.id)
        if (!track) return null
        const audible = isAudible(tracks, track)
        const percent = Math.round(track.gain * 100)
        const style = { '--pos': `${gainToPos(track.gain) * 100}%`, '--fill': audible ? '#2b7bd6' : '#b5c3d4' } as CSSProperties
        const toggle = (active: boolean, on: string) =>
          `h-7 flex-1 rounded-md border font-mono text-[11px] font-medium transition-colors ${active ? on : 'border-line bg-surface text-subtle hover:border-line-strong hover:text-fg'}`
        return (
          <div key={stem.id} className={`flex min-h-[280px] w-28 shrink-0 flex-col items-center gap-2 rounded-2xl border p-3 transition-colors ${audible ? 'border-line bg-surface-2/60' : 'border-line bg-surface-2/20 opacity-60'}`}>
            <EditableName value={stem.label} onRename={onRename ? (name) => onRename(stem.id, name) : undefined} wrap className="w-full text-center text-[13px] leading-tight font-medium tracking-tight" />
            <span className="font-mono text-[10px] uppercase tracking-wider text-subtle">{stem.kind}</span>
            <div className="flex min-h-0 w-full flex-1 items-stretch justify-center py-1">
              <input
                type="range"
                min={0}
                max={1}
                step={0.001}
                value={gainToPos(track.gain)}
                aria-label={`${stem.label} volume`}
                title="Volume (double-click to reset)"
                style={style}
                className="fader mixer-fader"
                onChange={(e) => onAction({ type: 'set_gain', stemId: stem.id, value: Math.round(posToGain(Number(e.target.value)) * 100) / 100 }, `gain:${stem.id}`)}
                onDoubleClick={() => onAction({ type: 'set_gain', stemId: stem.id, value: 1 })}
              />
            </div>
            <div className="text-center">
              <div className="font-mono text-base font-semibold tabular-nums text-fg">{percent}%</div>
              <div className="font-mono text-[10px] tabular-nums text-subtle">{toDb(track.gain)}</div>
            </div>
            {track.automation.length > 0 && <span className="text-center text-[10px] text-accent" title="This layer also has a volume curve on the Tide chart; the slider scales it.">has a curve</span>}
            <div className="flex w-full gap-1">
              <button aria-label={track.muted ? `Unmute ${stem.label}` : `Mute ${stem.label}`} aria-pressed={track.muted} className={toggle(track.muted, 'border-red-200 bg-red-50 text-red-600')} onClick={() => onAction({ type: 'set_mute', stemId: stem.id, muted: !track.muted })}>M</button>
              <button aria-label={track.solo ? `Unsolo ${stem.label}` : `Solo ${stem.label}`} aria-pressed={track.solo} className={toggle(track.solo, 'border-amber-300 bg-amber-50 text-amber-700')} onClick={() => onAction({ type: 'set_solo', stemId: stem.id, solo: !track.solo })}>S</button>
              {onDelete && <DeleteLayer label={stem.label} residualLabel={residualLabel} isResidual={stem.id === residualId} onDelete={(fold) => onDelete(stem.id, fold)} side="top" className="self-center" />}
            </div>
          </div>
        )
      })}
    </div>
  )
}
