import type { EditAction, Stem, TrackState } from '../types'

export type TimelineRow = { type: 'stem'; stem: Stem } | { type: 'group'; id: string; label: string; stems: Stem[] }

/** Only saved group identities join clips. Identical names alone are not evidence. */
export function timelineRows(stems: Stem[], expanded: ReadonlySet<string>): TimelineRow[] {
  const groups = new Map<string, Stem[]>()
  for (const stem of stems) if (stem.effectGroupId) groups.set(stem.effectGroupId, [...(groups.get(stem.effectGroupId) || []), stem])
  const emitted = new Set<string>(), rows: TimelineRow[] = []
  for (const stem of stems) {
    const id = stem.effectGroupId, members = id ? groups.get(id) : undefined
    if (!id || !members || members.length < 2) { rows.push({ type:'stem', stem }); continue }
    if (emitted.has(id)) continue
    emitted.add(id)
    rows.push({ type:'group', id, label:stem.effectLabel || stem.label, stems:members })
    if (expanded.has(id)) for (const member of members) rows.push({ type:'stem', stem:member })
  }
  return rows
}

/** One history entry can apply these together; gain preserves relative clip levels. */
export function groupActions(tracks: TrackState[], ids: readonly string[], change: { type:'gain'; value:number } | { type:'mute' | 'solo'; value:boolean }): EditAction[] {
  const members=tracks.filter(t=>ids.includes(t.stemId))
  const max=Math.max(0,...members.map(t=>t.gain))
  return members.map(t => {
    if (change.type === 'mute') return { type:'set_mute', stemId:t.stemId, muted:change.value }
    if (change.type === 'solo') return { type:'set_solo', stemId:t.stemId, solo:change.value }
    const gain=change as {type:'gain';value:number}
    return { type:'set_gain', stemId:t.stemId, value:Math.min(2,Math.max(0,max ? t.gain*gain.value/max : gain.value)) }
  })
}
