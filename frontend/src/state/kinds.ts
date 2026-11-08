import type { Stem, StemKind, TrackState } from '../types'

/** Kinds picked by hand, by stem id. Real stems have no kind of their own. */
export type KindOverrides = ReadonlyMap<string, StemKind>

/** What undo covers. */
export interface EditorState {
  tracks: TrackState[]
  kinds: KindOverrides
}

export function createEditorState(tracks: TrackState[]): EditorState {
  return { tracks, kinds: new Map() }
}

export function setKindOverride(kinds: KindOverrides, stemId: string, kind: StemKind): KindOverrides {
  return new Map(kinds).set(stemId, kind)
}

/** Stems with their overrides applied. Overrides for stems not in the list do nothing. */
export function mergeKinds(stems: Stem[], kinds: KindOverrides): Stem[] {
  return stems.map((stem) => {
    const kind = kinds.get(stem.id)
    return kind ? { ...stem, kind } : stem
  })
}
