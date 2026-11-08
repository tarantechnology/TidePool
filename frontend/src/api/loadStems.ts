import { initialTracks } from '../edits'
import type { Stem, TrackState } from '../types'
import type { StemSource } from './stemSources'

/** The part of AudioEngine a load needs, so this can be tested without an AudioContext. */
export interface StemLoader {
  replaceStems(stems: Stem[]): Promise<Stem[] | null>
}

export interface LoadedStems {
  stems: Stem[]
  tracks: TrackState[]
}

/**
 * Load a source into the engine and work out the editor's starting tracks.
 * Returns null when a newer load has taken over, which must leave the editor as
 * it is. Edits still queued for the backend land before saved state is read, or
 * the read would miss them and the editor would start behind the backend.
 */
export async function loadFromSource(
  source: StemSource,
  engine: StemLoader,
  isLatest: () => boolean,
  flush: () => Promise<void>,
): Promise<LoadedStems | null> {
  // Do not replace local edits until all pending writes have succeeded.
  await flush()
  if (!isLatest()) return null
  const sourceStems = await source.load()
  if (!isLatest()) return null
  const stems = await engine.replaceStems(sourceStems)
  if (!stems || !isLatest()) return null
  return { stems, tracks: source.savedTracks ?? initialTracks(stems) }
}
