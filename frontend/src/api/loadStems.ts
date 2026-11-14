import { describeStemFailures } from '../audio/engine'
import type { StemLoadFailure, StemLoadResult } from '../audio/engine'
import { initialTracks } from '../edits'
import type { Stem, TrackState } from '../types'
import type { StemSource } from './stemSources'

/** The part of AudioEngine a load needs, so this can be tested without an AudioContext. */
export interface StemLoader {
  replaceStems(stems: Stem[]): Promise<StemLoadResult | null>
}

export interface LoadedStems {
  stems: Stem[]
  tracks: TrackState[]
  /** Stems that are missing from the timeline because they would not decode. */
  failures: StemLoadFailure[]
}

/** What to show when part of a load failed. Empty when all of it worked. */
export function stemFailureMessage(failures: readonly StemLoadFailure[]): string {
  if (failures.length === 0) return ''
  return `Could not load ${describeStemFailures(failures)}. The other stems are loaded.`
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
  const result = await engine.replaceStems(sourceStems)
  if (!result || !isLatest()) return null
  const { stems, failures } = result
  // Saved state covers the stems the project has, which may include ones that
  // did not decode, so tracks follow what actually loaded.
  const loadedIds = new Set(stems.map((s) => s.id))
  const saved = source.savedTracks?.filter((t) => loadedIds.has(t.stemId))
  return { stems, tracks: saved ?? initialTracks(stems), failures }
}
