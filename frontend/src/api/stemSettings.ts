import type { MuteRange } from '../types'
import type { SavedStem } from './backend'
import { isConflict } from './errors'

// Saving a stem's name, offset and remote mute ranges. The panel's stem list is
// a snapshot from "Load stem settings", and a remote command, another editor or
// a separation can move the stem on at any time, so the revision that goes with
// the write is read immediately before it, never taken from the snapshot.

export interface StemSettings {
  name: string
  start_offset: number
  mute_ranges?: MuteRange[]
}

/** The two calls this needs, so the sequence can be tested without a server. */
export interface StemSettingsIo {
  /** The project's stems as they are right now. */
  readStems: () => Promise<SavedStem[]>
  /** PATCH carrying `stem.revision` as the expected revision. */
  patch: (stem: SavedStem, settings: StemSettings) => Promise<void>
}

export type StemSettingsSave =
  | { ok: true; saved: SavedStem }
  /** Nothing was written. `stems` is the refreshed snapshot to show instead. */
  | { ok: false; message: string; stems: SavedStem[] }

export const GONE_MESSAGE = 'That stem is no longer part of the project. Load stem settings again.'
export const CONFLICT_MESSAGE =
  'This stem changed while you were editing it, so the settings were not saved. The values now shown are the ones the project holds.'

/**
 * Write stem settings against the revision the project holds now. A rejected
 * write (409) comes back as a message with a fresh snapshot, so the panel can
 * show what the project holds rather than keep offering a stale revision.
 */
export async function saveStemSettings(io: StemSettingsIo, stemId: string, settings: StemSettings): Promise<StemSettingsSave> {
  const current = await io.readStems()
  const fresh = current.find((s) => s.id === stemId)
  if (!fresh) return { ok: false, message: GONE_MESSAGE, stems: current }
  try {
    await io.patch(fresh, settings)
  } catch (err) {
    if (!isConflict(err)) throw err
    return { ok: false, message: CONFLICT_MESSAGE, stems: await io.readStems() }
  }
  return { ok: true, saved: fresh }
}
