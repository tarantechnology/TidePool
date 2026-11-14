// Shared contracts. Anything the team (audio pipeline, vision, backend) needs
// to agree on lives here: Stem comes in, EditList goes out to export.

/**
 * What part a stem plays. 'residual' is everything a separation did not pull
 * out, which the backend labels itself; the rest are picked by hand.
 */
export type StemKind = 'dialogue' | 'sfx' | 'music' | 'ambience' | 'residual'

/** One isolated audio layer, as produced by the audio pipeline. */
export interface Stem {
  id: string
  label: string
  kind: StemKind
  /** Fetchable audio URL (wav/mp3). Mock stems use blob URLs. */
  url: string
  /** Length in seconds. */
  duration: number
  /** Timeline position of the first sample, in seconds. Absent means 0. */
  startOffset?: number
  /** Timeline spans where the stem is silent, inclusive at both ends. Read-only here. */
  muteRanges?: MuteRange[]
  /** How many separation passes produced this stem, when the backend says. Read-only here. */
  refinement?: Refinement
  /** Instruments heard in a music stem, most likely first ("Piano", "Strings"). Read-only here. */
  tags?: string[]
  /** The canonical music label (`other`, `vocals`…) when the stem came from a music split. */
  canonicalLabel?: string
  /** Name of the track this one was separated out of ("Separate the rest"), for grouping. */
  group?: string
}

/**
 * Extra separation passes rerun the prompt on what was left, to catch sound the
 * first pass only quieted. Mapped from the backend's `refinement` by toStem.
 */
export interface Refinement {
  passesRequested: number
  passesRun: number
  /** Energy pulled out by each pass, in dB, in pass order. */
  perPassEnergyDb: number[]
  /** The backend stopped before passesRequested because a pass found little left. */
  stoppedEarly: boolean
}

/** [start, end] in timeline seconds, as the backend stores them. */
export type MuteRange = [number, number]

/** Gain multiplier at a point on the timeline. 1 = unchanged, 0 = silent. */
export interface Keyframe {
  /** Seconds from the start of the timeline. */
  t: number
  value: number
}

/** Editable state of one stem. Linear interpolation between keyframes. */
export interface TrackState {
  stemId: string
  /** Fader gain, 0..2 (1 = unity). */
  gain: number
  muted: boolean
  solo: boolean
  automation: Keyframe[]
}

export type EditAction =
  | { type: 'set_gain'; stemId: string; value: number }
  | { type: 'set_mute'; stemId: string; muted: boolean }
  | { type: 'set_solo'; stemId: string; solo: boolean }
  | { type: 'set_automation'; stemId: string; keyframes: Keyframe[] }
  | { type: 'clear_automation'; stemId: string }
  /**
   * Make a stem's gain follow how LOUD the source stems are (a proxy for
   * intensity, not emotion detection). Resolved to set_automation before it is
   * applied, because that needs decoded audio. minValue is the gain at low
   * loudness, maxValue at high loudness; maxValue below minValue ducks.
   */
  | { type: 'follow_intensity'; stemId: string; sourceStemIds: string[]; minValue?: number; maxValue?: number }

/** What the exporter (FFmpeg) should render. Same data the preview plays. */
export interface EditList {
  version: 1
  tracks: TrackState[]
}
