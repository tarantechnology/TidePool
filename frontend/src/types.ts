// Shared contracts. Anything the team (audio pipeline, vision, backend) needs
// to agree on lives here: Stem comes in, EditList goes out to export.

export type StemKind = 'dialogue' | 'sfx' | 'music' | 'ambience'

/** One isolated audio layer, as produced by the audio pipeline. */
export interface Stem {
  id: string
  label: string
  kind: StemKind
  /** Fetchable audio URL (wav/mp3). Mock stems use blob URLs. */
  url: string
  /** Length in seconds. */
  duration: number
}

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
