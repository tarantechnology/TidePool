import { envelopeAt, isAudible, withMuteRanges } from '../edits'
import type { MuteRange, TrackState } from '../types'

// Making room. A layer pushed past 100% mostly drives the mix into the limiter,
// which holds its absolute level, so on its own a 500% voice ends up only about
// 12 dB over the rest. So while a boosted layer is actually making sound, every
// other layer steps back by the same factor: at level g the others play at 1/g.
// At 500% a voice then sits about 26 dB over the rest while it talks, and the
// scene comes back between its lines. The server export applies the same rule
// (app/room.py); keep the two in step.

/** Activity is measured in hops of this many seconds. */
export const ROOM_HOP = 0.01
const HOLD_HOPS = 25 // stay active 250 ms after the sound stops, so short pauses do not pump
const ATTACK_STEP = 0.5 // reach full room in 20 ms
const RELEASE_STEP = 0.04 // give it back over 250 ms
const SILENT_DB = -60
const FLOOR_DB = -45
const BELOW_LOUD_DB = 24

/** 0..1 per hop: how much this layer is sounding (1 = clearly sounding). */
export function activity(channels: Float32Array[], sampleRate: number): Float32Array {
  const hop = Math.max(1, Math.round(sampleRate * ROOM_HOP))
  const length = channels.length ? channels[0].length : 0
  const n = Math.floor(length / hop)
  const db = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    let sum = 0
    for (let i = k * hop; i < (k + 1) * hop; i++) {
      let mono = 0
      for (const c of channels) mono += c[i]
      mono /= channels.length
      sum += mono * mono
    }
    db[k] = 20 * Math.log10(Math.sqrt(sum / hop) + 1e-12)
  }
  const voiced = Array.from(db).filter((d) => d > SILENT_DB).sort((a, b) => a - b)
  const out = new Float32Array(n)
  if (!voiced.length) return out
  const threshold = Math.max(FLOOR_DB, voiced[Math.floor(0.95 * (voiced.length - 1))] - BELOW_LOUD_DB)
  let a = 0
  let lastOn = -Infinity
  for (let k = 0; k < n; k++) {
    if (db[k] > threshold) lastOn = k
    a = k - lastOn <= HOLD_HOPS ? Math.min(1, a + ATTACK_STEP) : Math.max(0, a - RELEASE_STEP)
    out[k] = a
  }
  return out
}

/** A layer's level at timeline time t: fader x volume curve, with mute ranges. */
export function levelAt(track: TrackState, muteRanges: MuteRange[] | undefined, t: number): number {
  return track.gain * envelopeAt(withMuteRanges(track.automation, muteRanges), t)
}

/** The highest level this layer ever reaches. */
export function peakLevel(track: TrackState): number {
  return track.gain * (track.automation.length ? Math.max(...track.automation.map((k) => k.value)) : 1)
}

/** Audible layers other than stemId that go past 100% at some point; only they make the others step back. */
export function boostersFor(tracks: TrackState[], stemId: string): TrackState[] {
  return tracks.filter((t) => t.stemId !== stemId && isAudible(tracks, t) && peakLevel(t) > 1)
}

export interface RoomSource {
  muteRanges?: MuteRange[]
  /** Timeline second where the layer's audio starts. */
  start: number
  activity: Float32Array
}

/** The factor a layer plays at, at timeline time t, to make room for the boosted layers (1 = untouched). */
export function roomAt(boosters: TrackState[], source: (stemId: string) => RoomSource | undefined, t: number): number {
  let push = 0
  for (const b of boosters) {
    const s = source(b.stemId)
    if (!s) continue
    const k = Math.floor((t - s.start) / ROOM_HOP + 1e-9)
    const a = k >= 0 && k < s.activity.length ? s.activity[k] : 0
    if (a > 0) push = Math.max(push, (levelAt(b, s.muteRanges, t) - 1) * a)
  }
  return 1 / (1 + push)
}
