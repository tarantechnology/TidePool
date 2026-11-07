import type { EditAction, EditList, Keyframe, Stem, TrackState } from './types'

export const MAX_GAIN = 2

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

export function initialTracks(stems: Stem[]): TrackState[] {
  return stems.map((s) => ({
    stemId: s.id,
    gain: 1,
    muted: false,
    solo: false,
    automation: [],
  }))
}

export function toEditList(tracks: TrackState[]): EditList {
  return { version: 1, tracks }
}

function sortKeyframes(kfs: Keyframe[]): Keyframe[] {
  return [...kfs].sort((a, b) => a.t - b.t)
}

export function applyAction(tracks: TrackState[], action: EditAction): TrackState[] {
  return tracks.map((t) => {
    if (t.stemId !== action.stemId) return t
    switch (action.type) {
      case 'set_gain':
        return { ...t, gain: clamp(action.value, 0, MAX_GAIN) }
      case 'set_mute':
        return { ...t, muted: action.muted }
      case 'set_solo':
        return { ...t, solo: action.solo }
      case 'set_automation':
        return { ...t, automation: sortKeyframes(action.keyframes) }
      case 'clear_automation':
        return { ...t, automation: [] }
      case 'follow_intensity':
        // Needs decoded audio: src/audio/resolve.ts turns it into set_automation.
        return t
    }
  })
}

/** Automation multiplier at time t. Holds the first/last value outside the range. */
export function envelopeAt(keyframes: Keyframe[], t: number): number {
  if (keyframes.length === 0) return 1
  if (t <= keyframes[0].t) return keyframes[0].value
  const last = keyframes[keyframes.length - 1]
  if (t >= last.t) return last.value
  for (let i = 1; i < keyframes.length; i++) {
    const a = keyframes[i - 1]
    const b = keyframes[i]
    if (t <= b.t) {
      const span = b.t - a.t
      if (span <= 0) return b.value
      return a.value + ((b.value - a.value) * (t - a.t)) / span
    }
  }
  return last.value
}

/**
 * What to schedule when playback starts at t0: the gain now, then the ramps
 * after it. envelopeAt is left-continuous, so at an equal-time step it reads the
 * value before the step, which would ramp from the wrong level here.
 */
export function automationFrom(keyframes: Keyframe[], t0: number): { start: number; ramps: Keyframe[] } {
  const at = keyframes.filter((k) => k.t === t0)
  return {
    start: at.length > 0 ? at[at.length - 1].value : envelopeAt(keyframes, t0),
    ramps: keyframes.filter((k) => k.t > t0),
  }
}

export function isAudible(tracks: TrackState[], track: TrackState): boolean {
  if (track.muted) return false
  const anySolo = tracks.some((t) => t.solo)
  return !anySolo || track.solo
}

export function toDb(gain: number): string {
  if (gain <= 0) return '-inf'
  const db = 20 * Math.log10(gain)
  return `${db >= 0 ? '+' : ''}${db.toFixed(1)} dB`
}

// ---------------------------------------------------------------------------
// Prompt parsing. This is a STUB: a small regex parser so the prompt bar can
// drive the edit pipeline before the LLM agent exists. It is not the agent.
// ---------------------------------------------------------------------------

export interface ParseResult {
  actions: EditAction[]
  message: string
}

const RANGE_RE =
  /from\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*(?:to|-|until)\s*(\d+(?:\.\d+)?)/
const AT_RE = /\bat\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?/
const INTENSITY_RE = /\b(?:whenever|when|as|while)\b.*\b(?:argument|heated|intense|angry|louder)\b/
const LOUDER_RE = /\b(louder|increase|raise|boost|up)\b/
const QUIETER_RE = /\b(quieter|lower|reduce|decrease|softer|down|duck)\b/
/** N% or N percent. The sign is captured so "-20%" is rejected, not read as 20%. */
const PERCENT_RE = /(?<![\d.,])(-?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:%|percent\b)/g
/** Amounts with no exact meaning here. They get a question, never a guess. */
const UNSUPPORTED_QUANTITY_RE =
  /\d\s*(?:d\s?b|decibels?|x|times)\b|\bx\s*\d|\b(?:two|three|four|five|ten)\s+times\b|\b(?:twice|thrice|double|triple|half|halve|quarter)\b/
const MAX_PERCENT = 200

const ASK_PERCENT = 'Give a percentage, for example "to 30%" or "by 20%".'

type Quantity =
  | { kind: 'none' }
  | { kind: 'clarify'; message: string }
  | { kind: 'absolute'; percent: number }
  | { kind: 'relative'; percent: number; louder: boolean }

function parseQuantity(text: string): Quantity {
  const unsupported = UNSUPPORTED_QUANTITY_RE.exec(text)
  if (unsupported) return { kind: 'clarify', message: `"${unsupported[0].trim()}" cannot be applied exactly. ${ASK_PERCENT}` }
  const hits = [...text.matchAll(PERCENT_RE)]
  if (hits.length === 0) return { kind: 'none' }
  if (hits.length > 1) return { kind: 'clarify', message: `Found more than one percentage. ${ASK_PERCENT}` }
  const [hit] = hits
  const percent = Number(hit[1])
  if (percent < 0 || percent > MAX_PERCENT) {
    return { kind: 'clarify', message: `Percentages must be between 0 and ${MAX_PERCENT}, got ${hit[1]}%.` }
  }
  const before = text.slice(0, hit.index)
  const after = text.slice(hit.index + hit[0].length)
  const outside = `${before} ${after}`
  const up = LOUDER_RE.test(outside)
  const down = QUIETER_RE.test(outside)
  // "20% quieter" and "by 20%" are relative. "to 20%" and "20% volume" are levels.
  const comparative = /^\s*(louder|quieter|softer)\b/.exec(after)
  if (comparative) return { kind: 'relative', percent, louder: comparative[1] === 'louder' }
  if (/\bby\s*$/.test(before)) {
    if (up === down) return { kind: 'clarify', message: `Should "by ${percent}%" raise or lower it?` }
    return { kind: 'relative', percent, louder: up }
  }
  if (/\b(?:to|at)\s*$/.test(before) || /^\s*(?:volume|level|gain)\b/.test(after) || (!up && !down)) {
    return { kind: 'absolute', percent }
  }
  return { kind: 'clarify', message: `Did you mean "to ${percent}%" or "by ${percent}%"?` }
}

function rangeCurve(a: number, b: number, value: number): Keyframe[] {
  const ramp = Math.min(0.5, (b - a) / 4)
  return [
    { t: a, value: 1 },
    { t: a + ramp, value },
    { t: b - ramp, value },
    { t: b, value: 1 },
  ]
}

function percentEdit(
  quantity: Extract<Quantity, { percent: number }>,
  stem: Stem,
  track: TrackState,
  range: RegExpExecArray | null,
  at: RegExpExecArray | null,
): ParseResult {
  const { percent } = quantity
  const from = range ? Number(range[1]) : at ? Number(at[1]) : null
  const to = range ? Math.min(Number(range[2]), stem.duration) : stem.duration
  if (from !== null && to <= from) return { actions: [], message: 'End time must be after start time.' }
  const where = from === null ? '' : ` from ${from}s to ${to}s`

  if (quantity.kind === 'absolute') {
    const level = percent / 100
    if (from === null) {
      return {
        actions: [{ type: 'set_gain', stemId: stem.id, value: level }],
        message: `${stem.label} fader to ${percent}% (${toDb(level)}).`,
      }
    }
    // Automation multiplies the fader, so the curve value is level / fader.
    if (track.gain <= 0) {
      return { actions: [], message: `${stem.label}'s fader is at 0, so no curve can reach ${percent}%. Raise the fader first.` }
    }
    const multiplier = level / track.gain
    if (multiplier > MAX_GAIN) {
      return {
        actions: [],
        message: `${stem.label}'s fader is too low to reach ${percent}%: that needs a curve above ${MAX_GAIN * 100}%. Raise the fader first.`,
      }
    }
    return {
      actions: [{ type: 'set_automation', stemId: stem.id, keyframes: rangeCurve(from, to, multiplier) }],
      message: `${stem.label} at ${percent}%${where}.`,
    }
  }

  const factor = clamp(quantity.louder ? 1 + percent / 100 : 1 - percent / 100, 0, MAX_GAIN)
  const change = `${quantity.louder ? 'up' : 'down'} ${percent}%`
  if (from === null) {
    const next = clamp(track.gain * factor, 0, MAX_GAIN)
    return {
      actions: [{ type: 'set_gain', stemId: stem.id, value: next }],
      message: `${stem.label} ${change}, fader to ${toDb(next)}.`,
    }
  }
  return {
    actions: [{ type: 'set_automation', stemId: stem.id, keyframes: rangeCurve(from, to, factor) }],
    message: `${stem.label} ${change}${where}.`,
  }
}

export function parsePrompt(text: string, stems: Stem[], tracks: TrackState[]): ParseResult {
  const q = text.toLowerCase().trim()
  const stem = stems.find((s) => q.includes(s.label.toLowerCase()))
  if (!stem) {
    const names = stems.map((s) => s.label).join(', ')
    return { actions: [], message: `No stem matched. Mention one of: ${names}.` }
  }
  const track = tracks.find((t) => t.stemId === stem.id)
  if (!track) return { actions: [], message: `No track state for ${stem.label}.` }

  // Labels can hold digits, so they are out of the way before reading amounts.
  const rest = stems.reduce((text, s) => text.replaceAll(s.label.toLowerCase(), ' '), q)
  const quantity = parseQuantity(rest)
  if (quantity.kind === 'clarify') return { actions: [], message: quantity.message }

  // Percentages go first, so "at 50%" cannot be read as a time.
  const timing = q.replaceAll(PERCENT_RE, ' ')
  const range = RANGE_RE.exec(timing)
  const at = AT_RE.exec(timing)
  /** Digits nothing has claimed, which no branch below knows the meaning of. */
  const unexplainedNumber = (consumedAt: boolean) => {
    const left = rest.replaceAll(PERCENT_RE, ' ').replace(RANGE_RE, ' ')
    return /\d/.test(consumedAt ? left.replace(AT_RE, ' ') : left)
  }

  const louder = LOUDER_RE.test(q)
  const quieter = QUIETER_RE.test(q)
  const unmute = /\bunmute\b/.test(q)
  const mute = !unmute && /\bmute\b/.test(q)
  const unsolo = /\bunsolo\b/.test(q)
  const solo = !unsolo && /\bsolo\b/.test(q)

  if ((mute || unmute || solo || unsolo) && quantity.kind !== 'none') {
    return { actions: [], message: 'Mute and solo do not take a percentage. Use a time range, or ask for a volume change.' }
  }

  if (mute || unmute) {
    if (mute && (range || at)) {
      const from = range ? Number(range[1]) : Number(at![1])
      const to = Math.min(range ? Number(range[2]) : stem.duration, stem.duration)
      if (to <= from) return { actions: [], message: 'End time must be after start time.' }
      const e = 0.05
      return {
        actions: [
          {
            type: 'set_automation',
            stemId: stem.id,
            keyframes: [
              { t: Math.max(0, from - e), value: 1 },
              { t: from, value: 0 },
              { t: to, value: 0 },
              { t: Math.min(stem.duration, to + e), value: 1 },
            ],
          },
        ],
        message: `Muting ${stem.label} from ${from}s to ${to}s.`,
      }
    }
    return {
      actions: [{ type: 'set_mute', stemId: stem.id, muted: mute }],
      message: `${mute ? 'Muted' : 'Unmuted'} ${stem.label}.`,
    }
  }

  if (solo || unsolo) {
    return {
      actions: [{ type: 'set_solo', stemId: stem.id, solo }],
      message: `${solo ? 'Soloed' : 'Unsoloed'} ${stem.label}.`,
    }
  }

  if (quantity.kind !== 'none') {
    if (INTENSITY_RE.test(q) || unexplainedNumber(true)) {
      return { actions: [], message: `That percentage could not be read exactly. ${ASK_PERCENT}` }
    }
    return percentEdit(quantity, stem, track, range, at)
  }

  if ((louder || quieter) && unexplainedNumber(false)) {
    return { actions: [], message: `That amount or time could not be read exactly. ${ASK_PERCENT}` }
  }

  const intensity = range ? null : INTENSITY_RE.exec(q)
  if (intensity) {
    // The edit is in the main clause, either side of the condition: "make the
    // siren louder when they get angry" or "when they get louder, duck the music".
    const head = q.slice(0, intensity.index)
    const main = LOUDER_RE.test(head) || QUIETER_RE.test(head) ? head : q.slice(q.lastIndexOf(',') + 1)
    const duck = QUIETER_RE.test(main)
    if (duck || LOUDER_RE.test(main)) {
      const target = stems.find((s) => main.includes(s.label.toLowerCase())) ?? stem
      const sources = stems.filter((s) => s.kind === 'dialogue').map((s) => s.id)
      if (sources.length === 0) {
        return { actions: [], message: 'No dialogue stems to follow. Intensity uses dialogue loudness.' }
      }
      return {
        actions: [
          duck
            ? { type: 'follow_intensity', stemId: target.id, sourceStemIds: sources, minValue: 1, maxValue: 0.3 }
            : { type: 'follow_intensity', stemId: target.id, sourceStemIds: sources },
        ],
        message: `${target.label} ${duck ? 'ducks' : 'rises'} with dialogue loudness (a proxy, not emotion detection).`,
      }
    }
  }

  if (louder || quieter) {
    const factor = louder ? 2 : 0.3
    if (range) {
      const a = Number(range[1])
      const b = Math.min(Number(range[2]), stem.duration)
      if (b <= a) return { actions: [], message: 'End time must be after start time.' }
      return {
        actions: [{ type: 'set_automation', stemId: stem.id, keyframes: rangeCurve(a, b, factor) }],
        message: `${stem.label} ${louder ? 'louder' : 'quieter'} from ${a}s to ${b}s.`,
      }
    }
    const next = clamp(track.gain * (louder ? 1.5 : 0.5), 0, MAX_GAIN)
    return {
      actions: [{ type: 'set_gain', stemId: stem.id, value: next }],
      message: `${stem.label} fader to ${toDb(next)}.`,
    }
  }

  return {
    actions: [],
    message: 'Stub parser understands: mute, unmute, solo, louder, quieter, to 30%, by 20% (optionally "from 8 to 16 seconds").',
  }
}
