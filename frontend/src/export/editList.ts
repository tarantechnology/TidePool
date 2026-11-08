import { MAX_GAIN, toEditList } from '../edits'
import type { EditList, Keyframe, TrackState } from '../types'

export type ParseResult = { ok: true; editList: EditList } | { ok: false; errors: string[] }

export function serializeEditList(tracks: TrackState[]): string {
  return JSON.stringify(toEditList(tracks), null, 2)
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

const inRange = (v: unknown, lo: number, hi: number): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi

function parseKeyframe(raw: unknown, at: string, errors: string[]): Keyframe | null {
  if (!isRecord(raw) || !inRange(raw.t, 0, Infinity) || !inRange(raw.value, 0, MAX_GAIN)) {
    errors.push(`${at}: invalid keyframe`)
    return null
  }
  return { t: raw.t, value: raw.value }
}

function parseTrack(raw: unknown, at: string, errors: string[]): TrackState | null {
  if (!isRecord(raw)) {
    errors.push(`${at}: not an object`)
    return null
  }
  const { stemId, gain, muted, solo, automation } = raw
  const before = errors.length
  if (typeof stemId !== 'string' || stemId === '') errors.push(`${at}: stemId must be a string`)
  if (!inRange(gain, 0, MAX_GAIN)) errors.push(`${at}: gain must be within [0, ${MAX_GAIN}]`)
  if (typeof muted !== 'boolean') errors.push(`${at}: muted must be a boolean`)
  if (typeof solo !== 'boolean') errors.push(`${at}: solo must be a boolean`)
  if (!Array.isArray(automation)) errors.push(`${at}: automation must be an array`)
  const keyframes = (Array.isArray(automation) ? automation : [])
    .map((k: unknown, i) => parseKeyframe(k, `${at} keyframe ${i}`, errors))
    .filter((k) => k !== null)
    .sort((a, b) => a.t - b.t)
  if (errors.length > before) return null
  return {
    stemId: stemId as string,
    gain: gain as number,
    muted: muted as boolean,
    solo: solo as boolean,
    automation: keyframes,
  }
}

/** Validate an edit list. Accepts a parsed object or its JSON text. */
export function parseEditList(input: unknown): ParseResult {
  let raw = input
  if (typeof raw === 'string') {
    try {
      raw = JSON.parse(raw)
    } catch {
      return { ok: false, errors: ['not valid JSON'] }
    }
  }
  if (!isRecord(raw)) return { ok: false, errors: ['edit list must be an object'] }
  if (raw.version !== 1) return { ok: false, errors: ['unsupported version'] }
  if (!Array.isArray(raw.tracks)) return { ok: false, errors: ['tracks must be an array'] }
  const errors: string[] = []
  const tracks = raw.tracks
    .map((t: unknown, i) => parseTrack(t, `track ${i}`, errors))
    .filter((t) => t !== null)
  return errors.length > 0 ? { ok: false, errors } : { ok: true, editList: { version: 1, tracks } }
}
