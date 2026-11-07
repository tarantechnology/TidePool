import { MAX_GAIN, MAX_KEYFRAMES } from '../edits'
import type { EditAction, Keyframe, Stem } from '../types'

// Agent output is untrusted input: nothing from it reaches the edit pipeline
// without passing through here. Accepted actions are rebuilt field by field, so
// anything extra the model sent is dropped.

export const MAX_ACTIONS = 20
export { MAX_KEYFRAMES }

export type ValidationResult = { ok: true; actions: EditAction[] } | { ok: false; errors: string[] }

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

const inGainRange = (v: unknown): v is number => isFiniteNumber(v) && v >= 0 && v <= MAX_GAIN

function checkKeyframes(raw: unknown, duration: number, at: string, errors: string[]): Keyframe[] {
  if (!Array.isArray(raw) || raw.length < 1 || raw.length > MAX_KEYFRAMES) {
    errors.push(`${at}: keyframes must be an array of 1 to ${MAX_KEYFRAMES} entries`)
    return []
  }
  const out: Keyframe[] = []
  raw.forEach((kf: unknown, i) => {
    if (!isRecord(kf) || !isFiniteNumber(kf.t) || kf.t < 0 || kf.t > duration) {
      errors.push(`${at}: keyframe ${i} t must be within [0, ${duration}]`)
    } else if (!inGainRange(kf.value)) {
      errors.push(`${at}: keyframe ${i} value must be within [0, ${MAX_GAIN}]`)
    } else {
      out.push({ t: kf.t, value: kf.value })
    }
  })
  return out
}

function checkAction(raw: unknown, ids: Set<string>, duration: number, at: string, errors: string[]): EditAction | null {
  if (!isRecord(raw)) {
    errors.push(`${at}: not an object`)
    return null
  }
  const before = errors.length
  const { type, stemId } = raw
  if (typeof stemId !== 'string' || !ids.has(stemId)) errors.push(`${at}: unknown stemId`)
  const id = stemId as string

  switch (type) {
    case 'set_gain': {
      if (!inGainRange(raw.value)) errors.push(`${at}: value must be within [0, ${MAX_GAIN}]`)
      return errors.length > before ? null : { type, stemId: id, value: raw.value as number }
    }
    case 'set_mute': {
      if (typeof raw.muted !== 'boolean') errors.push(`${at}: muted must be a boolean`)
      return errors.length > before ? null : { type, stemId: id, muted: raw.muted as boolean }
    }
    case 'set_solo': {
      if (typeof raw.solo !== 'boolean') errors.push(`${at}: solo must be a boolean`)
      return errors.length > before ? null : { type, stemId: id, solo: raw.solo as boolean }
    }
    case 'set_automation': {
      const keyframes = checkKeyframes(raw.keyframes, duration, at, errors)
      return errors.length > before ? null : { type, stemId: id, keyframes }
    }
    case 'clear_automation':
      return errors.length > before ? null : { type, stemId: id }
    case 'follow_intensity': {
      const { sourceStemIds, minValue, maxValue } = raw
      if (
        !Array.isArray(sourceStemIds) ||
        sourceStemIds.length === 0 ||
        !sourceStemIds.every((s) => typeof s === 'string' && ids.has(s))
      ) {
        errors.push(`${at}: sourceStemIds must be a non-empty list of known stems`)
      }
      if (minValue !== undefined && !inGainRange(minValue)) errors.push(`${at}: minValue out of range`)
      if (maxValue !== undefined && !inGainRange(maxValue)) errors.push(`${at}: maxValue out of range`)
      if (errors.length > before) return null
      return {
        type,
        stemId: id,
        sourceStemIds: sourceStemIds as string[],
        ...(minValue !== undefined ? { minValue: minValue as number } : {}),
        ...(maxValue !== undefined ? { maxValue: maxValue as number } : {}),
      }
    }
    default:
      errors.push(`${at}: unknown action type`)
      return null
  }
}

export function validateActions(actions: unknown, stems: Stem[], duration: number): ValidationResult {
  if (!Array.isArray(actions)) return { ok: false, errors: ['actions must be an array'] }
  if (actions.length > MAX_ACTIONS) return { ok: false, errors: [`at most ${MAX_ACTIONS} actions allowed`] }
  const ids = new Set(stems.map((s) => s.id))
  const errors: string[] = []
  const out: EditAction[] = []
  actions.forEach((raw: unknown, i) => {
    const action = checkAction(raw, ids, duration, `action ${i}`, errors)
    if (action) out.push(action)
  })
  return errors.length > 0 ? { ok: false, errors } : { ok: true, actions: out }
}
