import type { EditAction } from '../types'
import {
  combineEnvelopes,
  DEFAULT_WINDOW_SEC,
  intensityToKeyframes,
  normalizeEnvelope,
  rmsEnvelope,
  smoothEnvelope,
} from './analysis'

/** The part of AudioEngine this needs, so it can be tested without an AudioContext. */
export interface SampleSource {
  getSamples(stemId: string): { data: Float32Array; sampleRate: number } | null
}

export interface ResolveResult {
  actions: EditAction[]
  warnings: string[]
}

/**
 * Replace each follow_intensity action with the concrete set_automation it
 * implies. Source levels are combined raw, not normalized per stem, so a
 * speaker who is quiet throughout does not get boosted into looking intense.
 */
export function resolveActions(actions: EditAction[], source: SampleSource, duration: number): ResolveResult {
  const resolved: EditAction[] = []
  const warnings: string[] = []
  for (const action of actions) {
    if (action.type !== 'follow_intensity') {
      resolved.push(action)
      continue
    }
    const envs: Float32Array[] = []
    for (const id of action.sourceStemIds) {
      const samples = source.getSamples(id)
      if (samples) envs.push(rmsEnvelope(samples.data, samples.sampleRate, DEFAULT_WINDOW_SEC))
      else warnings.push(`No audio for source stem ${id}.`)
    }
    if (envs.length === 0) {
      warnings.push(`Skipped follow_intensity on ${action.stemId}: no source audio.`)
      continue
    }
    // The empty envelope stretches the result to the timeline, so a source that
    // ends early reads as silence instead of holding its last level.
    const timeline = new Float32Array(Math.ceil(duration / DEFAULT_WINDOW_SEC))
    // Two passes make a triangular filter. One box pass lets the syllable rate
    // (around 5 Hz) ripple through and the keyframes zigzag.
    const env = normalizeEnvelope(smoothEnvelope(smoothEnvelope(combineEnvelopes([...envs, timeline], 'max'))))
    resolved.push({
      type: 'set_automation',
      stemId: action.stemId,
      keyframes: intensityToKeyframes(env, DEFAULT_WINDOW_SEC, {
        minValue: action.minValue,
        maxValue: action.maxValue,
        duration,
      }),
    })
  }
  return { actions: resolved, warnings }
}
