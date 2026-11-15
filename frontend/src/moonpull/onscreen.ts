import type { VisualMetadata } from '../api/backend'
import { cleanMoments, MAX_MOMENTS } from './rules'
import type { Moment } from './rules'
import { trackTitle } from '../objects'

// "Whenever this person is on screen": the moments come from the object tracker. Each frame the
// detector analysed lists the tracked objects in it, so an object's moments are the runs of frames
// it appears in. The person is picked by clicking them in the video, never by a description.

export interface ScreenOptions {
  /** A gap shorter than this (a missed frame, someone walking past) does not end a moment. */
  bridge?: number
  /** Moments shorter than this are dropped: a face flashing by is not an appearance. */
  minLength?: number
}

/** Moments where any of the tracks is on screen, in timeline seconds. */
export function onScreenMoments(metadata: VisualMetadata, trackIds: number[], duration: number, opts: ScreenOptions = {}): Moment[] {
  const { bridge = 0.5, minLength = 0.4 } = opts
  const wanted = new Set(trackIds)
  if (wanted.size === 0) return []
  const runs: Array<{ start: number; end: number; labels: Set<string> }> = []
  const frames = [...metadata.frames].sort((a, b) => a.timestamp_seconds - b.timestamp_seconds)
  for (const frame of frames) {
    const seen = frame.objects.filter((o) => o.track_id !== null && wanted.has(o.track_id))
    if (seen.length === 0) continue
    const start = frame.timestamp_seconds
    const end = Number.isFinite(frame.valid_until_seconds) && frame.valid_until_seconds > start ? frame.valid_until_seconds : start + 1 / metadata.fps
    const last = runs.at(-1)
    if (last && start - last.end <= bridge) last.end = Math.max(last.end, end)
    else runs.push({ start, end, labels: new Set() })
    for (const o of seen) runs.at(-1)!.labels.add(trackTitle(o))
  }
  const round = (v: number) => Math.round(v * 100) / 100
  return cleanMoments(
    runs.filter((r) => r.end - r.start >= minLength).map((r) => ({ start: round(r.start), end: round(r.end), note: `${[...r.labels].join(', ')} on screen` })),
    duration,
  ).slice(0, MAX_MOMENTS)
}

/** Seconds of the clip the tracker looked at; past this nobody is "on screen" as far as it knows. */
export const analysedSeconds = (metadata: VisualMetadata) => metadata.processed_duration_seconds

export interface ScreenCommand {
  /** Who or what, in the user's words, for the rule's name. */
  subject: string
  /** The sound to add, in the user's words; empty when they did not say. */
  sound: string
}

const ON_SCREEN = String.raw`(?:on[\s-]?screen|in (?:the )?(?:frame|shot|picture|scene|view)|on camera|visible|in view)`
const APPEARS = String.raw`(?:appears?|shows? up|comes? (?:on|in)|walks? in|enters?|is seen)`

/**
 * "Whenever X is on screen, add/play Y" and close variants. Anything else returns null and goes to
 * the general planner. Only the wording is read here; who X is gets settled by a click on the video.
 */
export function parseOnScreenCommand(text: string): ScreenCommand | null {
  const t = text.trim().replace(/\s+/g, ' ')
  const when = String.raw`(?:whenever|when|every ?time|each time|while|as long as|anytime|any time)`
  const subjectFirst = new RegExp(String.raw`\b${when}\s+(.+?)\s+(?:(?:is|are|'s|gets?)\s+${ON_SCREEN}|${APPEARS}(?:\s+${ON_SCREEN})?)\b`, 'i')
  const m = subjectFirst.exec(t)
  if (!m) return null
  const subject = m[1].trim()
  if (!subject || subject.split(' ').length > 8) return null
  // A numbered detector track is an explicit reference and should go to the
  // sound-effect planner, which can resolve that track without another click.
  if (/\b(?:person|dog|cat|car|object|animal|bird|vehicle)\s*#?\s*\d+\b/i.test(subject)) return null
  const verb = /\b(?:add|play|put|bring in|cue|start|layer in|use)\s+(?:in\s+|on\s+)?(.+)$/i.exec(t.slice(0, m.index) + ' ' + t.slice(m.index + m[0].length))
  let sound = verb ? verb[1] : ''
  sound = sound.replace(/[.!?]+$/, '').replace(/\b(?:whenever|when|every ?time|each time|while)\b.*$/i, '').replace(/^(?:some|a|an|the)\s+/i, '').trim()
  return { subject, sound: sound.slice(0, 120) }
}

/** Point a rule at whoever is selected in the video: their track, and the moments they are on screen. */
export function followSelection<T extends { trackIds?: number[]; moments: Moment[] }>(rule: T, metadata: VisualMetadata | null, trackId: number | null | undefined, duration: number): T {
  if (!metadata || trackId === null || trackId === undefined) return rule
  const trackIds = [trackId]
  return { ...rule, trackIds, moments: onScreenMoments(metadata, trackIds, duration) }
}
