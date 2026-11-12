import type { SavedStem } from './backend'

// What the separation requests carry, and how their results are matched back to
// the words the user used. Kept out of ProjectPanel so both can be tested.

/**
 * `visual_prompt` for a request with no visual selection. It has to be sent:
 * without the field the backend substitutes the project's stored detections
 * document, which is megabytes of masks and does not survive the HTTP adapter.
 */
export const NO_VISUAL_PROMPT: Record<string, never> = {}

/**
 * The `visual_prompt` body for a separation. `selection` is the object the
 * video panel or an imported JSON file produced; the interval comes from the
 * two range inputs, as strings, so it can be corrected before it is sent.
 */
export function buildVisualPrompt(selection: unknown, rangeStart: string, rangeEnd: string): object {
  if (!selection) return NO_VISUAL_PROMPT
  const start = Number(rangeStart)
  const end = Number(rangeEnd)
  if (!rangeStart || !rangeEnd || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    throw new Error('Enter a valid visual selection interval.')
  }
  return { ...(selection as object), interval: { start_seconds: start, end_seconds: end } }
}

/**
 * Stem names come back as the caller's own wording with whitespace collapsed,
 * so "  Dog   barking " and "dog barking" are the same sound.
 */
export function normalizeStemName(name: string): string {
  return name.trim().replace(/\s+/g, ' ').toLowerCase()
}

/** The stem holding the residual audio, by role when the backend sends one. */
export const RESIDUAL_NAME = 'residual'

export interface RemoveTargets {
  /** The requested sound, which "remove X" mutes. */
  target: SavedStem | null
  /** Everything else, which has to exist for muting the target to mean anything. */
  residual: SavedStem | null
}

const matchesDescription = (stem: SavedStem, wanted: string): boolean =>
  [stem.prompt, stem.label, stem.name].some((value) => typeof value === 'string' && normalizeStemName(value) === wanted)

/**
 * Pick the two stems a "remove X" needs out of what the project holds now.
 * Either may be null: the `fake` separation adapter returns one stem and no
 * residual, and a role-less backend may name its stems differently again. The
 * caller reports that, it does not treat it as a failed edit.
 */
export function findRemoveTargets(stems: readonly SavedStem[], description: string): RemoveTargets {
  const wanted = normalizeStemName(description)
  const residual =
    stems.find((s) => s.role === RESIDUAL_NAME) ??
    stems.find((s) => normalizeStemName(s.name) === RESIDUAL_NAME) ??
    null
  if (stems.some((s) => typeof s.role === 'string')) {
    const targets = stems.filter((s) => s.role === 'target')
    // One prompt was sent, so a lone target is the sound that was asked for even
    // if the service reworded it. Several, and only a name match will do.
    const target = targets.find((s) => matchesDescription(s, wanted)) ?? (targets.length === 1 ? targets[0] : null)
    return { target, residual }
  }
  const named = stems.filter((s) => normalizeStemName(s.name) === wanted)
  return { target: named.length === 1 ? named[0] : null, residual }
}

/** Why nothing was muted, in terms of what came back rather than what failed. */
export function missingResidualMessage({ target, residual }: RemoveTargets, description: string): string {
  const missing = !target && !residual ? `no "${description}" stem and no residual` : !residual ? 'no residual audio' : `no "${description}" stem`
  return `Separation returned ${missing}, so nothing was muted. The stems are loaded; mute by hand, or separate again with a real model.`
}
