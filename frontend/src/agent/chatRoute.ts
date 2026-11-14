import { fxCommand } from '../fxCommand'
import type { FxCommand, FxStem } from '../fxCommand'
import { namesSfxSound, requestsSfxInsert } from './sfxIntent'

/** "add a phone ring at 0:12", "put an echo at the end": a new sound, not an effect on a layer. */
const INSERT_PHRASING = /^(?:(?:please|can you|could you|would you|just|now|also)\s+)*(?:add|insert|put|drop|throw(?: in)?|stick|slap|play|place)\s+(?:a|an|some|one|the sound of)\b/i
const ON_A_LAYER = /\b(?:to|on|onto|over)\s+(?:the|my|his|her|their|this|that|it|everything|all|every|whole)\b|\b(?:effects?|filters?)\b/i

export interface ChatRoute {
  /** Local rule-based effect plan, used when the AI planner is unavailable. Null when it is not an effect ask. */
  fx: FxCommand | null
  /** Ask the planner for a library sound insert first. */
  sfx: boolean
}

/**
 * The local fast paths of the chat effects (fxCommand) and the library sound inserts (sfxIntent)
 * overlap on a few words ("phone", "echo", "radio"). Decide which owns an ask so neither swallows
 * the other: a named library sound or "add a <thing> at <time>" phrasing is an insert; "add reverb
 * to the vocals" is an effect.
 */
export function chatRoute(text: string, stems: FxStem[] = []): ChatRoute {
  const fx = fxCommand(text, stems)
  const sfx = requestsSfxInsert(text)
  if (!sfx) return { fx, sfx: false }
  if (!fx || fx.kind === 'clarify') return { fx: null, sfx: true }
  const t = text.trim()
  if (namesSfxSound(t) || (INSERT_PHRASING.test(t) && !ON_A_LAYER.test(t))) return { fx: null, sfx: true }
  return { fx, sfx: false }
}
