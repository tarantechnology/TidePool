import { fxCommand } from '../fxCommand'
import type { FxCommand, FxStem } from '../fxCommand'
import { replacementFollowup } from './replacementFollowup'
import { requestsSoundReplacement } from './replacementIntent'
import { namesSfxSound, requestsSfxInsert } from './sfxIntent'
import { requestsGeneratePart } from './generateIntent'

/** "add a phone ring at 0:12", "put an echo at the end": a new sound, not an effect on a layer. */
const INSERT_PHRASING = /^(?:(?:please|can you|could you|would you|just|now|also)\s+)*(?:add|insert|put|drop|throw(?: in)?|stick|slap|play|place)\s+(?:a|an|some|one|the sound of)\b/i
const ON_A_LAYER = /\b(?:to|on|onto|over)\s+(?:the|my|his|her|their|this|that|it|everything|all|every|whole)\b|\b(?:effects?|filters?)\b/i
/** Explicit replacement verbs: these never become an insert or an effect. */
const REPLACE_VERB = /\b(?:replace|replacing|swap|substitute)\b|\binstead of\b/i

export interface ChatRoute {
  /** Local rule-based effect plan, used when the AI planner is unavailable. Null when it is not an effect ask. */
  fx: FxCommand | null
  /** Ask the planner for a library sound insert first. */
  sfx: boolean
  /** The sound-replacement request to run (completed with a follow-up time), when the ask is, or falls back to, a replacement. */
  replacement: string | null
  /** A new instrument part ("add violins", "add a piano part"): the planner answers kind 'generate'. */
  generate: boolean
}

/**
 * The command bar's local fast paths, in precedence order:
 *   1. replacement: an answer to a pending replacement question ("at 12") or replace/swap/substitute;
 *   2. insert: a named library sound or "add a <thing> at/when/before…" (falls back to replacement
 *      when the library has no match and the ask reads as one, e.g. "add a monster roar at 12");
 *   3. generate: an instrument part ("add violins", "add a piano part", "add a funky bassline"); the
 *      server's agent route catches the same phrasings (app/generate_part.py rule_plan) after inserts;
 *   4. fx: effect-shaped asks ("add reverb to the vocals", "make the drums sound like a phone");
 *   5. anything else ("isolate the radio chatter", "turn the chorus into a verse") goes to the planner,
 *      the separation fallback, or replacement for "make/turn X into Y" with no effect word.
 */
export function chatRoute(text: string, stems: FxStem[] = [], pendingReplacement = ''): ChatRoute {
  const followup = replacementFollowup(pendingReplacement, text)
  if (followup) return { fx: null, sfx: false, replacement: followup, generate: false }
  if (REPLACE_VERB.test(text)) return { fx: null, sfx: false, replacement: text, generate: false }
  const t = text.trim()
  const asReplacement = requestsSoundReplacement(text) ? text : null
  const sfx = requestsSfxInsert(text)
  // A named library sound ("add a sad trombone at the end") is an insert before any instrument reading.
  if (sfx && namesSfxSound(t)) return { fx: null, sfx: true, replacement: asReplacement, generate: false }
  if (requestsGeneratePart(t)) return { fx: null, sfx: false, replacement: null, generate: true }
  const fx = fxCommand(text, stems)
  if (sfx && (!fx || fx.kind === 'clarify')) return { fx: null, sfx: true, replacement: asReplacement, generate: false }
  if (sfx && INSERT_PHRASING.test(t) && !ON_A_LAYER.test(t)) return { fx: null, sfx: true, replacement: asReplacement, generate: false }
  return fx ? { fx, sfx: false, replacement: null, generate: false } : { fx: null, sfx: false, replacement: asReplacement, generate: false }
}
