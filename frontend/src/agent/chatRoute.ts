import { fxCommand } from '../fxCommand'
import type { FxCommand, FxStem } from '../fxCommand'
import { replacementFollowup } from './replacementFollowup'
import { requestsSoundReplacement } from './replacementIntent'
import { namesSfxSound, requestsSfxInsert } from './sfxIntent'
import { requestsGeneratePart } from './generateIntent'
import { requestsCensorship } from './censorshipIntent'
import { extractLayerScope } from './layerScope'
import { separationCommand } from '../separationCommand'

/** "add a phone ring at 0:12", "put an echo at the end": a new sound, not an effect on a layer. */
const INSERT_PHRASING = /^(?:(?:please|can you|could you|would you|just|now|also)\s+)*(?:add|insert|put|drop|throw(?: in)?|stick|slap|play|place)\s+(?:a|an|some|one|the sound of)\b/i
const ON_A_LAYER = /\b(?:to|on|onto|over)\s+(?:the|my|his|her|their|this|that|it|everything|all|every|whole)\b|\b(?:effects?|filters?)\b/i
/** Explicit replacement verbs: these never become an insert or an effect. */
const REPLACE_VERB = /\b(?:replace|replacing|swap|substitute)\b|\binstead of\b/i

export interface ChatRoute {
  /** "Bleep the swearing": the profanity beep flow (local transcription) owns it, before anything else. */
  censorship?: boolean
  /** "Isolate the dog in the Music layer": separation from one named layer (main's layer-scope flow). */
  layer?: { text: string; name: string }
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
 *   0. censorship: "bleep the swearing", "censor the cuss words" go to the profanity beep flow, always first;
 *   1. layer scope: a separate/remove ask naming one layer ("isolate the dog in the Music layer");
 *      layer-scoped replacements stay replacements (they use the selected source layer);
 *   2. replacement: an answer to a pending replacement question ("at 12") or replace/swap/substitute;
 *      a named library sound with insert wording ("add a vine boom at 0:12") wins over a pending question;
 *   3. insert: a named library sound or "add a <thing> at/when/before…" (falls back to replacement
 *      when the library has no match and the ask reads as one, e.g. "add a monster roar at 12");
 *   4. generate: an instrument part ("add violins", "add a piano part", "add a funky bassline"); the
 *      server's agent route catches the same phrasings (app/generate_part.py rule_plan) after inserts;
 *   5. fx: effect-shaped asks ("add reverb to the vocals", "make the drums sound like a phone");
 *   6. anything else ("isolate the radio chatter", "turn the chorus into a verse") goes to the planner,
 *      the separation fallback, or replacement for "make/turn X into Y" with no effect word.
 * Throws main's layer-scope error when a separation ask names more than one layer.
 */
export function chatRoute(text: string, stems: FxStem[] = [], pendingReplacement = ''): ChatRoute {
  const none = { fx: null, sfx: false, replacement: null, generate: false }
  if (requestsCensorship(text)) return { ...none, censorship: true }
  const t = text.trim()
  const namedInsert = requestsSfxInsert(t) && namesSfxSound(t) && INSERT_PHRASING.test(t)
  const replacementShaped = REPLACE_VERB.test(text) || requestsSoundReplacement(text)
  const layer = replacementShaped || namedInsert ? null : layerScope(text)
  if (layer) {
    // Effects and inserts on a named layer keep their own flows; anything else is main's layer separation.
    if (separationCommand(layer.text)) return { ...none, layer }
  }
  const followup = namedInsert ? null : replacementFollowup(pendingReplacement, text)
  if (followup) return { ...none, replacement: followup }
  if (REPLACE_VERB.test(text)) return { ...none, replacement: text }
  const asReplacement = requestsSoundReplacement(text) ? text : null
  const sfx = requestsSfxInsert(text)
  // A named library sound ("add a sad trombone at the end") is an insert before any instrument reading.
  if (sfx && namesSfxSound(t)) return { ...none, sfx: true, replacement: asReplacement }
  if (requestsGeneratePart(t)) return { ...none, generate: true }
  const fx = fxCommand(text, stems)
  if (sfx && (!fx || fx.kind === 'clarify')) return { ...none, sfx: true, replacement: asReplacement }
  if (sfx && INSERT_PHRASING.test(t) && !ON_A_LAYER.test(t)) return { ...none, sfx: true, replacement: asReplacement }
  if (fx) return { ...none, fx }
  if (layer) return { ...none, layer }
  return { ...none, replacement: asReplacement }
}

function layerScope(text: string): { text: string; name: string } | null {
  try { return extractLayerScope(text) } catch (error) {
    // Ambiguous layer wording only matters for a separation ask; anything else ignores it.
    if (/^\s*(?:remove|mute|isolate|separate)\b/i.test(text)) throw error
    return null
  }
}
