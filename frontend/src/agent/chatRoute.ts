import { fxCommand } from '../fxCommand'
import type { FxCommand, FxStem } from '../fxCommand'
import { replacementFollowup } from './replacementFollowup'
import { requestsSoundReplacement } from './replacementIntent'
import { namesSfxSound, requestsSfxInsert } from './sfxIntent'
import { requestsCensorship } from './censorshipIntent'
import { extractLayerScope } from './layerScope'
import { separationCommand } from '../separationCommand'

/** "add a phone ring at 0:12", "put an echo at the end": a new sound, not an effect on a layer. */
const INSERT_PHRASING = /^(?:(?:please|can you|could you|would you|just|now|also)\s+)*(?:add|insert|put|drop|throw(?: in)?|stick|slap|play|place)\s+(?:a|an|some|one|the sound of)\b/i
const ON_A_LAYER = /\b(?:to|on|onto|over)\s+(?:the|my|his|her|their|this|that|it|everything|all|every|whole)\b|\b(?:effects?|filters?)\b/i
/** Main's object-appearance phrasings (see replacementIntent.ts): "... every time we see person 5", "when car 2 appears". */
const OBJECT_APPEARANCE = [/\b(?:every\s+time|whenever|when)\s+(?:we\s+)?see\s+/i, /\b[a-z][a-z -]*\s*#?\s*\d+\s+(?:appears|shows\s+up|is\s+visible)\b/i]
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
 *   4. fx: effect-shaped asks ("add reverb to the vocals", "make the drums sound like a phone");
 *   5. anything else ("isolate the radio chatter", "turn the chorus into a verse") goes to the planner,
 *      the separation fallback, or replacement for "make/turn X into Y" with no effect word.
 * Throws main's layer-scope error when a separation ask names more than one layer.
 */
export function chatRoute(text: string, stems: FxStem[] = [], pendingReplacement = ''): ChatRoute {
  if (requestsCensorship(text)) return { fx: null, sfx: false, replacement: null, censorship: true }
  const t = text.trim()
  const namedInsert = requestsSfxInsert(t) && namesSfxSound(t) && INSERT_PHRASING.test(t)
  const replacementShaped = REPLACE_VERB.test(text) || requestsSoundReplacement(text)
  const layer = replacementShaped || namedInsert ? null : layerScope(text)
  if (layer) {
    // Effects and inserts on a named layer keep their own flows; anything else is main's layer separation.
    if (separationCommand(layer.text)) return { fx: null, sfx: false, replacement: null, layer }
  }
  const followup = namedInsert ? null : replacementFollowup(pendingReplacement, text)
  if (followup) return { fx: null, sfx: false, replacement: followup }
  if (REPLACE_VERB.test(text)) return { fx: null, sfx: false, replacement: text }
  // Sounds on tracked object appearances ("add a gong every time we see person 5") are main's generation flow.
  if (requestsSoundReplacement(text) && OBJECT_APPEARANCE.some(re => re.test(text))) return { fx: null, sfx: false, replacement: text }
  const fx = fxCommand(text, stems)
  const asReplacement = requestsSoundReplacement(text) ? text : null
  const sfx = requestsSfxInsert(text)
  if (sfx && (!fx || fx.kind === 'clarify')) return { fx: null, sfx: true, replacement: asReplacement }
  if (sfx && (namesSfxSound(t) || (INSERT_PHRASING.test(t) && !ON_A_LAYER.test(t)))) return { fx: null, sfx: true, replacement: asReplacement }
  if (fx) return { fx, sfx: false, replacement: null }
  if (layer) return { fx: null, sfx: false, replacement: null, layer }
  return { fx: null, sfx: false, replacement: asReplacement }
}

function layerScope(text: string): { text: string; name: string } | null {
  try { return extractLayerScope(text) } catch (error) {
    // Ambiguous layer wording only matters for a separation ask; anything else ignores it.
    if (/^\s*(?:remove|mute|isolate|separate)\b/i.test(text)) throw error
    return null
  }
}
