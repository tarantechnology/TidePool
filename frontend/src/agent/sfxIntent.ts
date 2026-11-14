/**
 * Language that may ask for a library sound effect ("add a vine boom when he says 'what'",
 * "record scratch at 0:12", "sad trombone at the end"). Such prompts go to the backend planner
 * first: it answers with an `insert` plan, a clarify, or something else for the normal flow.
 * Replacement wording ("replace X with Y", "make X sound like Y") never matches.
 */
const VERB = /^(?:(?:please|can you|could you|would you|i want(?: you to)?|i'd like(?: you)? to|let's|just|now|also)\s+)*(?:add|insert|put|drop|throw(?: in)?|stick|slap|play|place|include|give (?:me|it|us))\b/i
const ANCHOR = /\b(?:when(?:ever)?|every\s*time|each\s+time|the\s+(?:first|second|third|last)\s+time|right\s+(?:after|before)|after|before)\s+(?:\w+\s+)?(?:says?|said|goes|yells?|shouts?|screams?)\b|\b(?:at|by|on)\s+the\s+(?:very\s+)?(?:end|start|beginning)\b|\bduring\s+the\s+(?:awkward\s+)?(?:silence|pause)\b|\bon\s+(?:the\s+)?(?:beat|downbeat)\b|\b(?:at|@)\s+\d{1,2}:\d{2}\b/i
const MEME = /\b(?:vine\s*boom|bruh|record\s+scratch|sad\s+trombone|rimshot|ba[\s-]?dum|laugh(?:ter|\s+track)?|applause|crickets|air\s*horn|whoosh|riser|ding|ka-?ching|cash\s+register|buzzer|drum\s*roll|tada|ta-da|boing|slide\s+whistle|gasp|oof|dun\s+dun|sting|womp|fanfare)\b/i
const REPLACE = /\b(?:replace|replacing|swap|substitute|instead\s+of)\b|\b(?:make|turn|change)\b.+\b(?:sound\s+like|into)\b/i

/** True when the text names a meme/editor sound from the library ("vine boom", "riser", "laugh track"). */
export function namesSfxSound(text: string): boolean {
  return MEME.test(text)
}

export function requestsSfxInsert(text: string): boolean {
  const t = text.trim()
  if (!t || REPLACE.test(t)) return false
  return MEME.test(t) || (VERB.test(t) && ANCHOR.test(t))
}

/** "Deep boom (vine-boom style) at 1.50s" lines for the preview panel. */
export function placementLines(placements: ReadonlyArray<{ title: string; start: number; anchor_time: number }>): string[] {
  return placements.map((p) => `${p.title} at ${p.start.toFixed(2)}s${Math.abs(p.start - p.anchor_time) > 0.005 ? ` (anchor ${p.anchor_time.toFixed(2)}s)` : ''}`)
}

/**
 * After "I can't tell when that happens, give me a timestamp", a bare time ("0:42", "at 12s")
 * completes the previous request: "add laugh track at 0:42". Null when the text is not a bare time.
 */
export function sfxFollowup(pendingQuery: string, text: string): string | null {
  const match = text.trim().match(/^(?:(?:at|around)\s+)?(\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:s|secs?|seconds?)?)[.!?]?$/i)
  return pendingQuery && match ? `add ${pendingQuery} at ${match[1]}` : null
}
