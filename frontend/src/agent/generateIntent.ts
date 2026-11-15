/**
 * "add violins", "add a piano part", "add a funky bassline": a new generated instrument part.
 * Mirrors app/generate_part.py rule_plan (the server catches the same phrasings before the AI planner).
 * Effect, mix and sound-effect words keep their own flows: "add reverb to the strings" is fx,
 * "add a vine boom" is an insert, "add more bass" is EQ, "add chorus to the guitar" is an effect.
 */
const VERB = /^\s*(?:please\s+)?(?:can you\s+)?(?:add|put|layer|throw in|bring in)\s+/i
const TAIL = /\s+(?:to|in|into|under|over|on|during|for|behind|throughout|at|from|after|before|starting|when)\b.*$/i
const INSTRUMENTS = /\b(?:violins?|viola|cellos?|strings?|string pad|orchestra(?:l)?|fiddle|double bass|upright bass|contrabass|pizzicato|piano|keys|rhodes|organ|harpsichord|celesta|glockenspiel|music box|vibraphone|marimba|xylophone|harp|guitars?|riff|bass(?:line| line| guitar)?|synths?|arp(?:eggio)?|pad|lead|pluck|trumpets?|trombones?|horns?|brass|tuba|sax(?:ophone)?|flutes?|clarinet|oboe|bassoon|choir|voices|aahs|oohs|banjo|accordion|harmonica|sitar|ukulele|mandolin|kalimba|steel drums?|countermelody|counter-melody|melody|chords|bells)\b/i
const EFFECTS = /\b(?:reverb\w*|echo\w*|delay|effects?|eq|filter\w*|fade\w*|boom|sfx|sound effects?|laugh\w*|compress\w*|distort\w*|saturat\w*|overdrive|flanger|phaser|tremolo|auto-?tune|lo-?fi|vinyl|bitcrush\w*|duck\w*|limiter|riser|whoosh|swell|impact|applause|crickets|air\s*horn|rimshot|record scratch|sad trombone)\b/i
const EQ_ASK = /^(?:some\s+)?(?:more|less|extra)\s+(?:bass|treble|low[- ]end|lows|highs|top end)\b/i

export function requestsGeneratePart(text: string): boolean {
  const t = text.trim()
  const m = VERB.exec(t)
  if (!m || EFFECTS.test(t)) return false
  const after = t.slice(m[0].length)
  if (EQ_ASK.test(after)) return false
  return INSTRUMENTS.test(after.replace(TAIL, ''))
}
