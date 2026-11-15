/**
 * "Bass boost the bad words", "add echo whenever he says McLovin": an effect that follows spoken words.
 * The server's planner owns these (app/word_fx.py finds the words in a local transcript), so the command bar
 * sends them there instead of to a local effect, insert or replacement fast path. Mirrors word_fx.parse.
 * "Add a vine boom when he says 'what'" names no effect, so it stays a sound insert; "bleep the swearing" stays censorship.
 */
const EFFECT = /\b(?:reverb\w*|echo\w*|delay|chorus|flanger|phaser|tremolo|distort\w*|overdrive|eq|bass(?:\s*[- ]?boost\w*|ier)?|boost\w*|treble|compress\w*|pitch\w*|deep(?:en|er)?|muffle\w*|muffled|lo-?fi|bitcrush\w*|telephone|phone|radio|megaphone|underwater|robot\w*|chipmunk|squeak\w*|helium|high[- ]pitched|higher|giant|announcer|church|cathedral|stadium|hall|bathroom|dreamy|warm(?:er)?|bright(?:er)?|louder|quieter|loud|quiet|deep[- ]fried|earrape|8-?bit|vinyl|wobble|stutter|slow(?:er|ed)?|effect)\b/i
const PROFANITY = /\b(?:(?:bad|swear|curse|cuss|dirty|naughty|foul|explicit)\s*[- ]?\s*words?|swears?|swearing|curses|cursing|cuss(?:es|ing)?|profanit(?:y|ies)|expletives?|f[- ]bombs?)\b/i
const SAID = /\b(?:whenever|every\s*time|each\s+time|any\s*time|when)\s+\S+(?:\s+\d+)?\s+(?:says?|said|saying|yells?|shouts?)\s+\S/i
const ON_WORD = /\b(?:(?:on|to|over|for|at|during)\s+)?(?:the\s+|every\s+|each\s+)?(?:word|name|phrase)s?\s+["“'‘]?[a-z0-9]/i
/** "every Muhammad": a capitalised name after every/each (case-sensitive, so never "every time"). */
const EVERY_NAME = /\b(?:every|each)\s+["“'‘]?[A-Z][a-z]/
const INSTANCE = /\b(?:every|each|all|any)\s+(?:the\s+)?(?:instances?|occurrences?|mentions?|uses?)\s+of\s+\S/i

export function requestsWordEffect(text: string): boolean {
  return EFFECT.test(text) && (PROFANITY.test(text) || SAID.test(text) || ON_WORD.test(text) || INSTANCE.test(text) || EVERY_NAME.test(text))
}
