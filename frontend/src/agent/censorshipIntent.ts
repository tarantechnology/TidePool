/** Explicit requests opt into local transcription; do not infer censorship from unrelated text. */
export function requestsCensorship(text: string): boolean {
  return !/\b(?:do not|don't|never)\s+(?:beep|bleep|censor|mute|remove)\b/i.test(text)
    && /\b(?:beep|bleep|censor|mute|remove)\b/i.test(text)
    && /\b(?:cuss(?:ing)?|swear(?:ing)?|profanity|curse|cursing|expletives?)\b/i.test(text)
}
