/** Route replacement language before any generic source-separation planner. */
export function requestsSoundReplacement(text:string):boolean {
  return /\b(?:replace|replacing|swap|substitute)\b/i.test(text) || /\b(?:make|turn|change)\b.+\b(?:sound\s+like|into)\b/i.test(text) || /\badd\s+.+?\s+(?:at\s+\d|(?:after|when|whenever)\s+(?:(?:the|every|each)\s+)?(?:word\b|["“']))/i.test(text) || (/\b(?:add|insert|place|generate|put|trigger|cue|overlay)\b/i.test(text) && /\b(?:sound|sounds|effect|sfx|every\s+time|whenever|when|word)\b/i.test(text)) || (/\bplay\b/i.test(text) && /\b(?:every\s+time|whenever|when)\b/i.test(text))
}
