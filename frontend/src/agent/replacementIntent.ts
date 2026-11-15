/** Route replacement language before any generic source-separation planner. */
export function requestsSoundReplacement(text:string):boolean {
  return /\b(?:replace|replacing|swap|substitute)\b/i.test(text) || /\b(?:make|turn|change)\b.+\b(?:sound\s+like|into)\b/i.test(text) || /\badd\s+.+?\s+(?:at\s+\d|(?:after|when|whenever)\s+(?:(?:the|every|each)\s+)?(?:word\b|["“']))/i.test(text) || /\b(?:add|play)\s+.+?\s+(?:every\s+time|whenever|when)\s+(?:we\s+)?see\s+/i.test(text) || (/\b(?:add|play)\b/i.test(text) && /\b(?:every\s+time|whenever|when)\b/i.test(text) && /\b[a-z][a-z -]*\s*#?\s*\d+\s+(?:appears|shows\s+up|is\s+visible)\b/i.test(text))
}
