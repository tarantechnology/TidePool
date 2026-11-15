/** Route replacement language before any generic source-separation planner. */
export function requestsSoundReplacement(text:string):boolean {
  return /\b(?:replace|replacing|swap|substitute)\b/i.test(text) || /\b(?:make|turn|change)\b.+\b(?:sound\s+like|into)\b/i.test(text) || /\badd\s+.+?\s+(?:at\s+\d|(?:after|when|whenever)\s+(?:(?:the|every|each)\s+)?(?:word\b|["“']))/i.test(text)
}
