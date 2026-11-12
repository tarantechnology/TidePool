/** Explicit whole-track requests only; timed or compound requests need clarification. */
export function separationCommand(text: string): {description: string; remove: boolean} | null {
  const match = /^\s*(remove|isolate|separate)\s+(?:the\s+)?(.+?)\s*[.!]?\s*$/i.exec(text)
  if (!match) return null
  const description = match[2].replace(/[.!]+$/, '').trim()
  if (!description || /\b(?:from|between|until|after|before|when|while|then|and|at)\b|\d/i.test(description)) return null
  return {description, remove: match[1].toLowerCase() === 'remove'}
}
