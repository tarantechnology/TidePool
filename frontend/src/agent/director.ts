import type { DirectorIntent } from '../api/backend'

/** Narrow local fallback: requests with extra constraints go to the language planner. */
export function creativeCommand(text:string):DirectorIntent|null {
  const value=text.trim().toLowerCase()
  if (/^(?:please\s+)?(?:emphasi[sz]e|boost|highlight|bring out)\s+(?:the\s+)?funny\s+(?:sounds?|effects?)[.!?]?$/.test(value)) return 'funny'
  if (/^(?:(?:could|can)\s+you\s+)?(?:please\s+)?make\s+(?:(?:the|this)\s+)?(?:whole\s+)?(?:video|audio|clip|mix|it)\s+(?:(?:sound|feel)\s+)?(?:more\s+)?(?:dramatic|cool)[.!?]?$/.test(value)) return /\bcool\b/.test(value)?'cool':'dramatic'
  return null
}
export function directorIntent(text:string):DirectorIntent {
  return /\b(funny|comedy|comic|humorous)\b/i.test(text) ? 'funny' : /\bdramatic\b/i.test(text) ? 'dramatic' : /\bcool\b/i.test(text) ? 'cool' : 'cinematic'
}
