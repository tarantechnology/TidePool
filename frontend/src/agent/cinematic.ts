import type { CinematicStyle } from '../api/backend'

/** Conservative local fallback for the common intent; other requests go to the planner. */
export function cinematicCommand(text: string, refining: boolean): CinematicStyle | null {
  const value = text.trim().toLowerCase()
  if (/^(?:(?:could|can)\s+you\s+)?(?:please\s+)?make\s+(?:(?:the|this)\s+)?(?:whole\s+)?(?:video|audio|clip|mix|it)\s+(?:(?:sound|feel)\s+)?(?:more\s+)?cinematic[.!?]?$/.test(value)) return 'cinematic'
  if (refining && /^(?:please\s+)?(?:make\s+it\s+)?(?:less dramatic|more subtle|subtler|gentler)[.!]?$/.test(value)) return 'subtle'
  if (refining && /^(?:please\s+)?(?:make\s+it\s+)?more dramatic[.!]?$/.test(value)) return 'dramatic'
  return null
}
