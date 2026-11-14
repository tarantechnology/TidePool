import { useEffect, useState } from 'react'

/**
 * #/                landing page
 * #/app             your projects
 * #/app/new         the editor, empty, ready for a new clip
 * #/app/p/<id>      the editor with that project open
 */
export type Route = 'landing' | 'projects' | 'editor'

export interface RouteState {
  route: Route
  /** The project the editor opens; null for a new one. */
  projectId: string | null
  /** Changes on every navigation, so the editor starts fresh whenever you switch projects. */
  visit: number
}

export function parseHash(hash: string): Omit<RouteState, 'visit'> {
  if (!hash.startsWith('#/app')) return { route: 'landing', projectId: null }
  const rest = hash.slice('#/app'.length).replace(/^\/+/, '')
  if (rest.startsWith('p/')) {
    const id = decodeURIComponent(rest.slice(2).split(/[/?#]/)[0] ?? '')
    if (id) return { route: 'editor', projectId: id }
  }
  if (rest === 'new' || rest.startsWith('new/') || rest.startsWith('new?')) return { route: 'editor', projectId: null }
  return { route: 'projects', projectId: null }
}

let visits = 0

export function useHashRoute(): RouteState {
  const [state, setState] = useState<RouteState>(() => ({ ...parseHash(window.location.hash), visit: visits }))
  useEffect(() => {
    const onChange = () => setState({ ...parseHash(window.location.hash), visit: ++visits })
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return state
}

/** Go somewhere in the app. Navigating to where you already are still starts the editor fresh. */
export function navigate(hash: string) {
  if (window.location.hash === hash) window.dispatchEvent(new HashChangeEvent('hashchange'))
  else window.location.hash = hash
}
