import { useEffect, useState } from 'react'
import { claimShareLink } from '../projects/registry'

/**
 * #/                landing page
 * #/app             your projects
 * #/app/new         the editor, empty, ready for a new clip
 * #/app/p/<id>      the editor with that project open
 * #/app/open/<id>?t=<token>&n=<name>   a shared link: adds the project here, then opens it
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

/** A shared project link is added to this browser and swapped for the project's own address. */
function current(): Omit<RouteState, 'visit'> {
  const claimed = claimShareLink(window.location.hash)
  if (claimed) { try { window.history.replaceState(null, '', claimed) } catch { /* parse the claimed address anyway */ } return parseHash(claimed) }
  return parseHash(window.location.hash)
}

export function useHashRoute(): RouteState {
  const [state, setState] = useState<RouteState>(() => ({ ...current(), visit: visits }))
  useEffect(() => {
    const onChange = () => setState({ ...current(), visit: ++visits })
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
