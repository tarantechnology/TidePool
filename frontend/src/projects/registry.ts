// The projects this browser has opened, so going in and out of them never needs a pasted ID
// and token. Each project's token lives here (this browser only) and is sent only as a bearer
// header. Forgetting a project removes it from this list; it stays on the server.

import { EMPTY_STEPS, parseSteps } from '../state/steps'
import type { StepStack } from '../state/steps'

export interface SavedProject {
  id: string
  token: string
  name: string
  createdAt: number
  openedAt: number
  duration?: number
  layers?: number
  hasVideo?: boolean
  /** A small JPEG data URL of a frame, for the project card. */
  thumb?: string
  /** The name was typed by the user, so the clip's filename no longer replaces it. */
  named?: boolean
  /** When the user last pressed Save. */
  savedAt?: number
}

const KEY = 'tidepool.projects'
const STEPS_KEY = (id: string) => `tidepool.steps:${id}`
export const MAX_PROJECTS = 40
const listeners = new Set<() => void>()
/** The list as last read, so React's useSyncExternalStore gets the same array until it changes. */
let cached: SavedProject[] | null = null

function read(): SavedProject[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]') as unknown
    if (!Array.isArray(raw)) return []
    return raw.filter((p): p is SavedProject => typeof p?.id === 'string' && typeof p.token === 'string' && typeof p.name === 'string')
  } catch { return [] }
}

function write(list: SavedProject[]) {
  const sorted = [...list].sort((a, b) => b.openedAt - a.openedAt).slice(0, MAX_PROJECTS)
  try { localStorage.setItem(KEY, JSON.stringify(sorted)) } catch {
    // Out of space: thumbnails are the big part, so drop the oldest ones and try again.
    try { localStorage.setItem(KEY, JSON.stringify(sorted.map((p, i) => (i < 8 ? p : { ...p, thumb: undefined })))) } catch { /* the list just won't persist */ }
  }
  cached = null
  listeners.forEach((l) => l())
}

/** Most recently opened first. */
export function listProjects(): SavedProject[] {
  return read().sort((a, b) => b.openedAt - a.openedAt)
}

export function getProject(id: string): SavedProject | null {
  return read().find((p) => p.id === id) ?? null
}

/** Add or update a project. Opening one moves it to the top. */
export function saveProject(patch: Partial<SavedProject> & { id: string }, opened = false) {
  const list = read()
  const now = Date.now()
  const prev = list.find((p) => p.id === patch.id)
  if (!prev && !patch.token) return
  const next: SavedProject = { name: 'Untitled clip', createdAt: now, openedAt: now, token: '', ...prev, ...patch, ...(opened ? { openedAt: now } : {}) }
  write([next, ...list.filter((p) => p.id !== patch.id)])
}

export function forgetProject(id: string) {
  write(read().filter((p) => p.id !== id))
  try { localStorage.removeItem(STEPS_KEY(id)) } catch { /* nothing to clean */ }
}

/** Same array until the list changes; for useSyncExternalStore. */
export function projectsSnapshot(): SavedProject[] {
  return (cached ??= listProjects())
}

export function subscribeProjects(fn: () => void): () => void {
  listeners.add(fn)
  const onStorage = (e: StorageEvent) => { if (e.key === KEY) { cached = null; fn() } }
  window.addEventListener('storage', onStorage)
  return () => { listeners.delete(fn); window.removeEventListener('storage', onStorage) }
}

/** Give a project its own name; it shows in the top bar and on its card. */
export function renameProject(id: string, name: string) {
  const next = name.trim().slice(0, 80)
  if (next) saveProject({ id, name: next, named: true })
}

/**
 * A link that opens this project in any browser. The ID and token ride in the address's # part,
 * which browsers never send to a server; whoever has the link can edit the project.
 */
export function shareLink(p: Pick<SavedProject, 'id' | 'token' | 'name'>, base = `${window.location.origin}${window.location.pathname}`): string {
  const q = new URLSearchParams({ t: p.token, n: p.name })
  return `${base}#/app/open/${encodeURIComponent(p.id)}?${q.toString()}`
}

/** Read a link from shareLink: the project to add to this browser, or null. */
export function parseShareLink(hash: string): { id: string; token: string; name: string | null } | null {
  const m = /^#\/app\/open\/([^?#/]+)\?(.*)$/.exec(hash)
  if (!m) return null
  const q = new URLSearchParams(m[2])
  const token = q.get('t')
  const id = decodeURIComponent(m[1])
  if (!id || !token) return null
  return { id, token, name: q.get('n')?.trim() || null }
}

/** Add a shared project to this browser's list. Returns its editor address, or null for any other address. */
export function claimShareLink(hash: string): string | null {
  const link = parseShareLink(hash)
  if (!link) return null
  saveProject({ id: link.id, token: link.token, ...(link.name ? { name: link.name, named: true } : {}) }, true)
  return `#/app/p/${encodeURIComponent(link.id)}`
}

/** A project name as a file name: "Rain scene: final" becomes "Rain scene final". */
export function fileBase(name: string | null | undefined): string {
  return (name ?? '').replace(/[^\p{L}\p{N} _-]+/gu, ' ').replace(/\s+/g, ' ').trim().slice(0, 60) || 'tidepool'
}

/** "rain-scene_final.mp4" reads as "rain scene final". */
export function nameFromFile(filename: string | null | undefined): string {
  const base = (filename ?? '').replace(/\.[a-z0-9]{2,4}$/i, '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim()
  return base || 'Untitled clip'
}

export function loadSteps(id: string | null): StepStack {
  if (!id) return EMPTY_STEPS
  try { return parseSteps(JSON.parse(localStorage.getItem(STEPS_KEY(id)) ?? 'null')) } catch { return EMPTY_STEPS }
}

export function saveSteps(id: string | null, steps: StepStack) {
  if (!id) return
  try { localStorage.setItem(STEPS_KEY(id), JSON.stringify(steps)) } catch { /* undo still works in this tab */ }
}

/** A frame from a little way into the clip, as a small JPEG, for the project card. */
export async function captureThumb(src: string, width = 320): Promise<string | undefined> {
  const video = document.createElement('video')
  video.muted = true; video.preload = 'auto'; video.playsInline = true; video.src = src
  try {
    await new Promise<void>((resolve, reject) => { video.onloadeddata = () => resolve(); video.onerror = () => reject(new Error('no frame')); setTimeout(() => reject(new Error('timeout')), 8000) })
    const at = Math.min(1.5, (video.duration || 0) / 3)
    if (at > 0) await new Promise<void>((resolve) => { video.onseeked = () => resolve(); video.currentTime = at; setTimeout(resolve, 3000) })
    if (!video.videoWidth) return undefined
    const canvas = document.createElement('canvas')
    canvas.width = width; canvas.height = Math.round((width * video.videoHeight) / video.videoWidth)
    canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL('image/jpeg', 0.72)
  } catch { return undefined } finally { video.removeAttribute('src'); video.load() }
}

/** A file picked on the projects page, handed to the editor once it opens. */
let pending: File | null = null
export function setPendingUpload(file: File | null) { pending = file }
export function takePendingUpload(): File | null { const f = pending; pending = null; return f }
