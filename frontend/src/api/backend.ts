import type { Keyframe, Stem, TrackState } from '../types'

// Client for the upstream FastAPI backend. Paths and shapes come from
// INTEGRATION.md and app/main.py at AkhilNam/SoundCanvas commit 9fc2216. They
// are DOCUMENTED there, but this code has only run against mocks, not a live
// server. Project tokens stay in memory: never in env vars, the bundle or storage.

export interface BackendConfig {
  /** For example http://localhost:8000, or /api behind the Vite proxy. */
  baseUrl: string
  /** Project access token from POST /api/projects. */
  token: string
}

/** Where edits to loaded stems are sent. */
export interface SyncTarget {
  config: BackendConfig
  projectId: string
}

export interface BackendStem {
  id: string
  name: string
  duration: number
  audio_url: string
}

export interface BackendJob {
  status: string
  progress?: number
  error?: string | null
  stems?: BackendStem[]
  /** Read from get_job in app/main.py. INTEGRATION.md does not list this field. */
  project_id?: string
}

/** Covers a whole request, including large stem downloads. */
const REQUEST_TIMEOUT_MS = 60_000

/** Paths already start with /api, so a same-origin base only supplies the origin. */
function toUrl(path: string, baseUrl: string): URL {
  return new URL(path, new URL(baseUrl, globalThis.location?.href))
}

async function request(
  config: BackendConfig,
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<Response> {
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  const res = await fetch(toUrl(path, config.baseUrl), {
    ...init,
    signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
    headers: { ...init.headers, Authorization: `Bearer ${config.token}` },
  })
  if (res.ok) return res
  // FastAPI errors are {"detail": "message"}, or an array for validation errors.
  const body = (await res.json().catch(() => null)) as { detail?: unknown } | null
  const detail = typeof body?.detail === 'string' ? `: ${body.detail}` : ''
  throw new Error(`Backend returned ${res.status}${detail}`)
}

export async function getJob(config: BackendConfig, jobId: string, signal?: AbortSignal): Promise<BackendJob> {
  const res = await request(config, `/api/jobs/${encodeURIComponent(jobId)}`, {}, signal)
  return (await res.json()) as BackendJob
}

/** Audio needs the bearer header, so it cannot be a plain URL on the Stem. */
export async function fetchStemBlobUrl(config: BackendConfig, audioUrl: string): Promise<string> {
  const res = await request(config, audioUrl)
  return URL.createObjectURL(await res.blob())
}

export function toStem(stem: BackendStem, url: string): Stem {
  // The backend has no kind field, and kinds are not guessed from names.
  return { id: stem.id, label: stem.name, kind: 'sfx', url, duration: stem.duration }
}

export interface StemPatch {
  volume: number
  muted: boolean
  solo: boolean
  automation: Array<{ time: number; value: number }>
}

/** The body editor sync sends. See toSyncPatch for when automation is left out. */
export type StemSyncPatch = Omit<StemPatch, 'automation'> & Partial<Pick<StemPatch, 'automation'>>

/** The body for PATCH /api/projects/{id}/stems/{stem_id}. Times pass through unchanged. */
export function trackToStemPatch(track: TrackState): StemPatch {
  return {
    volume: track.gain,
    muted: track.muted,
    solo: track.solo,
    automation: track.automation.map(({ t, value }) => ({ time: t, value })),
  }
}

/** Times that do not strictly increase, which the backend rejects. */
export function hasStepKeyframes(keyframes: Keyframe[]): boolean {
  return keyframes.some((k, i) => i > 0 && k.t <= keyframes[i - 1].t)
}

/**
 * What editor sync sends. Every edit goes through normalizeKeyframes, so the
 * curve here is the one the preview plays. The guard is defensive: nothing in
 * the editor can reach it (see stemSync.test.ts), but a curve the backend would
 * reject must not fail the whole write and take volume and mute down with it.
 */
export function toSyncPatch(track: TrackState): StemSyncPatch {
  const { automation, ...rest } = trackToStemPatch(track)
  return hasStepKeyframes(track.automation) ? rest : { ...rest, automation }
}

export async function sendStemPatch(
  { config, projectId }: SyncTarget,
  stemId: string,
  body: StemSyncPatch,
): Promise<void> {
  const path = `/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stemId)}`
  await request(config, path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function patchStem(config: BackendConfig, projectId: string, track: TrackState): Promise<void> {
  await sendStemPatch({ config, projectId }, track.stemId, toSyncPatch(track))
}
