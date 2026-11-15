import type { BackendConfig } from '../api/backend'
import type { Effect } from './rules'

// Client for app/moonpull_routes.py. Times are timeline seconds; the server converts them to
// each layer's own time.

export interface TimedText { start: number; end: number; text: string }
export interface Transcript { text: string; lines: TimedText[]; words: TimedText[] }

const TIMEOUT_MS = 180_000

async function call<T>(config: BackendConfig, path: string, body?: unknown): Promise<T> {
  const url = new URL(path, new URL(config.baseUrl, globalThis.location?.href))
  const res = await fetch(url, {
    method: body === undefined ? 'GET' : 'POST',
    signal: AbortSignal.timeout(TIMEOUT_MS),
    headers: { Authorization: `Bearer ${config.token}`, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    const data = (await res.json().catch(() => null)) as { detail?: unknown } | null
    const detail = typeof data?.detail === 'string' ? data.detail : Array.isArray(data?.detail) ? 'The server rejected the rule.' : `The server returned ${res.status}.`
    throw new Error(detail)
  }
  return (await res.json()) as T
}

const base = (projectId: string, stemId: string) => `/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stemId)}/moonpull`

export const applyEffect = (c: BackendConfig, pid: string, sid: string, body: { effect: Effect; ranges: Array<[number, number]>; amount: number; rule: string }) =>
  call<{ moonpull_file: string }>(c, `${base(pid, sid)}/effect`, body)

export const applyEcho = (c: BackendConfig, pid: string, sid: string, body: { source_stem_id: string; source_range: [number, number] | null; times: number[]; gain: number; rule: string }) =>
  call<{ moonpull_file: string }>(c, `${base(pid, sid)}/echo`, body)

/** Adds a layer that plays the sound during each range. job_id names the layers to load next. */
export const addLayer = (c: BackendConfig, pid: string, body: { asset_id: string; name: string; ranges: Array<[number, number]>; gain: number; rule: string }) =>
  call<{ stem: { id: string }; job_id: string }>(c, `/api/projects/${encodeURIComponent(pid)}/moonpull/layer`, body)

export const revert = (c: BackendConfig, pid: string, sid: string, file: string) => call<unknown>(c, `${base(pid, sid)}/revert`, { file })

export const findMotif = (c: BackendConfig, pid: string, sid: string, seconds = 2) =>
  call<{ start: number; end: number }>(c, `${base(pid, sid)}/motif?seconds=${seconds}`)

export const transcribe = (c: BackendConfig, pid: string, sid: string) => call<Transcript>(c, `${base(pid, sid)}/transcript`, {})

/** The job a withLayers result says to load the layers from, when its work changed which layers exist. */
export function layersJob(result: unknown): string | null {
  const id = result && typeof result === 'object' ? (result as { layersJobId?: unknown }).layersJobId : undefined
  return typeof id === 'string' && id ? id : null
}
