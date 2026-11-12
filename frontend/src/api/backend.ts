import { MAX_GAIN } from '../edits'
import type { Keyframe, MuteRange, Stem, TrackState } from '../types'

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
  revisions?: Record<string, number>
}

export interface BackendStem {
  id: string
  name: string
  duration: number
  audio_url: string
  start_offset?: number
  mute_ranges?: MuteRange[]
}

/** A row of GET /api/projects/{id}/stems: the stem as saved right now. */
export interface SavedStem extends BackendStem {
  revision: number
  volume: number
  muted: boolean
  solo: boolean
  automation: Array<{ time: number; value: number }>
}

export interface BackendJob {
  status: string
  progress?: number
  error?: string | null
  stems?: BackendStem[]
  /** Read from get_job in app/main.py. INTEGRATION.md does not list this field. */
  project_id?: string
  inference?: { model?: { is_mock?: boolean; backend?: string }; warnings?: string[] } | null
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
  timeoutMs = REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const timeout = AbortSignal.timeout(timeoutMs)
  const res = await fetch(toUrl(path, config.baseUrl), {
    ...init,
    signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
    headers: { ...init.headers, Authorization: `Bearer ${config.token}` },
  })
  if (res.ok) return res
  // FastAPI errors are {"detail": "message"}, or an array for validation errors.
  const body = (await res.json().catch(() => null)) as { detail?: unknown } | null
  const detail = typeof body?.detail === 'string' ? `: ${body.detail}` : Array.isArray(body?.detail) ? ': '+body.detail.map((e:{msg?:string})=>e.msg || 'Invalid input').join('; ') : ''
  throw new Error(`Backend returned ${res.status}${detail}`)
}

export async function getJob(config: BackendConfig, jobId: string, signal?: AbortSignal): Promise<BackendJob> {
  const res = await request(config, `/api/jobs/${encodeURIComponent(jobId)}`, {}, signal)
  return (await res.json()) as BackendJob
}

const inGainRange = (v: unknown): boolean => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= MAX_GAIN

function isSavedAutomation(raw: unknown): boolean {
  if (!Array.isArray(raw)) return false
  return raw.every((k: { time?: unknown; value?: unknown } | null, i) => {
    if (typeof k?.time !== 'number' || !Number.isFinite(k.time) || k.time < 0 || !inGainRange(k.value)) return false
    return i === 0 || k.time > (raw[i - 1] as { time: number }).time
  })
}

const isMuteRange = (r: unknown): boolean =>
  Array.isArray(r) && r.length === 2 && r.every((v) => typeof v === 'number' && Number.isFinite(v))

function isSavedStem(raw: Partial<SavedStem> | null): raw is SavedStem {
  return (
    typeof raw?.id === 'string' &&
    typeof raw.name === 'string' &&
    typeof raw.duration === 'number' &&
    typeof raw.audio_url === 'string' &&
    Number.isInteger(raw.revision) && (raw.revision ?? -1) >= 0 &&
    inGainRange(raw.volume) &&
    typeof raw.muted === 'boolean' &&
    typeof raw.solo === 'boolean' &&
    isSavedAutomation(raw.automation) &&
    typeof raw.start_offset === 'number' &&
    Number.isFinite(raw.start_offset) &&
    raw.start_offset >= 0 &&
    Array.isArray(raw.mute_ranges) &&
    raw.mute_ranges.every(isMuteRange)
  )
}

/**
 * The project's current stems. Throws on a row that does not have every field,
 * because a patch built from a guessed value would overwrite the saved one.
 */
export async function getSavedStems(config: BackendConfig, projectId: string): Promise<SavedStem[]> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/stems`)
  const rows = (await res.json()) as unknown
  if (!Array.isArray(rows) || !rows.every(isSavedStem)) throw new Error('Backend returned malformed stems')
  return rows
}

/** The editable state saved for a stem. Volume 0..2 is the same range as gain. */
export function toTrackState(stem: SavedStem): TrackState {
  return {
    stemId: stem.id,
    gain: stem.volume,
    muted: stem.muted,
    solo: stem.solo,
    automation: stem.automation.map(({ time, value }) => ({ t: time, value })),
  }
}

/** Audio needs the bearer header, so it cannot be a plain URL on the Stem. */
export async function fetchStemBlobUrl(config: BackendConfig, audioUrl: string, timeoutMs = REQUEST_TIMEOUT_MS): Promise<string> {
  const res = await request(config, audioUrl, {}, undefined, timeoutMs)
  return URL.createObjectURL(await res.blob())
}

export function toStem(stem: BackendStem, url: string): Stem {
  // The backend has no kind field, and kinds are not guessed from names.
  return {
    id: stem.id,
    label: stem.name,
    kind: 'sfx',
    url,
    duration: stem.duration,
    startOffset: stem.start_offset ?? 0,
    muteRanges: stem.mute_ranges ?? [],
  }
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
  expectedRevision?: number,
): Promise<number | undefined> {
  const path = `/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stemId)}`
  const response = await request(config, path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(expectedRevision === undefined ? body : { ...body, expected_revision: expectedRevision }),
  })
  if (expectedRevision === undefined) return undefined
  const result = await response.json() as { revision?: unknown }
  if (!Number.isInteger(result.revision) || (result.revision as number) <= expectedRevision) {
    throw new Error('Backend did not confirm the stem revision; reload before retrying')
  }
  return result.revision as number
}

export async function patchStem(config: BackendConfig, projectId: string, track: TrackState): Promise<void> {
  await sendStemPatch({ config, projectId }, track.stemId, toSyncPatch(track))
}

export interface ProjectInfo {
  project_id: string
  access_token?: string
  status: string
  filename?: string | null
  metadata?: { has_video: boolean; duration: number; audio_quality?:{findings:string[]} } | null
  latest_job?: { id: string; status: string } | null
  latest_visual_job?: { id: string; status: string } | null
  last_completed_job_id?: string | null
  visual_metadata?: unknown
}
export interface ExportInfo { status: string; error?: string | null; download_url?: string | null }

export async function createProject(baseUrl: string): Promise<ProjectInfo> {
  const res = await request({baseUrl, token: ''}, '/api/projects', {method: 'POST'})
  return res.json() as Promise<ProjectInfo>
}
export async function getProject(config: BackendConfig, projectId: string): Promise<ProjectInfo> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}`)
  return res.json() as Promise<ProjectInfo>
}
export async function uploadProject(config: BackendConfig, projectId: string, file: File): Promise<ProjectInfo> {
  const body = new FormData(); body.append('file', file)
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/upload`, {method:'POST', body}, undefined, 3_600_000)
  return res.json() as Promise<ProjectInfo>
}
export async function storeVisualMetadata(config: BackendConfig, projectId: string, metadata: unknown): Promise<void> {
  await request(config, `/api/projects/${encodeURIComponent(projectId)}/visual-metadata`, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(metadata)})
}
export async function processProject(config: BackendConfig, projectId: string, prompts: string[], visualPrompt?: unknown, append = false): Promise<string> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/process`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompts, append, visual_prompt: visualPrompt ?? {}})})
  return ((await res.json()) as {job_id:string}).job_id
}
export async function startExport(config: BackendConfig, projectId: string, format: 'mp4'|'wav'|'mp3'): Promise<string> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/export`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({format,include_video:format==='mp4'})})
  return ((await res.json()) as {export_id:string}).export_id
}
export async function getExport(config: BackendConfig, exportId: string, signal?: AbortSignal): Promise<ExportInfo> {
  const res = await request(config, `/api/exports/${encodeURIComponent(exportId)}`, {}, signal)
  return res.json() as Promise<ExportInfo>
}
export async function downloadExport(config: BackendConfig, url: string, filename: string): Promise<void> {
  const res = await request(config, url); const blobUrl = URL.createObjectURL(await res.blob())
  try { const a = document.createElement('a'); a.href = blobUrl; a.download = filename; a.click() }
  finally { setTimeout(() => URL.revokeObjectURL(blobUrl), 1000) }
}
export async function localCommand(config: BackendConfig, projectId: string, text: string): Promise<{status:string; message:string; export_id?:string}> {
  const res = await request(config, '/api/webhooks/local', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({project_id:projectId,token:config.token,text})})
  return res.json() as Promise<{status:string;message:string;export_id?:string}>
}

export interface VisualObject { detection_id: string; track_id: number | null; label: string; bbox_normalized: number[]; mask?: unknown; [key:string]: unknown }
export interface VisualFrame { frame_index:number; timestamp_seconds:number; valid_until_seconds:number; objects:VisualObject[] }
export interface VisualMetadata { schema_version:string; video_id:string; analysis_id:string; source_sha256:string; source:string; width:number; height:number; fps:number; processed_frames:number; processed_duration_seconds:number; frames:VisualFrame[] }
export async function startVisualAnalysis(config: BackendConfig, projectId: string): Promise<string> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/visual-analyze`,{method:'POST'})
  return ((await res.json()) as {job_id:string}).job_id
}
export async function getVisualJob(config: BackendConfig, jobId: string): Promise<BackendJob> {
  const res=await request(config,`/api/visual-jobs/${encodeURIComponent(jobId)}`)
  return res.json() as Promise<BackendJob>
}
export async function getVisualMetadata(config: BackendConfig, projectId: string): Promise<VisualMetadata> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/visual-metadata`)
  return res.json() as Promise<VisualMetadata>
}

export async function getMessages(config: BackendConfig, projectId: string): Promise<Array<{id:string;text:string}>> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/messages`)
  return res.json() as Promise<Array<{id:string;text:string}>>
}

export async function updateStemSettings(config: BackendConfig, projectId: string, stem: SavedStem, settings: {name:string;start_offset:number;mute_ranges?:MuteRange[]}): Promise<void> {
  await request(config, `/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stem.id)}`, {
    method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({...settings,expected_revision:stem.revision}),
  })
}

export interface AgentPlan { edit_scope?:EditScope;guidance?:{anchors:Anchor[]}; use_visual?:boolean; kind:'edit'|'separate'|'clarify'; description:string|null; remove:boolean; actions:unknown; message:string }
export async function planProjectCommand(config:BackendConfig, projectId:string, input: {prompt:string;stems:unknown[];tracks:unknown[];duration:number;has_project_stems:boolean;selected_visual?:{label:string;start:number;end:number}|null}):Promise<AgentPlan> {
  const response=await request(config,`/api/projects/${encodeURIComponent(projectId)}/agent`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)},undefined,90_000)
  return response.json() as Promise<AgentPlan>
}

export async function startPrepareVisual(config: BackendConfig, projectId:string):Promise<string> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/prepare-visual`,{method:'POST'})
  return ((await res.json()) as {job_id:string}).job_id
}
export async function getPrepareVisual(config:BackendConfig, jobId:string):Promise<BackendJob> {
  const res=await request(config,`/api/prepare-jobs/${encodeURIComponent(jobId)}`)
  return res.json() as Promise<BackendJob>
}

export interface Anchor {token:"+"|"-";start:number;end:number}
export interface EditScope {action:"keep"|"mute"|"lower";interval?:{start:number;end:number}|null}
export interface PreviewOptions {guidance?:{anchors:Anchor[]};edit_scope?:EditScope;visual_prompt?:unknown;profile?:"preview"|"quality"}
export interface Capabilities {quality:boolean;visual:boolean;reason:string}
export async function getCapabilities(config:BackendConfig,projectId:string):Promise<Capabilities> {
  return (await request(config,`/api/projects/${encodeURIComponent(projectId)}/capabilities`)).json() as Promise<Capabilities>
}
export interface CandidateInfo {mix_url?:string;original_url?:string;id:string;status:string;progress?:number;error?:string|null;audio_url?:string|null;inference?:{model?:{reranking_candidates?:number};warnings?:string[]}|null}
export async function createCandidate(config:BackendConfig,projectId:string,stemId:string|null,prompt:string,candidates:number,options:PreviewOptions={}):Promise<string> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/candidates`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem_id:stemId,prompt,reranking_candidates:candidates,...options})})
  return ((await res.json()) as {candidate_id:string}).candidate_id
}
export async function getCandidate(config:BackendConfig,id:string):Promise<CandidateInfo> {
  const res=await request(config,`/api/candidates/${encodeURIComponent(id)}`);return res.json() as Promise<CandidateInfo>
}
export async function applyCandidate(config:BackendConfig,id:string):Promise<{job_id:string}> {
  return (await request(config,`/api/candidates/${encodeURIComponent(id)}/apply`,{method:'POST'})).json() as Promise<{job_id:string}>
}
export async function undoCandidate(config:BackendConfig,id:string):Promise<{job_id:string}> {
  return (await request(config,`/api/candidates/${encodeURIComponent(id)}/undo`,{method:'POST'})).json() as Promise<{job_id:string}>
}

export async function addMuteRange(config:BackendConfig,projectId:string,stem:SavedStem,start:number,end:number):Promise<void> {
  await request(config,`/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stem.id)}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_revision:stem.revision,mute_ranges:[...(stem.mute_ranges || []),[start,end]]})})
}
