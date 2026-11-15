import { MAX_GAIN } from '../edits'
import type { Keyframe, MuteRange, Refinement, Stem, TrackState } from '../types'
import { readRotation } from '../rotation'

// Client for the upstream FastAPI backend. Paths and shapes come from
// INTEGRATION.md and app/main.py at AkhilNam/SoundCanvas commit 9fc2216. They
// are DOCUMENTED there, but this code has only run against mocks, not a live
// server. Project tokens are sent in bearer headers, never in env vars or the bundle.

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
  role?: string
  effect_group_id?: string | null
  effect_label?: string | null
  effect_asset_id?: string | null
  sound_group_id?: string | null
  sound_group_label?: string | null
  sound_start?: number | null
  sound_end?: number | null
  source_stem_id?: string
  prompt?: string
  label?: string
  spans?: unknown
  sample_rate?: number
  channels?: number
  format?: string
  /** Sent by backends that run multi-pass separation. Absent or null otherwise. */
  refinement?: BackendRefinement | null
  /** Instruments heard in a music stem, highest score first. Absent or null otherwise. */
  tags?: Array<{ label: string; score: number }> | null
}

export interface BackendRefinement {
  passes_requested: number
  passes_run: number
  per_pass_energy_db: number[]
  stopped_early: boolean
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
  /** What a running process job is doing: `splitting` or `separating`. */
  stage?: string | null
  stems?: BackendStem[]
  /** Read from get_job in app/main.py. INTEGRATION.md does not list this field. */
  project_id?: string
  inference?: { model?: { is_mock?: boolean; backend?: string }; warnings?: string[] } | null
  /** Which separator the audio service picked, and why. Absent from older backends and the prompted fake. */
  routing?: unknown
}

/** How the audio service separates: `auto` lets its router decide (a song becomes music stems). */
export type SeparationMode = 'auto' | 'prompted' | 'music_stems'

/** Stem names the music splitters use. The service narrows this to its model's own list. */
export const MUSIC_STEMS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'] as const

/** A routing decision, from a job's `routing` or from GET /route. */
export interface Routing {
  mode: SeparationMode
  engine: string | null
  modelId: string | null
  reason: string | null
  stems: string[] | null
}

/** Null for anything that is not a recognisable decision; it only feeds hints and a status line. */
export function toRouting(raw: unknown): Routing | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  if (r.mode !== 'auto' && r.mode !== 'prompted' && r.mode !== 'music_stems') return null
  const str = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : null)
  const stems = Array.isArray(r.stems) && r.stems.every((s) => typeof s === 'string') ? (r.stems as string[]) : null
  return { mode: r.mode, engine: str(r.engine), modelId: str(r.model_id), reason: str(r.reason), stems }
}

/** One line for a status card: "Music stems · demucs (htdemucs_ft): the content is music". */
export function describeRouting(r: Routing): string {
  const mode = r.mode === 'music_stems' ? 'Music stems' : r.mode === 'prompted' ? 'Prompted separation' : 'Automatic'
  const model = r.engine ? ` · ${r.engine}${r.modelId && r.modelId !== r.engine ? ` (${r.modelId})` : ''}` : ''
  return `${mode}${model}${r.reason ? `: ${r.reason}` : ''}`
}

/** Covers a whole request, including large stem downloads. */
const REQUEST_TIMEOUT_MS = 60_000

/** Paths already start with /api, so a same-origin base only supplies the origin. */
function toUrl(path: string, baseUrl: string): URL {
  return new URL(path, new URL(baseUrl, globalThis.location?.href))
}

export async function request(
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

/** The editable state saved for a stem. Volume 0..MAX_GAIN is the same range as gain. */
export function toTrackState(stem: SavedStem): TrackState {
  return {
    stemId: stem.id,
    gain: stem.volume,
    muted: stem.muted,
    solo: stem.solo,
    automation: stem.automation.map(({ time, value }) => ({ t: time, value })),
  }
}

/**
 * Audio needs the bearer header, so it cannot be a plain URL on the Stem.
 * Never from the HTTP cache: Extract more rewrites a stem's audio in place, at
 * the same URL, and a cached copy would play the old separation.
 */
/**
 * A Blob of the response body, built in the page. Chrome builds `Response.blob()` for a network
 * response in the browser process and spills large bodies to disk; when that fails (seen with a
 * nearly full disk on 2026-11-14: an 11 MB WAV) it rejects with a bare "Failed to fetch" even though
 * every byte arrived. Reading the bytes and wrapping them avoids that path.
 */
export async function responseBlob(res: Response): Promise<Blob> {
  return new Blob([await res.arrayBuffer()], { type: res.headers.get('content-type') ?? '' })
}

export async function fetchStemBlobUrl(config: BackendConfig, audioUrl: string, timeoutMs = REQUEST_TIMEOUT_MS): Promise<string> {
  const res = await request(config, audioUrl, { cache: 'no-store' }, undefined, timeoutMs)
  return URL.createObjectURL(await responseBlob(res))
}

const isCount = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 1

/**
 * The backend's refinement block, or undefined when it is absent or does not
 * make sense. It only feeds a badge, so a bad one is dropped rather than fatal.
 */
export function toRefinement(raw: unknown): Refinement | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const r = raw as Partial<Record<keyof BackendRefinement, unknown>>
  if (!isCount(r.passes_run)) return undefined
  return {
    passesRequested: isCount(r.passes_requested) ? r.passes_requested : r.passes_run,
    passesRun: r.passes_run,
    perPassEnergyDb: Array.isArray(r.per_pass_energy_db)
      ? r.per_pass_energy_db.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      : [],
    stoppedEarly: r.stopped_early === true,
  }
}

export function toStem(stem: BackendStem, url: string): Stem {
  const refinement = toRefinement(stem.refinement)
  // The backend has no kind field, and kinds are not guessed from names.
  const label = stem.label?.trim()
  const tags = Array.isArray(stem.tags) ? stem.tags.map((t) => (typeof t?.label === 'string' ? t.label.trim() : '')).filter(Boolean) : []
  return {
    id: stem.id,
    // A music split labels stems canonically ("vocals") and names them for display ("Vocals").
    label: label && (MUSIC_STEMS as readonly string[]).includes(label) && stem.name.trim() ? stem.name.trim() : label || stem.name,
    kind: stem.role === 'residual' ? 'residual' : 'sfx',
    url,
    duration: stem.duration,
    startOffset: stem.start_offset ?? 0,
    muteRanges: stem.mute_ranges ?? [],
    ...(stem.role === 'effect' && stem.effect_group_id ? { effectGroupId: stem.effect_group_id, effectLabel: stem.effect_label || stem.name, isEffectClip:true } : {}),
    ...(stem.role === 'target' && stem.sound_group_id ? {effectGroupId:stem.sound_group_id,effectLabel:stem.sound_group_label || stem.name,...(typeof stem.sound_start==='number' && typeof stem.sound_end==='number' && stem.sound_start>=0 && stem.sound_end>stem.sound_start ? {soundInterval:{start:stem.sound_start,end:stem.sound_end}} : {})} : {}),
    ...(refinement ? { refinement } : {}),
    ...(tags.length ? { tags } : {}),
  }
}

/** The tags worth showing beside a track's name: "Strings · Synthesizer" for a track named Piano. */
export function tagLine(name: string, tags: readonly string[] = []): string {
  return tags.filter((t) => t.toLowerCase() !== name.trim().toLowerCase()).join(' · ')
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
  latest_undoable_candidate_id?: string | null
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
/**
 * The `passes` field for a separation request. One pass is the backend's
 * default, so it is left out and a backend without multi-pass support still
 * accepts the request.
 */
export function passesField(passes = 1): { passes?: number } {
  return passes > 1 ? { passes } : {}
}

/** How a /process request routes. Left out entirely, the service's default (auto) applies. */
export interface ProcessRouting { mode?: SeparationMode; stems?: string[] | null; edit_scope?: EditScope; /** Explode: the server runs the tuned multi-layer recipe in one job. */ explode?: boolean; /** Explode only: split music layers into instruments. */ splitMusic?: boolean }
export async function processProject(config: BackendConfig, projectId: string, prompts: string[], visualPrompt?: unknown, append = false, passes = 1, routing: ProcessRouting = {}): Promise<string> {
  const {mode, stems, edit_scope, explode, splitMusic} = routing
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/process`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompts, ...(append ? {append:true} : {}), ...(explode ? {explode:true} : {}), ...(explode && splitMusic ? {split_music:true} : {}), ...passesField(passes), visual_prompt: visualPrompt ?? {}, ...(mode ? {mode} : {}), ...(stems?.length ? {stems} : {}), ...(edit_scope ? {edit_scope} : {})})})
  return ((await res.json()) as {job_id:string}).job_id
}
/** Split a song into one track per instrument: all of them, or `stems` plus one residual. */
export function splitTracks(config: BackendConfig, projectId: string, stems: string[] | null = null, editScope?: EditScope): Promise<string> {
  return processProject(config, projectId, [], undefined, false, 1, {mode: 'music_stems', stems, ...(editScope ? {edit_scope: editScope} : {})})
}
/** The audio service's dry-run routing decision for the upload. Null when it gives none. */
export async function getRoute(config: BackendConfig, projectId: string, signal?: AbortSignal): Promise<Routing | null> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/route`, {}, signal, 120_000)
  return toRouting(await res.json())
}
export async function startExport(config: BackendConfig, projectId: string, format: 'mp4'|'wav'|'mp3'): Promise<string> {
  // An MP4 is turned to match the editor's picture; audio-only exports have no picture to turn.
  const rotation = format==='mp4' ? readRotation(projectId) ?? 0 : 0
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/export`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({format,include_video:format==='mp4',...(rotation ? {rotation} : {})})})
  return ((await res.json()) as {export_id:string}).export_id
}
export async function getExport(config: BackendConfig, exportId: string, signal?: AbortSignal): Promise<ExportInfo> {
  const res = await request(config, `/api/exports/${encodeURIComponent(exportId)}`, {}, signal)
  return res.json() as Promise<ExportInfo>
}
export async function downloadExport(config: BackendConfig, url: string, filename: string): Promise<void> {
  const res = await request(config, url); const blobUrl = URL.createObjectURL(await responseBlob(res))
  try { const a = document.createElement('a'); a.href = blobUrl; a.download = filename; a.click() }
  finally { setTimeout(() => URL.revokeObjectURL(blobUrl), 1000) }
}
export async function localCommand(config: BackendConfig, projectId: string, text: string): Promise<{status:string; message:string; export_id?:string}> {
  const res = await request(config, '/api/webhooks/local', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({project_id:projectId,token:config.token,text})})
  return res.json() as Promise<{status:string;message:string;export_id?:string}>
}

export interface DiscoveredSound {label:string;prompt:string;intervals:Array<{start:number;end:number}>;score?:number}
export interface DiscoveryJob {status:string;error?:string|null;result?:{sources:DiscoveredSound[];notice:string;model:string;is_mock:boolean}|null}
export async function startSoundDiscovery(config:BackendConfig,projectId:string):Promise<string> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/discover`,{method:'POST'})
  return ((await res.json()) as {job_id:string}).job_id
}
export async function getSoundDiscovery(config:BackendConfig,id:string):Promise<DiscoveryJob> {
  return (await request(config,`/api/discovery-jobs/${encodeURIComponent(id)}`)).json() as Promise<DiscoveryJob>
}

export interface VisualObject { detection_id: string; track_id: number | null; label: string; bbox_normalized: number[]; mask?: unknown; [key:string]: unknown }
export interface VisualFrame { frame_index:number; timestamp_seconds:number; valid_until_seconds:number; objects:VisualObject[] }
export interface VisualMetadata { discovery_mode?:'automatic'|'prompted'; selection_only?:boolean; schema_version:string; video_id:string; analysis_id:string; source_sha256:string; source:string; width:number; height:number; fps:number; processed_frames:number; processed_duration_seconds:number; frames:VisualFrame[] }
/** POST /visual-analyze answers 202 with the job, and says whether it had to re-encode a compatible copy first. */
export interface VisualAnalysisStart { jobId: string; prepared: boolean }
export async function startVisualAnalysis(config: BackendConfig, projectId: string, concept = 'auto'): Promise<VisualAnalysisStart> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/visual-analyze`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({concept})})
  const body=(await res.json()) as {job_id?:unknown;prepared?:unknown}
  if (typeof body?.job_id !== 'string' || !body.job_id) throw new Error('Backend did not return a visual job id')
  return {jobId:body.job_id, prepared:body.prepared === true}
}
/** A visual job also reports which stage it is in. Both fields are optional. */
export interface VisualJob extends BackendJob { stage?: string }
export async function getVisualJob(config: BackendConfig, jobId: string, signal?: AbortSignal): Promise<VisualJob> {
  const res=await request(config,`/api/visual-jobs/${encodeURIComponent(jobId)}`,{},signal)
  return res.json() as Promise<VisualJob>
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

/**
 * Rename a layer. The server sets the label, and the name too unless it is the residual (which it
 * finds by name). Returns the layer's new revision, so later fader saves carry the right one.
 */
export async function renameStem(config: BackendConfig, projectId: string, stemId: string, label: string, expectedRevision?: number): Promise<number | undefined> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stemId)}`, {
    method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({rename:label,...(expectedRevision === undefined ? {} : {expected_revision:expectedRevision})}),
  })
  const revision = ((await res.json()) as { revision?: unknown }).revision
  return Number.isInteger(revision) ? revision as number : undefined
}

/**
 * Remove a layer. fold=true adds its sound to the residual, so the mix still has it; fold=false
 * takes the sound out too. Returns the job to load the remaining layers from.
 */
export async function deleteStem(config: BackendConfig, projectId: string, stemId: string, fold: boolean): Promise<{ job_id: string; folded: boolean }> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stemId)}?fold=${fold}`, { method:'DELETE' })
  return (await res.json()) as { job_id: string; folded: boolean }
}

/** A sound the model heard in the clip: a separation-ready prompt, and the 10-second windows it was heard in. */
export interface HeardSound { prompt:string; group:string; tier:string; score:number; windows:number[] }
export interface SoundSuggestions { sounds:HeardSound[]; is_mock:boolean; window_seconds?:number; reason?:string }
export function toSoundSuggestions(raw:unknown):SoundSuggestions {
  const r=(raw && typeof raw==='object' ? raw : {}) as Record<string,unknown>
  const sounds=Array.isArray(r.sounds) ? r.sounds.filter((s):s is HeardSound=>Boolean(s) && typeof (s as HeardSound).prompt==='string' && Boolean((s as HeardSound).prompt.trim())).map(s=>({prompt:s.prompt.trim(),group:String(s.group??''),tier:String(s.tier??''),score:Number(s.score)||0,windows:Array.isArray(s.windows)?s.windows.filter(Number.isInteger):[]})) : []
  return {sounds,is_mock:r.is_mock===true,...(typeof r.window_seconds==='number'?{window_seconds:r.window_seconds}:{}),...(typeof r.reason==='string'&&r.reason?{reason:r.reason}:{})}
}
/** Ask the model which sounds are in the project's audio. */
export async function suggestSounds(config:BackendConfig, projectId:string):Promise<SoundSuggestions> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/suggest-sounds`,{method:'POST'},undefined,300_000)
  return toSoundSuggestions(await res.json())
}

export interface AgentPlan { processing_time?:number|null; edit_scope?:EditScope;guidance?:{anchors:Anchor[]}; use_visual?:boolean; kind:'edit'|'separate'|'cinematic'|'director'|'replacement'|'censorship'|'clip_edit'|'clarify'; description:string|null; remove:boolean; actions:unknown; message:string; separation_mode?:SeparationMode|null; stems?:string[]|null;clip_edit?:{stem_id:string;operation:'move';start_offset:number} }
export async function planProjectCommand(config:BackendConfig, projectId:string, input: {prompt:string;stems:unknown[];tracks:unknown[];duration:number;has_project_stems:boolean;selected_visual?:{label:string;start:number;end:number}|null}):Promise<AgentPlan> {
  const response=await request(config,`/api/projects/${encodeURIComponent(projectId)}/agent`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)},undefined,90_000)
  return response.json() as Promise<AgentPlan>
}

export async function startPrepareVisual(config: BackendConfig, projectId:string):Promise<string> {
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/prepare-visual`,{method:'POST'})
  return ((await res.json()) as {job_id:string}).job_id
}
export async function getPrepareVisual(config:BackendConfig, jobId:string, signal?:AbortSignal):Promise<BackendJob> {
  const res=await request(config,`/api/prepare-jobs/${encodeURIComponent(jobId)}`,{},signal)
  return res.json() as Promise<BackendJob>
}

export interface Anchor {token:"+"|"-";start:number;end:number}
export interface EditScope {action:"keep"|"mute"|"lower";interval?:{start:number;end:number}|null}
export interface ObjectSelectionContext {analysis_id:string;detection_id:string;track_id:number;frame_index:number}
export interface PreviewOptions {source_stem_id?:string;object_selection?:ObjectSelectionContext;processing_interval?:{start:number;end:number};processing_time?:number;guidance?:{anchors:Anchor[]};edit_scope?:EditScope;visual_prompt?:unknown;profile?:"preview"|"quality";passes?:number}
/** The object detector, as GET /capabilities reports it. Absent fields are null, not guessed. */
export interface VisionCapability {automatic?:boolean;prompted?:boolean;retryable?:boolean;selectionOnly?:boolean;available:boolean;reason:string|null;model:string|null;weightsCached:boolean|null;device:string|null}
/**
 * `visual` is SAM Audio's visual prompting; `vision` is the object detector. An older backend sends no `vision`.
 * max_passes and refine come from backends with multi-pass separation; older ones omit them.
 */
export interface Capabilities {quality:boolean;visual:boolean;reason:string;spanAnchors?:boolean;vision?:VisionCapability;max_passes?:number;refine?:boolean;backend?:string;modelId?:string;loaded?:boolean;preserveVoices?:boolean;musicStems?:{available:boolean;stems:string[]};suggest?:boolean;explode?:boolean}
const text=(v:unknown):string|null=>typeof v==='string' && v.trim() ? v.trim() : null
export function toVisionCapability(raw:unknown):VisionCapability|undefined {
  if(!raw || typeof raw!=='object') return undefined
  const v=raw as Record<string,unknown>
  if(typeof v.available!=='boolean') return undefined
  return {...(typeof v.automatic==='boolean'?{automatic:v.automatic}:{}),...(typeof v.prompted==='boolean'?{prompted:v.prompted}:{}),...(typeof v.retryable==='boolean'?{retryable:v.retryable}:{}),...(typeof v.selection_only==='boolean'?{selectionOnly:v.selection_only}:{}),available:v.available,reason:text(v.reason),model:text(v.model),weightsCached:typeof v.weights_cached==='boolean'?v.weights_cached:null,device:text(v.device)}
}
export function toCapabilities(raw:unknown):Capabilities {
  const c=(raw && typeof raw==='object' ? raw : {}) as Record<string,unknown>
  const vision=toVisionCapability(c.vision)
  const suppression=c.suppression && typeof c.suppression==='object' ? c.suppression as Record<string,unknown> : null
  return {
    quality:c.quality===true,visual:c.visual===true,reason:typeof c.reason==='string'?c.reason:'',
    ...(typeof c.span_anchors==='boolean'?{spanAnchors:c.span_anchors}:{}),
    ...(vision?{vision}:{}),
    ...(typeof c.max_passes==='number'?{max_passes:c.max_passes}:{}),
    ...(typeof c.refine==='boolean'?{refine:c.refine}:{}),
    ...(text(c.backend)?{backend:text(c.backend)!}:{}),
    ...(text(c.model_id)?{modelId:text(c.model_id)!}:{}),
    ...(typeof c.loaded==='boolean'?{loaded:c.loaded}:{}),
    ...(suppression?{preserveVoices:suppression.mode==='conservative'}:{}),
    ...(typeof c.suggest==='boolean'?{suggest:c.suggest}:{}),
    ...(typeof c.explode==='boolean'?{explode:c.explode}:{}),
    ...(c.music_stems && typeof c.music_stems==='object' ? {musicStems:toMusicStems(c.music_stems as Record<string,unknown>)} : {}),
  }
}
function toMusicStems(m:Record<string,unknown>):{available:boolean;stems:string[]} {
  return {available:m.available===true,stems:Array.isArray(m.stems)?m.stems.filter((s):s is string=>typeof s==='string'):[]}
}
export async function getCapabilities(config:BackendConfig,projectId:string):Promise<Capabilities> {
  return toCapabilities(await (await request(config,`/api/projects/${encodeURIComponent(projectId)}/capabilities`)).json())
}
export type CinematicStyle = 'subtle'|'cinematic'|'dramatic'
export interface LayerEffects {bass:number;treble:number;highpass:number;compression:number}
export async function editEffectClip(config:BackendConfig,projectId:string,stem:SavedStem,operation:'move'|'delete',startOffset?:number):Promise<string> {
  const response=await request(config,`/api/projects/${encodeURIComponent(projectId)}/effect-clip-edit`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem_id:stem.id,expected_revision:stem.revision,operation,...(startOffset===undefined ? {} : {start_offset:startOffset})})})
  return ((await response.json()) as {candidate_id:string}).candidate_id
}
export async function createEffectsPreview(config:BackendConfig,projectId:string,stemId:string|null,effects:LayerEffects,groupId?:string):Promise<string> {
  const response=await request(config,`/api/projects/${encodeURIComponent(projectId)}/effects-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem_id:stemId,...(groupId ? {effect_group_id:groupId} : {}),...effects})})
  return ((await response.json()) as {candidate_id:string}).candidate_id
}
export type DirectorIntent = 'cinematic'|'dramatic'|'cool'|'funny'
export async function createDirectorPreview(config:BackendConfig,projectId:string,requestText:string,intent:DirectorIntent,style:CinematicStyle):Promise<string> {
  const response=await request(config,`/api/projects/${encodeURIComponent(projectId)}/director-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request:requestText,intent,style})})
  return ((await response.json()) as {candidate_id:string}).candidate_id
}
export async function createCinematicPreview(config:BackendConfig, projectId:string, style:CinematicStyle):Promise<string> {
  const response=await request(config,`/api/projects/${encodeURIComponent(projectId)}/cinematic-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({style})})
  return ((await response.json()) as {candidate_id:string}).candidate_id
}
export interface DirectorChange {label:string;action:string;intervals:Array<{start:number;end:number}>;target_stem_id:string;enabled:boolean}
export async function adjustDirectorPreview(config:BackendConfig,candidateId:string,stemId:string,enabled:boolean):Promise<string> {
  const response=await request(config,`/api/candidates/${encodeURIComponent(candidateId)}/director-adjust`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem_id:stemId,enabled})})
  return ((await response.json()) as {candidate_id:string}).candidate_id
}
export interface ReplacementRecovery {asset_id:string;effect_label:string;source_prompt:string;mode:'add';intervals:Array<{start:number;end:number}>}
export interface ReplacementEvent {id:string;group:number|null;source_peak:number;source_onset:number;offset:number;enabled:boolean}
export type ReplacementAdjustment = {operation:'alignment';alignment:'peak'|'onset'} | {operation:'toggle';event_id:string;enabled:boolean} | {operation:'move';event_id:string;time:number} | {operation:'add';time:number}
export async function adjustReplacementPreview(config:BackendConfig,candidateId:string,adjustment:ReplacementAdjustment):Promise<string> {
  const response=await request(config,`/api/candidates/${encodeURIComponent(candidateId)}/replacement-adjust`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(adjustment)})
  return ((await response.json()) as {candidate_id:string}).candidate_id
}
export interface CandidateInfo {recovery?:ReplacementRecovery|null;provenance?:{matches?:Array<{word:string;start:number;end:number}>;events?:ReplacementEvent[];alignment?:'peak'|'onset';peak_times?:number[];mode?:'replace'|'add';intervals?:Array<{start:number;end:number}>;requested_count?:number;changes?:DirectorChange[];kind?:string;style?:CinematicStyle;intent?:DirectorIntent;request?:string;steps?:string[];analysis?:{mean_dbfs?:number;peak_dbfs?:number};processing_interval?:{start:number;end:number}|null;processing_time?:number|null;localization?:{method:string;notice?:string}|null};mix_url?:string;original_url?:string;id:string;status:string;progress?:number;stage?:string|null;error?:string|null;audio_url?:string|null;inference?:{model?:{reranking_candidates?:number;is_mock?:boolean};warnings?:string[]}|null}
export async function createCandidate(config:BackendConfig,projectId:string,stemId:string|null,prompt:string,candidates:number,options:PreviewOptions={}):Promise<string> {
  const {passes,...rest}=options
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/candidates`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem_id:stemId,prompt,reranking_candidates:candidates,...rest,...passesField(passes)})})
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

/**
 * Rerun a target stem's prompt on the residual (Extract more). Resolves with the
 * job to poll. A 409 means the stem cannot be refined right now; its detail
 * says why (see refineConflictMessage).
 */
export async function startRefine(config:BackendConfig,projectId:string,stemId:string,passes?:number,signal?:AbortSignal):Promise<string> {
  const init:RequestInit=passes===undefined?{method:'POST'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({passes})}
  const res=await request(config,`/api/projects/${encodeURIComponent(projectId)}/stems/${encodeURIComponent(stemId)}/refine`,init,signal)
  const body=(await res.json()) as {job_id?:unknown}
  if(typeof body.job_id!=='string' || !body.job_id)throw new Error('Backend did not return a refine job')
  return body.job_id
}

export interface SoundEffectAsset {asset_id:string;duration:number;source:string}
export async function uploadSoundEffect(config:BackendConfig,projectId:string,file:File):Promise<SoundEffectAsset> {
  const data=new FormData();data.append('file',file)
  return (await (await request(config,`/api/projects/${encodeURIComponent(projectId)}/sound-effects/upload`,{method:'POST',body:data})).json()) as SoundEffectAsset
}
export async function generateSoundEffect(config:BackendConfig,projectId:string,prompt:string):Promise<string> {
  const result=await request(config,`/api/projects/${encodeURIComponent(projectId)}/sound-effects/generate`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})})
  return ((await result.json()) as {job_id:string}).job_id
}
export async function getSoundEffectJob(config:BackendConfig,id:string):Promise<{status:string;error?:string;result?:SoundEffectAsset}> {
  return (await (await request(config,`/api/sound-effect-jobs/${encodeURIComponent(id)}`)).json()) as {status:string;error?:string;result?:SoundEffectAsset}
}
export async function createReplacementPreview(config:BackendConfig,projectId:string,payload:{asset_id:string;effect_label:string;source_prompt:string;source_stem_id?:string;mode:'replace'|'add';intervals:Array<{start:number;end:number}>}):Promise<string> {
  return ((await (await request(config,`/api/projects/${encodeURIComponent(projectId)}/replacement-preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json()) as {candidate_id:string}).candidate_id
}

export async function createAutomaticReplacement(config:BackendConfig,projectId:string,text:string,sourceStemId?:string):Promise<string> {
  return ((await (await request(config,`/api/projects/${encodeURIComponent(projectId)}/automatic-replacement`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request:text,...(sourceStemId ? {source_stem_id:sourceStemId} : {})})})).json()) as {candidate_id:string}).candidate_id
}

export async function createCensorshipPreview(config:BackendConfig,projectId:string):Promise<string> {
  return ((await (await request(config,`/api/projects/${encodeURIComponent(projectId)}/censorship-preview`,{method:'POST'})).json()) as {candidate_id:string}).candidate_id
}
