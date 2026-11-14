import { toRouting } from './backend'
import type { BackendJob, ExportInfo, Routing } from './backend'

export type JobState = 'queued' | 'processing' | 'done' | 'failed'

export interface JobStatus {
  state: JobState
  /** 0 to 100, as the backend reports it. */
  progress?: number
  error?: string
  /** Which separator ran and why, when the backend says. */
  routing?: Routing
  /** What a running job is doing, such as `splitting`, when the backend says. */
  stage?: string
}

export const DEFAULT_POLL_MS = 1000
/** Jobs can run for minutes, so a brief outage should not end tracking. */
export const MAX_POLL_FAILURES = 3

/** The backend says `completed`, the UI type says `done`. Unknown states are errors, not guesses. */
export function toJobStatus(job: BackendJob): JobStatus {
  const { status, progress, error, stage } = job
  const routing = toRouting(job.routing)
  if (status !== 'queued' && status !== 'processing' && status !== 'completed' && status !== 'failed') {
    throw new Error(`Unknown job status: ${status}`)
  }
  return {
    state: status === 'completed' ? 'done' : status,
    ...(typeof progress === 'number' ? { progress } : {}),
    ...(error ? { error } : {}),
    ...(routing ? { routing } : {}),
    ...(typeof stage === 'string' && stage ? { stage } : {}),
  }
}

/** "0:07", "1:05", "12:00": minutes and seconds since a job started. */
export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

const STAGE_LABELS: Record<string, string> = { splitting: 'Splitting into tracks', separating: 'Separating the sound', rendering: 'Rendering the preview' }

/**
 * The working line for a separation job, with a running clock so a long call
 * never looks frozen: "Splitting into tracks… 0:12 · 45%". `fallback` names the
 * work when the backend reports no stage.
 */
export function jobProgressMessage(status: Pick<JobStatus, 'state' | 'progress' | 'stage'>, elapsedMs: number, fallback: string): string {
  const what = status.state === 'queued' ? 'Waiting for the separator' : (status.stage && STAGE_LABELS[status.stage]) || fallback
  const percent = status.state === 'processing' && status.progress !== undefined ? ` · ${status.progress}%` : ''
  return `${what}… ${formatElapsed(elapsedMs)}${percent}`
}

/** Exports report the same state names as jobs, and no progress. */
export function toExportStatus(info: ExportInfo): JobStatus {
  return toJobStatus({ status: info.status, ...(info.error ? { error: info.error } : {}) })
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

export interface PollOptions<S extends JobStatus = JobStatus> {
  intervalMs?: number
  signal?: AbortSignal
  onUpdate?: (status: S) => void
}

/**
 * Poll until the job is done or failed, and resolve with the final status. A
 * failed job resolves, since the job itself reports why. Rejects after
 * MAX_POLL_FAILURES consecutive getStatus errors, or with the signal's reason
 * when aborted.
 */
export async function pollJob<S extends JobStatus = JobStatus>(
  getStatus: (signal?: AbortSignal) => Promise<S>,
  opts: PollOptions<S> = {},
): Promise<S> {
  const { intervalMs = DEFAULT_POLL_MS, signal, onUpdate } = opts
  let failures = 0
  for (;;) {
    signal?.throwIfAborted()
    try {
      const status = await getStatus(signal)
      signal?.throwIfAborted()
      failures = 0
      onUpdate?.(status)
      if (status.state === 'done' || status.state === 'failed') return status
    } catch (err) {
      signal?.throwIfAborted()
      if (++failures >= MAX_POLL_FAILURES) throw err
    }
    await sleep(intervalMs, signal)
  }
}
