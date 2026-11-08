import type { BackendJob } from './backend'

export type JobState = 'queued' | 'processing' | 'done' | 'failed'

export interface JobStatus {
  state: JobState
  /** 0 to 100, as the backend reports it. */
  progress?: number
  error?: string
}

export const DEFAULT_POLL_MS = 1000
/** Jobs can run for minutes, so a brief outage should not end tracking. */
export const MAX_POLL_FAILURES = 3

/** The backend says `completed`, the UI type says `done`. Unknown states are errors, not guesses. */
export function toJobStatus(job: BackendJob): JobStatus {
  const { status, progress, error } = job
  if (status !== 'queued' && status !== 'processing' && status !== 'completed' && status !== 'failed') {
    throw new Error(`Unknown job status: ${status}`)
  }
  return {
    state: status === 'completed' ? 'done' : status,
    ...(typeof progress === 'number' ? { progress } : {}),
    ...(error ? { error } : {}),
  }
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

export interface PollOptions {
  intervalMs?: number
  signal?: AbortSignal
  onUpdate?: (status: JobStatus) => void
}

/**
 * Poll until the job is done or failed, and resolve with the final status. A
 * failed job resolves, since the job itself reports why. Rejects after
 * MAX_POLL_FAILURES consecutive getStatus errors, or with the signal's reason
 * when aborted.
 */
export async function pollJob(
  getStatus: (signal?: AbortSignal) => Promise<JobStatus>,
  opts: PollOptions = {},
): Promise<JobStatus> {
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
