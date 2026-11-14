import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { getJob } from '../api/backend'
import { pollJob, toJobStatus } from '../api/jobs'
import type { JobStatus } from '../api/jobs'
import { HttpStemSource } from '../api/stemSources'
import type { StemSource } from '../api/stemSources'

interface Props {
  baseUrl: string
  onLoadStems: (source: StemSource, label: string) => void
}

const FIELD =
  'h-7 w-full rounded-md border border-line bg-tint/[0.03] px-2 font-mono text-[11px] text-fg outline-none placeholder:text-faint focus:border-accent/50'

/**
 * Follows a processing job on the backend and loads its stems when it finishes.
 * The project token stays in this component's memory. Nothing stores it.
 */
export function JobStatusCard({ baseUrl, onLoadStems }: Props) {
  const [jobId, setJobId] = useState('')
  const [token, setToken] = useState('')
  const [job, setJob] = useState<JobStatus | null>(null)
  const [error, setError] = useState('')
  const [polling, setPolling] = useState(false)
  const poll = useRef<AbortController | null>(null)

  useEffect(() => () => poll.current?.abort(), [])

  const start = async (e: FormEvent) => {
    e.preventDefault()
    const id = jobId.trim()
    if (!id || !token) return
    poll.current?.abort()
    const ctrl = new AbortController()
    poll.current = ctrl
    const config = { baseUrl, token }
    setError('')
    setJob(null)
    setPolling(true)
    try {
      const final = await pollJob(async (signal) => toJobStatus(await getJob(config, id, signal)), {
        signal: ctrl.signal,
        onUpdate: setJob,
      })
      if (final.state === 'done') onLoadStems(new HttpStemSource(config, id), 'Job stems')
    } catch (err) {
      if (ctrl.signal.aborted) return
      // Tracking stopped, so the last status no longer says anything about the job.
      setJob(null)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (!ctrl.signal.aborted) setPolling(false)
    }
  }

  const cancel = () => {
    poll.current?.abort()
    setPolling(false)
    setJob(null)
  }

  return (
    <div className="shrink-0 rounded-xl border border-line bg-surface/60 p-3 backdrop-blur">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-wider text-subtle">Job</span>
        {job && (
          <span className={`font-mono text-[10px] ${job.state === 'failed' ? 'text-red-600' : 'text-accent'}`}>
            {job.state}
            {job.progress !== undefined && ` ${job.progress}%`}
          </span>
        )}
      </div>
      <form onSubmit={start} className="space-y-1.5">
        <input value={jobId} onChange={(e) => setJobId(e.target.value)} placeholder="Job ID" className={FIELD} />
        <div className="flex gap-1.5">
          <input
            type="password"
            autoComplete="off"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Project token"
            className={FIELD}
          />
          {polling ? (
            <button
              type="button"
              onClick={cancel}
              className="h-7 shrink-0 rounded-md border border-line px-2.5 font-mono text-[11px] text-muted transition-colors hover:border-line-strong hover:text-fg"
            >
              Cancel
            </button>
          ) : (
            <button
              type="submit"
              disabled={!jobId.trim() || !token}
              className="h-7 shrink-0 rounded-md bg-accent px-2.5 font-mono text-[11px] font-medium text-white disabled:bg-tint/[0.06] disabled:text-faint"
            >
              Track
            </button>
          )}
        </div>
      </form>
      {(error || job?.error) && (
        <div className="mt-2 break-words font-mono text-[10px] text-red-600">{error || job?.error}</div>
      )}
    </div>
  )
}
