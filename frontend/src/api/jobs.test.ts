import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MAX_POLL_FAILURES, pollJob, toExportStatus, toJobStatus } from './jobs'
import type { JobStatus } from './jobs'

describe('toJobStatus', () => {
  it('maps completed to done and keeps progress', () => {
    expect(toJobStatus({ status: 'completed', progress: 100 })).toEqual({ state: 'done', progress: 100 })
  })

  it('passes queued and processing through', () => {
    expect(toJobStatus({ status: 'queued', progress: 0 }).state).toBe('queued')
    expect(toJobStatus({ status: 'processing', progress: 5 }).state).toBe('processing')
  })

  it('keeps the error of a failed job and drops a null one', () => {
    expect(toJobStatus({ status: 'failed', progress: 5, error: 'boom' })).toEqual({
      state: 'failed',
      progress: 5,
      error: 'boom',
    })
    expect(toJobStatus({ status: 'queued', error: null })).toEqual({ state: 'queued' })
  })

  it('throws on a state it does not know', () => {
    expect(() => toJobStatus({ status: 'weird' })).toThrow('Unknown job status: weird')
  })

  it('carries a routing decision, and drops one it cannot read', () => {
    const routing = { mode: 'music_stems', engine: 'demucs', model_id: 'htdemucs_ft', reason: 'no prompts; music', scores: { music: 0.9 }, stems: ['vocals'] }
    expect(toJobStatus({ status: 'completed', routing }).routing).toEqual({ mode: 'music_stems', engine: 'demucs', modelId: 'htdemucs_ft', reason: 'no prompts; music', stems: ['vocals'] })
    expect(toJobStatus({ status: 'completed', routing: { mode: 'magic' } })).not.toHaveProperty('routing')
    expect(toJobStatus({ status: 'completed', routing: null })).not.toHaveProperty('routing')
  })
})

describe('toExportStatus', () => {
  it('maps an export the same way, so exports can use the same poll', () => {
    expect(toExportStatus({ status: 'queued' })).toEqual({ state: 'queued' })
    expect(toExportStatus({ status: 'processing' })).toEqual({ state: 'processing' })
    expect(toExportStatus({ status: 'completed', download_url: '/d/1' })).toEqual({ state: 'done' })
  })

  it('keeps the reason a failed export gave, and drops a null one', () => {
    expect(toExportStatus({ status: 'failed', error: 'ffmpeg exited 1' })).toEqual({
      state: 'failed',
      error: 'ffmpeg exited 1',
    })
    expect(toExportStatus({ status: 'processing', error: null })).toEqual({ state: 'processing' })
  })

  it('throws on a state it does not know, instead of waiting forever', () => {
    expect(() => toExportStatus({ status: 'cancelled' })).toThrow('Unknown job status: cancelled')
  })
})

describe('pollJob', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  const sequence = (...states: JobStatus[]) => {
    const fn = vi.fn<() => Promise<JobStatus>>()
    states.forEach((s) => fn.mockResolvedValueOnce(s))
    return fn
  }

  it('polls at the interval until done, reporting each update', async () => {
    const get = sequence({ state: 'queued' }, { state: 'processing', progress: 50 }, { state: 'done', progress: 100 })
    const onUpdate = vi.fn()
    const pending = pollJob(get, { intervalMs: 1000, onUpdate })

    await vi.advanceTimersByTimeAsync(0)
    expect(get).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(999)
    expect(get).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(get).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(1000)

    expect(await pending).toEqual({ state: 'done', progress: 100 })
    expect(onUpdate.mock.calls.map((c) => c[0].state)).toEqual(['queued', 'processing', 'done'])
  })

  it('stops polling after done', async () => {
    const get = sequence({ state: 'done' })
    await pollJob(get, { intervalMs: 10 })
    await vi.advanceTimersByTimeAsync(1000)
    expect(get).toHaveBeenCalledTimes(1)
  })

  it('resolves with a failed job and stops', async () => {
    const get = sequence({ state: 'processing' }, { state: 'failed', error: 'boom' })
    const pending = pollJob(get, { intervalMs: 10 })
    await vi.advanceTimersByTimeAsync(10)
    expect(await pending).toEqual({ state: 'failed', error: 'boom' })
    expect(get).toHaveBeenCalledTimes(2)
  })

  it('rides out a brief outage, since the job keeps running', async () => {
    const get = vi
      .fn<() => Promise<JobStatus>>()
      .mockRejectedValueOnce(new Error('502'))
      .mockRejectedValueOnce(new Error('502'))
      .mockResolvedValue({ state: 'done' })
    const pending = pollJob(get, { intervalMs: 10 })
    await vi.advanceTimersByTimeAsync(20)
    expect(await pending).toEqual({ state: 'done' })
  })

  it('gives up after repeated failures with the last error', async () => {
    const get = vi.fn<() => Promise<JobStatus>>().mockRejectedValue(new Error('net'))
    const pending = pollJob(get, { intervalMs: 10 })
    const caught = pending.catch((e: unknown) => e)
    await vi.advanceTimersByTimeAsync(MAX_POLL_FAILURES * 10)
    expect(((await caught) as Error).message).toBe('net')
    expect(get).toHaveBeenCalledTimes(MAX_POLL_FAILURES)
  })

  it('passes the abort signal to getStatus, so a hung request can be cancelled', async () => {
    const ctrl = new AbortController()
    const get = vi.fn(
      (signal?: AbortSignal) =>
        new Promise<JobStatus>((_res, rej) => {
          signal?.addEventListener('abort', () => rej(signal.reason))
        }),
    )
    const pending = pollJob(get, { signal: ctrl.signal })
    const caught = pending.catch((e: unknown) => e)
    await vi.advanceTimersByTimeAsync(0)
    ctrl.abort()
    expect(((await caught) as DOMException).name).toBe('AbortError')
    expect(get).toHaveBeenCalledTimes(1)
  })

  it('stops on abort during the wait', async () => {
    const ctrl = new AbortController()
    const get = vi.fn<() => Promise<JobStatus>>().mockResolvedValue({ state: 'processing' })
    const pending = pollJob(get, { intervalMs: 1000, signal: ctrl.signal })
    const caught = pending.catch((e: unknown) => e)
    await vi.advanceTimersByTimeAsync(500)
    ctrl.abort()
    expect((await caught as DOMException).name).toBe('AbortError')
    await vi.advanceTimersByTimeAsync(5000)
    expect(get).toHaveBeenCalledTimes(1)
  })

  it('does not report an update that arrives after abort', async () => {
    const ctrl = new AbortController()
    const onUpdate = vi.fn()
    const get = vi.fn(async () => {
      ctrl.abort()
      return { state: 'done' } as JobStatus
    })
    await expect(pollJob(get, { signal: ctrl.signal, onUpdate })).rejects.toBeDefined()
    expect(onUpdate).not.toHaveBeenCalled()
  })

  it('does not start when already aborted', async () => {
    const ctrl = new AbortController()
    ctrl.abort()
    const get = vi.fn<() => Promise<JobStatus>>()
    await expect(pollJob(get, { signal: ctrl.signal })).rejects.toBeDefined()
    expect(get).not.toHaveBeenCalled()
  })
})
