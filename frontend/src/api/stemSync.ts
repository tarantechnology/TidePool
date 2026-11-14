import type { TrackState } from '../types'
import { sendStemPatch, toSyncPatch } from './backend'
import type { SyncTarget } from './backend'

/** Longer than the 400ms undo coalescing window, so one slider drag is one write. */
export const SYNC_DEBOUNCE_MS = 500

function bodyOf(track: TrackState): string {
  return JSON.stringify(toSyncPatch(track))
}

/**
 * Sends track edits to the backend. Each write carries the stem's whole state,
 * and a stem has one request in flight at most, so writes cannot land out of
 * order and the last one wins. Local tracks are never touched: a failure is
 * reported through onError, and the stem goes out again on the next change.
 */
export class StemSync {
  private target: SyncTarget
  private onError: (message: string) => void
  private latest = new Map<string, TrackState>()
  /** Last body the backend holds, by stem id. Starts as the state at load. Unknown after a failure. */
  private acked = new Map<string, string>()
  private inflight = new Set<Promise<void>>()
  private timers = new Map<string, ReturnType<typeof setTimeout>>()
  private sending = new Set<string>()
  private failures = new Map<string, string>()
  private closed = false
  private after: Promise<unknown>
  private revisions: Record<string, number> | undefined

  /** Writes wait for after, so a sync replacing another cannot overlap its writes. */
  constructor(
    target: SyncTarget,
    baseline: TrackState[],
    onError: (message: string) => void,
    after: Promise<unknown> = Promise.resolve(),
  ) {
    this.target = target
    this.revisions = target.revisions ? { ...target.revisions } : undefined
    this.onError = onError
    this.after = after
    for (const track of baseline) {
      this.latest.set(track.stemId, track)
      this.acked.set(track.stemId, bodyOf(track))
    }
  }

  update(tracks: TrackState[]): void {
    for (const track of tracks) {
      const id = track.stemId
      const changed = track !== this.latest.get(id)
      this.latest.set(id, track)
      // A stem with a timer running keeps it unless it changed, so a drag on one
      // stem cannot hold back another stem's write.
      if (changed || !this.timers.has(id)) this.schedule(id)
    }
  }

  /** Stop debouncing and send what is still unsent, once. Failures are still reported. */
  close(): void {
    this.closed = true
    this.failures.clear()
    for (const timer of this.timers.values()) clearTimeout(timer)
    this.timers.clear()
    for (const id of this.latest.keys()) this.send(id)
    this.report()
  }

  /** Send what is waiting on the debounce now, and resolve when it has landed. */
  async flush(): Promise<void> {
    // Retry failed unsaved edits once when explicitly flushing.
    for (const id of this.latest.keys()) if (!this.timers.has(id)) this.send(id)
    do {
      for (const [id, timer] of this.timers) {
        clearTimeout(timer)
        this.timers.delete(id)
        this.send(id)
      }
      // A write that finishes behind a newer edit schedules that edit, so go again.
      await this.drained()
    } while (this.timers.size > 0)
    const [failure] = this.failures.values()
    if (failure) throw new Error(`Unsaved edits: ${failure}. Copy your edits, then refresh to resolve the conflict.`)
  }

  /**
   * Run a server change to one stem (a rename, say) between this sync's writes: it starts once the
   * writes already sent have landed, and writes made meanwhile wait for it. work gets the stem's
   * expected revision and returns the revision the server now holds, so later writes stay valid.
   */
  exclusive(id: string, work: (revision: number | undefined) => Promise<number | undefined>): Promise<void> {
    const before = this.after
    const pending = [...this.inflight]
    const run = (async () => {
      await before.catch(() => {})
      await Promise.all(pending)
      const revision = await work(this.revisions?.[id])
      if (revision !== undefined && this.revisions) this.revisions[id] = revision
    })()
    this.after = run.catch(() => {})
    return run
  }

  /** Resolves when no write of this sync is in flight. */
  async drained(): Promise<void> {
    while (this.inflight.size > 0) await Promise.all(this.inflight)
  }

  private isSynced(id: string): boolean {
    const track = this.latest.get(id)
    return track !== undefined && bodyOf(track) === this.acked.get(id)
  }

  private schedule(id: string): void {
    clearTimeout(this.timers.get(id))
    this.timers.delete(id)
    if (this.isSynced(id)) return
    if (this.closed) {
      this.send(id)
      return
    }
    this.timers.set(
      id,
      setTimeout(() => {
        this.timers.delete(id)
        this.send(id)
      }, SYNC_DEBOUNCE_MS),
    )
  }

  private send(id: string): void {
    const track = this.latest.get(id)
    if (!track || this.sending.has(id) || this.isSynced(id)) return
    const body = bodyOf(track)
    this.sending.add(id)
    const write: Promise<void> = this.after
      .then(() => sendStemPatch(this.target, id, toSyncPatch(track), this.revisions?.[id]))
      .then(
        (revision) => {
          if (revision !== undefined && this.revisions) this.revisions[id] = revision
          this.acked.set(id, body)
          this.failures.delete(id)
        },
        (err: unknown) => {
          // A timeout can land after the backend applied the write.
          this.acked.delete(id)
          this.failures.set(id, err instanceof Error ? err.message : String(err))
        },
      )
      .then(() => {
        this.inflight.delete(write)
        this.sending.delete(id)
        this.report()
        // Anything newer goes out now. A failed write with nothing newer waits
        // for the next change.
        const newer = bodyOf(this.latest.get(id) ?? track) !== body
        if (newer && !this.timers.has(id)) this.schedule(id)
      })
    this.inflight.add(write)
  }

  private report(): void {
    const [first] = this.failures.values()
    this.onError(first ? `Sync failed: ${first}` : '')
  }
}

/** No target means mock stems, files or no job: no sync, so no network calls. */
export function connectStemSync(
  target: SyncTarget | null | undefined,
  baseline: TrackState[],
  onError: (message: string) => void,
  after?: Promise<unknown>,
): StemSync | null {
  return target ? new StemSync(target, baseline, onError, after) : null
}

/**
 * Follows the current stems. Only the current sync drives onError. A replaced
 * sync's final writes can fail after its stems are gone, which onLateError gets,
 * and its successes say nothing about the stems now loaded.
 */
export class StemSyncRunner {
  private onError: (message: string) => void
  private onLateError: (message: string) => void
  private target: SyncTarget | null = null
  private sync: StemSync | null = null

  constructor(onError: (message: string) => void, onLateError: (message: string) => void) {
    this.onError = onError
    this.onLateError = onLateError
  }

  /**
   * New stems arrive together with their target. They become the baseline of the
   * new sync, and never reach the old one as edits.
   */
  apply(target: SyncTarget | null, tracks: TrackState[]): void {
    if (target === this.target) {
      this.sync?.update(tracks)
      return
    }
    const old = this.sync
    old?.close()
    const after = old?.drained()
    this.target = target
    const sync = connectStemSync(
      target,
      tracks,
      (message) => {
        if (this.sync === sync) this.onError(message)
        else if (message) this.onLateError(`Previous stems: ${message}`)
      },
      after,
    )
    this.sync = sync
  }

  /** Everything edited so far has reached the backend, or failed trying. */
  flush(): Promise<void> {
    return this.sync?.flush() ?? Promise.resolve()
  }

  /** See StemSync.exclusive. With no sync (mock or local stems) the work just runs. */
  exclusive(id: string, work: (revision: number | undefined) => Promise<number | undefined>): Promise<void> {
    return this.sync ? this.sync.exclusive(id, work) : work(undefined).then(() => {})
  }

  close(): void {
    this.sync?.close()
    this.sync = null
    this.target = null
  }
}
