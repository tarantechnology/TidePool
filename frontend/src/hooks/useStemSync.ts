import { useCallback, useEffect, useRef, useState } from 'react'
import type { SyncTarget } from '../api/backend'
import { StemSyncRunner } from '../api/stemSync'
import type { TrackState } from '../types'

export type StemExclusive = (id: string, work: (revision: number | undefined) => Promise<number | undefined>) => Promise<void>

/**
 * Keeps the backend up to date with tracks. Returns the error to show, or '',
 * and flush, which resolves once queued edits have landed.
 * onLateError gets failed final writes for stems that have since been replaced.
 */
export function useStemSync(
  target: SyncTarget | null,
  tracks: TrackState[],
  onLateError: (message: string) => void,
): { syncError: string; flush: () => Promise<void>; exclusive: StemExclusive } {
  const [error, setError] = useState('')
  const runner = useRef<StemSyncRunner | null>(null)

  useEffect(() => {
    runner.current ??= new StemSyncRunner(setError, onLateError)
    runner.current.apply(target, tracks)
  }, [target, tracks, onLateError])

  useEffect(
    () => () => {
      runner.current?.close()
      runner.current = null
    },
    [],
  )

  const flush = useCallback(() => runner.current?.flush() ?? Promise.resolve(), [])

  const exclusive = useCallback<StemExclusive>((id, work) => runner.current ? runner.current.exclusive(id, work) : work(undefined).then(() => {}), [])

  return { syncError: error, flush, exclusive }
}
