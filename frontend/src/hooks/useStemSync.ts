import { useEffect, useRef, useState } from 'react'
import type { SyncTarget } from '../api/backend'
import { StemSyncRunner } from '../api/stemSync'
import type { TrackState } from '../types'

/**
 * Keeps the backend up to date with tracks. Returns the error to show, or ''.
 * onLateError gets failed final writes for stems that have since been replaced.
 */
export function useStemSync(
  target: SyncTarget | null,
  tracks: TrackState[],
  onLateError: (message: string) => void,
): string {
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

  return error
}
