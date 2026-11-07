// The engine is the master clock, because gain automation is scheduled on the
// audio clock. The video is muted and follows it.

export const SYNC_THRESHOLD_SEC = 0.08
export const DURATION_MISMATCH_SEC = 0.25

export type SyncAction = 'none' | 'seek-video'

export function computeSyncAction(
  videoTime: number,
  engineTime: number,
  thresholdSec = SYNC_THRESHOLD_SEC,
): SyncAction {
  return Math.abs(videoTime - engineTime) > thresholdSec ? 'seek-video' : 'none'
}

export function isDurationMismatch(videoDuration: number | null, engineDuration: number): boolean {
  if (videoDuration === null || !Number.isFinite(videoDuration)) return false
  return Math.abs(videoDuration - engineDuration) > DURATION_MISMATCH_SEC
}
