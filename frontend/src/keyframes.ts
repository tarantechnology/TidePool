import { clamp, MAX_GAIN } from './edits'
import type { Keyframe } from './types'

// Pure keyframe edits. Every function returns a new array and keeps it sorted.

/** Smallest allowed gap between two keyframes, in seconds. */
export const MIN_KEYFRAME_SPACING = 0.01

/**
 * Insert a keyframe. A point landing within MIN_KEYFRAME_SPACING of an existing
 * keyframe updates that keyframe instead of crowding it.
 */
export function addKeyframe(keyframes: Keyframe[], kf: Keyframe, duration: number): Keyframe[] {
  const next = { t: clamp(kf.t, 0, duration), value: clamp(kf.value, 0, MAX_GAIN) }
  const hit = keyframes.findIndex((k) => Math.abs(k.t - next.t) < MIN_KEYFRAME_SPACING - 1e-9)
  if (hit >= 0) return keyframes.map((k, i) => (i === hit ? { ...k, value: next.value } : k))
  return [...keyframes, next].sort((a, b) => a.t - b.t)
}

/**
 * Move the keyframe at index. It cannot cross its neighbours: t stops
 * MIN_KEYFRAME_SPACING short of them, so the index stays valid across a drag.
 */
export function moveKeyframe(keyframes: Keyframe[], index: number, kf: Keyframe, duration: number): Keyframe[] {
  if (index < 0 || index >= keyframes.length) return keyframes
  const lo = index > 0 ? keyframes[index - 1].t + MIN_KEYFRAME_SPACING : 0
  const hi = index < keyframes.length - 1 ? keyframes[index + 1].t - MIN_KEYFRAME_SPACING : duration
  // Neighbours already closer than the spacing leave no room: keep the old time.
  const t = lo <= hi ? clamp(kf.t, lo, hi) : keyframes[index].t
  const value = clamp(kf.value, 0, MAX_GAIN)
  return keyframes.map((k, i) => (i === index ? { t, value } : k))
}

export function removeKeyframe(keyframes: Keyframe[], index: number): Keyframe[] {
  if (index < 0 || index >= keyframes.length) return keyframes
  return keyframes.filter((_, i) => i !== index)
}

/** Map a pointer position inside a lane to timeline coordinates (top of lane = MAX_GAIN). */
export function pointToKeyframe(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  duration: number,
): Keyframe {
  const x = clamp((clientX - rect.left) / rect.width, 0, 1)
  const y = clamp((clientY - rect.top) / rect.height, 0, 1)
  return { t: x * duration, value: (1 - y) * MAX_GAIN }
}
