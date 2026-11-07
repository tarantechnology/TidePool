// Generic undo/redo. Pure, so the clock is passed in and tests stay deterministic.

export const HISTORY_LIMIT = 100
export const COALESCE_WINDOW_MS = 400

export interface History<T> {
  past: T[]
  present: T
  future: T[]
  /** Key and time of the last commit, for coalescing. Cleared by undo and redo. */
  lastKey?: string
  lastAt?: number
}

export interface CommitOptions {
  coalesceKey?: string
  now?: number
}

export function createHistory<T>(present: T): History<T> {
  return { past: [], present, future: [] }
}

export function canUndo<T>(h: History<T>): boolean {
  return h.past.length > 0
}

export function canRedo<T>(h: History<T>): boolean {
  return h.future.length > 0
}

/**
 * Record a new present. Commits sharing a coalesceKey less than or exactly
 * COALESCE_WINDOW_MS after the previous one merge into one undo step. The window
 * slides, so a continuous drag stays one step however long it lasts.
 */
export function commit<T>(h: History<T>, present: T, opts: CommitOptions = {}): History<T> {
  if (present === h.present) return h
  const { coalesceKey, now } = opts
  const merge =
    coalesceKey !== undefined &&
    coalesceKey === h.lastKey &&
    now !== undefined &&
    h.lastAt !== undefined &&
    now - h.lastAt <= COALESCE_WINDOW_MS
  return {
    past: merge ? h.past : [...h.past, h.present].slice(-HISTORY_LIMIT),
    present,
    future: [],
    lastKey: coalesceKey,
    lastAt: now,
  }
}

export function undo<T>(h: History<T>): History<T> {
  if (!canUndo(h)) return h
  return {
    past: h.past.slice(0, -1),
    present: h.past[h.past.length - 1],
    future: [h.present, ...h.future],
  }
}

export function redo<T>(h: History<T>): History<T> {
  if (!canRedo(h)) return h
  return {
    past: [...h.past, h.present],
    present: h.future[0],
    future: h.future.slice(1),
  }
}
