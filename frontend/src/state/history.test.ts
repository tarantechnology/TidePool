import { describe, expect, it } from 'vitest'
import { canRedo, canUndo, commit, COALESCE_WINDOW_MS, createHistory, HISTORY_LIMIT, redo, undo } from './history'

describe('history', () => {
  it('starts with nothing to undo or redo', () => {
    const h = createHistory(0)
    expect(canUndo(h)).toBe(false)
    expect(canRedo(h)).toBe(false)
    expect(undo(h)).toBe(h)
    expect(redo(h)).toBe(h)
  })

  it('undoes and redoes in order', () => {
    let h = commit(commit(createHistory(0), 1), 2)
    h = undo(h)
    expect(h.present).toBe(1)
    h = undo(h)
    expect(h.present).toBe(0)
    expect(canUndo(h)).toBe(false)
    h = redo(h)
    expect(h.present).toBe(1)
    h = redo(h)
    expect(h.present).toBe(2)
    expect(canRedo(h)).toBe(false)
  })

  it('a new commit clears redo', () => {
    const h = commit(undo(commit(commit(createHistory(0), 1), 2)), 5)
    expect(h.present).toBe(5)
    expect(canRedo(h)).toBe(false)
    expect(undo(h).present).toBe(1)
  })

  it('ignores a commit of the same present', () => {
    const h = commit(createHistory('a'), 'b')
    expect(commit(h, 'b')).toBe(h)
  })

  it('caps the past at the limit, dropping the oldest', () => {
    let h = createHistory(0)
    for (let i = 1; i <= HISTORY_LIMIT + 20; i++) h = commit(h, i)
    expect(h.past).toHaveLength(HISTORY_LIMIT)
    expect(h.past[0]).toBe(20)
  })

  it('never mutates the previous history', () => {
    const a = commit(createHistory(0), 1)
    const snapshot = structuredClone(a)
    commit(a, 2)
    undo(a)
    redo(undo(a))
    expect(a).toEqual(snapshot)
  })

  describe('coalescing', () => {
    const key = (now: number, coalesceKey = 'gain:a') => ({ coalesceKey, now })

    it('merges same-key commits inside the window', () => {
      let h = commit(createHistory(0), 1, key(1000))
      h = commit(h, 2, key(1100))
      h = commit(h, 3, key(1200))
      expect(h.past).toEqual([0])
      expect(undo(h).present).toBe(0)
    })

    it('treats the window boundary as inside', () => {
      const a = commit(createHistory(0), 1, key(1000))
      expect(commit(a, 2, key(1000 + COALESCE_WINDOW_MS)).past).toEqual([0])
      expect(commit(a, 2, key(1000 + COALESCE_WINDOW_MS + 1)).past).toEqual([0, 1])
    })

    it('slides the window so a long drag stays one step', () => {
      let h = commit(createHistory(0), 1, key(0))
      for (let i = 1; i <= 50; i++) h = commit(h, i + 1, key(i * 100))
      expect(h.past).toEqual([0])
    })

    it('does not merge different keys', () => {
      const h = commit(commit(createHistory(0), 1, key(1000, 'gain:a')), 2, key(1050, 'gain:b'))
      expect(h.past).toEqual([0, 1])
    })

    it('does not merge an unkeyed commit', () => {
      const h = commit(commit(createHistory(0), 1, key(1000)), 2, { now: 1050 })
      expect(h.past).toEqual([0, 1])
    })

    it('an unkeyed commit breaks the run', () => {
      let h = commit(createHistory(0), 1, key(1000))
      h = commit(h, 2, { now: 1050 })
      h = commit(h, 3, key(1100))
      expect(h.past).toEqual([0, 1, 2])
    })

    it('does not merge across undo', () => {
      let h = commit(commit(createHistory(0), 1), 2, key(1000))
      h = undo(h)
      h = commit(h, 9, key(1050))
      expect(h.past).toEqual([0, 1])
      expect(undo(h).present).toBe(1)
    })

    it('does not merge across redo', () => {
      let h = commit(createHistory(0), 1, key(1000))
      h = redo(undo(h))
      h = commit(h, 2, key(1050))
      expect(h.past).toEqual([0, 1])
    })

    it('clears redo even when merging', () => {
      let h = commit(commit(createHistory(0), 1), 2)
      h = undo(h)
      h = commit(h, 3, key(1000))
      h = commit(h, 4, key(1100))
      expect(canRedo(h)).toBe(false)
    })
  })
})
