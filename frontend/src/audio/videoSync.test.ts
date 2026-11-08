import { describe, expect, it } from 'vitest'
import { computeSyncAction, isDurationMismatch } from './videoSync'

describe('computeSyncAction', () => {
  it('does nothing within the threshold', () => {
    expect(computeSyncAction(10, 10)).toBe('none')
    expect(computeSyncAction(10.05, 10)).toBe('none')
    expect(computeSyncAction(9.95, 10)).toBe('none')
  })

  it('treats exactly the threshold as in sync', () => {
    expect(computeSyncAction(11, 10, 1)).toBe('none')
  })

  it('seeks when the video is ahead', () => {
    expect(computeSyncAction(10.2, 10)).toBe('seek-video')
  })

  it('seeks when the video is behind (negative drift)', () => {
    expect(computeSyncAction(9.8, 10)).toBe('seek-video')
  })

  it('honours a custom threshold', () => {
    expect(computeSyncAction(10.2, 10, 0.5)).toBe('none')
  })
})

describe('isDurationMismatch', () => {
  it('ignores a missing video and small differences', () => {
    expect(isDurationMismatch(null, 20)).toBe(false)
    expect(isDurationMismatch(20.2, 20)).toBe(false)
    expect(isDurationMismatch(19.8, 20)).toBe(false)
  })

  it('ignores a video that reports no finite duration', () => {
    expect(isDurationMismatch(Infinity, 20)).toBe(false)
    expect(isDurationMismatch(NaN, 20)).toBe(false)
  })

  it('flags differences above 0.25s either way', () => {
    expect(isDurationMismatch(20.5, 20)).toBe(true)
    expect(isDurationMismatch(19.5, 20)).toBe(true)
  })
})
