import { describe, expect, it } from 'vitest'
import { MAX_GAIN } from './edits'
import { pointToKeyframe, addKeyframe, MIN_KEYFRAME_SPACING, moveKeyframe, removeKeyframe } from './keyframes'

const KFS = [
  { t: 2, value: 1 },
  { t: 4, value: 0.5 },
  { t: 6, value: 1.5 },
]
const DUR = 10

describe('addKeyframe', () => {
  it('inserts in sorted position', () => {
    const next = addKeyframe(KFS, { t: 3, value: 0.2 }, DUR)
    expect(next.map((k) => k.t)).toEqual([2, 3, 4, 6])
  })

  it('clamps t to [0, duration]', () => {
    expect(addKeyframe([], { t: -5, value: 1 }, DUR)[0].t).toBe(0)
    expect(addKeyframe([], { t: 50, value: 1 }, DUR)[0].t).toBe(DUR)
  })

  it('clamps value to [0, MAX_GAIN]', () => {
    expect(addKeyframe([], { t: 1, value: -1 }, DUR)[0].value).toBe(0)
    expect(addKeyframe([], { t: 1, value: 9 }, DUR)[0].value).toBe(MAX_GAIN)
  })

  it('updates the existing keyframe rather than break spacing', () => {
    const next = addKeyframe(KFS, { t: 4.005, value: 0.1 }, DUR)
    expect(next).toHaveLength(3)
    expect(next[1]).toEqual({ t: 4, value: 0.1 })
  })

  it('inserts when exactly at the minimum spacing', () => {
    expect(addKeyframe(KFS, { t: 4 + MIN_KEYFRAME_SPACING, value: 0.1 }, DUR)).toHaveLength(4)
  })

  it('does not mutate its input', () => {
    const snapshot = structuredClone(KFS)
    addKeyframe(KFS, { t: 3, value: 0.2 }, DUR)
    addKeyframe(KFS, { t: 4, value: 0.2 }, DUR)
    expect(KFS).toEqual(snapshot)
  })
})

describe('moveKeyframe', () => {
  it('moves freely between neighbours', () => {
    expect(moveKeyframe(KFS, 1, { t: 5, value: 1 }, DUR)[1]).toEqual({ t: 5, value: 1 })
  })

  it('stops short of the left neighbour', () => {
    expect(moveKeyframe(KFS, 1, { t: 0, value: 1 }, DUR)[1].t).toBeCloseTo(2 + MIN_KEYFRAME_SPACING)
  })

  it('stops short of the right neighbour', () => {
    expect(moveKeyframe(KFS, 1, { t: 9, value: 1 }, DUR)[1].t).toBeCloseTo(6 - MIN_KEYFRAME_SPACING)
  })

  it('clamps the ends to [0, duration]', () => {
    expect(moveKeyframe(KFS, 0, { t: -3, value: 1 }, DUR)[0].t).toBe(0)
    expect(moveKeyframe(KFS, 2, { t: 99, value: 1 }, DUR)[2].t).toBe(DUR)
  })

  it('clamps value to [0, MAX_GAIN]', () => {
    expect(moveKeyframe(KFS, 1, { t: 4, value: 9 }, DUR)[1].value).toBe(MAX_GAIN)
    expect(moveKeyframe(KFS, 1, { t: 4, value: -9 }, DUR)[1].value).toBe(0)
  })

  it('keeps the time when neighbours leave no room', () => {
    const tight = [
      { t: 1, value: 1 },
      { t: 1.005, value: 1 },
      { t: 1.01, value: 1 },
    ]
    expect(moveKeyframe(tight, 1, { t: 5, value: 0 }, DUR)[1]).toEqual({ t: 1.005, value: 0 })
  })

  it('ignores an out of range index', () => {
    expect(moveKeyframe(KFS, 7, { t: 1, value: 1 }, DUR)).toBe(KFS)
  })

  it('does not mutate its input', () => {
    const snapshot = structuredClone(KFS)
    moveKeyframe(KFS, 1, { t: 5, value: 1 }, DUR)
    expect(KFS).toEqual(snapshot)
  })
})

describe('removeKeyframe', () => {
  it('removes by index', () => {
    expect(removeKeyframe(KFS, 1).map((k) => k.t)).toEqual([2, 6])
  })

  it('can empty the lane', () => {
    expect(removeKeyframe([{ t: 1, value: 1 }], 0)).toEqual([])
  })

  it('ignores an out of range index', () => {
    expect(removeKeyframe(KFS, -1)).toBe(KFS)
  })
})

describe('pointToKeyframe', () => {
  const rect = { left: 100, top: 50, width: 200, height: 100 }

  it('maps the lane box to time and gain', () => {
    expect(pointToKeyframe(200, 100, rect, 20)).toEqual({ t: 10, value: 1 })
    expect(pointToKeyframe(100, 50, rect, 20)).toEqual({ t: 0, value: 2 })
    expect(pointToKeyframe(300, 150, rect, 20)).toEqual({ t: 20, value: 0 })
  })

  it('clamps pointers dragged outside the lane', () => {
    expect(pointToKeyframe(0, 0, rect, 20)).toEqual({ t: 0, value: 2 })
    expect(pointToKeyframe(999, 999, rect, 20)).toEqual({ t: 20, value: 0 })
  })
})

it('does not add a 65th keyframe, but permits updating an existing point', () => {
  const points = Array.from({ length: 64 }, (_, i) => ({ t: i, value: 1 }))
  expect(addKeyframe(points, { t: 65, value: .5 }, 100)).toBe(points)
  expect(addKeyframe(points, { t: 10, value: .5 }, 100)[10].value).toBe(.5)
})
