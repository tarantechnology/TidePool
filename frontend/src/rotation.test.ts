import { describe, expect, it } from 'vitest'
import { displayAspect, fitRotated, readRotation, rotateBox, turn, writeRotation } from './rotation'

describe('rotation', () => {
  it('turns in quarter steps both ways', () => {
    expect(turn(0)).toBe(90)
    expect(turn(270)).toBe(0)
    expect(turn(0, -1)).toBe(270)
    expect(turn(90, -1)).toBe(0)
  })

  it('flips the aspect when sideways', () => {
    expect(displayAspect(16 / 9, 0)).toBeCloseTo(16 / 9)
    expect(displayAspect(16 / 9, 90)).toBeCloseTo(9 / 16)
    expect(displayAspect(16 / 9, 180)).toBeCloseTo(16 / 9)
  })

  it('moves boxes with the picture', () => {
    // A box in the top-left corner of the clip.
    const box = [0, 0, 0.25, 0.5]
    expect(rotateBox(box, 0)).toEqual([0, 0, 0.25, 0.5])
    // Clockwise: top-left goes to top-right, and the box becomes wide.
    expect(rotateBox(box, 90)).toEqual([0.5, 0, 1, 0.25])
    expect(rotateBox(box, 180)).toEqual([0.75, 0.5, 1, 1])
    expect(rotateBox(box, 270)).toEqual([0, 0.75, 0.5, 1])
    // Four quarter turns bring it back.
    let b = rotateBox(box, 90)
    for (let i = 0; i < 3; i++) b = rotateBox(b, 90)
    expect(b).toEqual(box)
  })

  it('fits a turned landscape clip as portrait', () => {
    const fit = fitRotated(400, 400, 1920, 1080, 90)
    expect(fit.height).toBeCloseTo(400)
    expect(fit.width).toBeCloseTo(400 * 1080 / 1920)
    // The unturned element keeps its own shape and shares the centre.
    expect(fit.element.width).toBeCloseTo(400)
    expect(fit.element.height).toBeCloseTo(fit.width)
    expect(fit.element.left + fit.element.width / 2).toBeCloseTo(200)
    expect(fit.element.top + fit.element.height / 2).toBeCloseTo(200)
    expect(fit.element.transform).toBe('rotate(90deg)')
  })

  it('remembers a turn per project', () => {
    expect(readRotation('p1')).toBeNull()
    writeRotation('p1', 270)
    expect(readRotation('p1')).toBe(270)
    expect(readRotation(null)).toBeNull()
  })
})
