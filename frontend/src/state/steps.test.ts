import { describe, expect, it } from 'vitest'
import { EMPTY_STEPS, STEP_LIMIT, dropRedo, nextRedo, nextUndo, parseSteps, pushStep, stepRedone, stepUndone } from './steps'

const step = (n: number) => ({ label: `step ${n}`, before: `b${n}`, after: `a${n}` })

describe('server steps', () => {
  it('undoes and redoes in order', () => {
    let s = pushStep(pushStep(EMPTY_STEPS, step(1)), step(2))
    expect(nextUndo(s)?.label).toBe('step 2')
    s = stepUndone(s)
    expect(nextUndo(s)?.label).toBe('step 1')
    expect(nextRedo(s)?.label).toBe('step 2')
    s = stepUndone(s)
    expect(nextUndo(s)).toBeNull()
    s = stepRedone(s)
    expect(nextUndo(s)?.before).toBe('b1')
    expect(nextRedo(s)?.after).toBe('a2')
  })
  it('a new step or a local edit clears redo', () => {
    const undone = stepUndone(pushStep(EMPTY_STEPS, step(1)))
    expect(pushStep(undone, step(2)).future).toEqual([])
    expect(dropRedo(undone).future).toEqual([])
    expect(dropRedo(EMPTY_STEPS)).toBe(EMPTY_STEPS)
  })
  it('keeps a bounded history and ignores junk from storage', () => {
    let s = EMPTY_STEPS
    for (let i = 0; i < STEP_LIMIT + 5; i++) s = pushStep(s, step(i))
    expect(s.past).toHaveLength(STEP_LIMIT)
    expect(parseSteps({ past: [step(1), { label: 1 }], future: 'x' })).toEqual(EMPTY_STEPS)
    expect(parseSteps({ past: [step(1), { label: 1 }], future: [] }).past).toEqual([step(1)])
    expect(parseSteps(null)).toEqual(EMPTY_STEPS)
  })
})
