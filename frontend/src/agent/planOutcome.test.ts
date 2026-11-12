import { describe, expect, it } from 'vitest'
import type { AgentPlan } from '../api/backend'
import { NO_PROJECT_HINT, planOutcome, plannerFailure } from './planOutcome'

const plan = (over: Partial<AgentPlan>): AgentPlan => ({
  kind: 'edit',
  description: null,
  remove: false,
  actions: [],
  message: '',
  ...over,
})

describe('planOutcome', () => {
  it('reports a separation the backend already ran', () => {
    expect(planOutcome(plan({ kind: 'separate', description: 'siren', remove: false }))).toEqual({
      message: 'Separated siren.',
      ok: true,
    })
    // The mute is the panel's to report: a separation can come back with no
    // residual to mute against, and this must not claim otherwise.
    expect(planOutcome(plan({ kind: 'separate', description: 'siren', remove: true }))).toEqual({
      message: 'Separated siren. The Project panel shows what was muted.',
      ok: true,
    })
  })

  it('shows what a clarify plan asked, including a refused separation', () => {
    expect(planOutcome(plan({ kind: 'clarify', message: 'This project already has stems. Which one did you mean?' }))).toEqual({
      message: 'This project already has stems. Which one did you mean?',
      ok: false,
    })
  })

  it('leaves an edit plan to the caller to validate and apply', () => {
    expect(planOutcome(plan({ kind: 'edit', actions: [{ type: 'set_gain' }], message: 'Lowered the siren.' }))).toBeNull()
  })
})

describe('plannerFailure', () => {
  it('passes a 502 configuration message through word for word', () => {
    const err = new Error('Backend returned 502: GPT is not configured on the server.')
    expect(plannerFailure(err)).toEqual({ message: 'GPT is not configured on the server.', ok: false })
  })

  it('reports a cancelled run as cancelled', () => {
    const abort = new DOMException('The operation was aborted.', 'AbortError')
    expect(plannerFailure(abort)?.ok).toBe(false)
    expect(plannerFailure(abort)?.message).toMatch(/cancel/i)
  })

  it('leaves anything else to be reported as the failure it is', () => {
    expect(plannerFailure(new Error('Backend returned 500: planner crashed'))).toBeNull()
    expect(plannerFailure(new Error('Backend returned 502'))).toBeNull()
    expect(plannerFailure(new Error('Failed to fetch'))).toBeNull()
  })
})

describe('NO_PROJECT_HINT', () => {
  it('tells the user where to upload a video', () => {
    expect(NO_PROJECT_HINT).toBe('Upload a video in Project to use AI commands')
  })
})
