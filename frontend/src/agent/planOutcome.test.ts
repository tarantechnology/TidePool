import { describe, expect, it } from 'vitest'
import type { AgentPlan, SavedStem } from '../api/backend'
import { muteExistingStem, NO_PROJECT_HINT, planOutcome, plannerFailure } from './planOutcome'

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
    expect(NO_PROJECT_HINT).toBe('Upload a video or song (step 1) to start giving commands')
  })
})

describe('muteExistingStem', () => {
  const saved = (id: string, name: string, role: string, label?: string): SavedStem => ({
    id, name, role, ...(label ? { label } : {}), duration: 4, audio_url: `/a/${id}`, revision: 0, volume: 1, muted: false, solo: false, automation: [], start_offset: 0, mute_ranges: [],
  })
  const split = [saved('v', 'Vocals', 'target', 'vocals'), saved('d', 'Drums', 'target', 'drums'), saved('b', 'Bass', 'target', 'bass'), saved('o', 'Other', 'target', 'other')]

  it('turns "remove vocals" on a split song into a mute of the Vocals track', () => {
    const edit = muteExistingStem(plan({ kind: 'separate', description: 'vocals', remove: true }), split)
    expect(edit?.kind).toBe('edit')
    expect(edit?.actions).toEqual([{ type: 'set_mute', stemId: 'v', muted: true }])
    // A karaoke plan names the stem rather than describing it.
    expect(muteExistingStem(plan({ kind: 'separate', description: 'karaoke', remove: true, separation_mode: 'music_stems', stems: ['vocals'] }), split)?.actions)
      .toEqual([{ type: 'set_mute', stemId: 'v', muted: true }])
  })

  it('leaves isolating, timed removals and unknown sounds to a separation', () => {
    expect(muteExistingStem(plan({ kind: 'separate', description: 'vocals', remove: false }), split)).toBeNull()
    expect(muteExistingStem(plan({ kind: 'separate', description: 'vocals', remove: true, edit_scope: { action: 'mute', interval: { start: 1, end: 2 } } }), split)).toBeNull()
    expect(muteExistingStem(plan({ kind: 'separate', description: 'vocals', remove: true, processing_time: 4 }), split)).toBeNull()
    expect(muteExistingStem(plan({ kind: 'separate', description: 'siren', remove: true }), split)).toBeNull()
  })

  it('never mutes a lone target that only stands in for another sound', () => {
    const prompted = [saved('t', 'dog', 'target', 'dog'), saved('r', 'residual', 'residual')]
    expect(muteExistingStem(plan({ kind: 'separate', description: 'siren', remove: true }), prompted)).toBeNull()
    expect(muteExistingStem(plan({ kind: 'separate', description: 'dog', remove: true }), prompted)?.actions).toEqual([{ type: 'set_mute', stemId: 't', muted: true }])
  })
})
