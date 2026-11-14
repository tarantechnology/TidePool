import { describe, expect, it } from 'vitest'
import type { AgentPlan } from '../api/backend'
import { separationSettings } from './separationSettings'

const plan: AgentPlan = {kind:'separate', description:'dog barking', remove:false, actions:[], message:''}
const anchors = [{token:'+' as const, start:4, end:5}]

describe('separation settings for the first preview', () => {
  it('uses selected guidance and timed edit when the command supplies neither', () => {
    expect(separationSettings(plan, anchors, 'mute', true, 6, 7)).toEqual({anchors, scope:{action:'mute', interval:{start:6,end:7}}})
  })
  it('prefers time guidance and an explicit timed edit from the command', () => {
    const supplied = [{token:'-' as const, start:1, end:2}]
    expect(separationSettings({...plan, guidance:{anchors:supplied}, edit_scope:{action:'mute',interval:{start:8,end:9}}}, anchors, 'lower', false, 0, 1)).toEqual({anchors:supplied, scope:{action:'mute',interval:{start:8,end:9}}})
  })
  it('uses panel anchors when the planner returns an empty list', () => {
    expect(separationSettings({...plan,guidance:{anchors:[]}}, anchors, 'keep', false, 0, 1).anchors).toEqual(anchors)
  })
})
