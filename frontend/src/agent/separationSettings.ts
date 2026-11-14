import type { AgentPlan, Anchor, EditScope } from '../api/backend'

/** Explicit command guidance takes precedence; otherwise use the editor controls. */
export function separationSettings(plan: AgentPlan, anchors: Anchor[], action: EditScope['action'], timed: boolean, start: number, end: number): {scope: EditScope; anchors: Anchor[]} {
  const planned = plan.edit_scope || {action: plan.remove ? 'mute' : 'keep'}
  const scope: EditScope = planned.action !== 'keep' ? planned : {
    action,
    ...(timed && action === 'mute' ? {interval: {start, end}} : {}),
  }
  return {scope, anchors: plan.guidance?.anchors?.length ? plan.guidance.anchors : anchors}
}
