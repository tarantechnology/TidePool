import type { AgentPlan } from '../api/backend'
import { backendDetail, backendStatus, CANCELLED_MESSAGE, isAbort } from '../api/errors'

// Turning what the planner said into what the prompt bar shows. Separate from
// Editor so the wording is testable without a DOM.

export interface PromptOutcome {
  message: string
  ok: boolean
}

/** Shown wherever AI commands are offered but cannot run yet. */
export const NO_PROJECT_HINT = 'Upload a video (step 1) to start giving commands'

/**
 * The outcome for a plan that carries no local edit: a separation the backend
 * already ran, or a question it wants answered. Null means the plan has actions
 * to validate and apply.
 */
export function planOutcome(plan: AgentPlan): PromptOutcome | null {
  if (plan.kind === 'separate') {
    return {
      // Whether the mute worked out is the panel's to report: the separation may
      // have come back without a residual to mute against.
      message: plan.remove ? `Separated ${plan.description}. The Project panel shows what was muted.` : `Separated ${plan.description}.`,
      ok: true,
    }
  }
  // The planner answers `clarify` for a question and for a separation it will
  // not run, such as one that would replace stems the project already has.
  if (plan.kind === 'clarify') return { message: plan.message, ok: false }
  return null
}

/**
 * A planner failure the user can act on, in the server's own words. A 502 means
 * the server has no GPT configured, which is a setup problem and not a bad
 * prompt, so that sentence has to reach the user unchanged. Cancelling is not a
 * failure either. Null for anything else, which the caller reports as the
 * failure it is.
 */
export function plannerFailure(err: unknown): PromptOutcome | null {
  if (isAbort(err)) return { message: CANCELLED_MESSAGE, ok: false }
  if (backendStatus(err) !== 502) return null
  const detail = backendDetail(err)
  return detail ? { message: detail, ok: false } : null
}
