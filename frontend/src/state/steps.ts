// Undo and redo for changes made on the server (applying a preview, Explode, a Current command,
// Moon pull...). Each step remembers a checkpoint from just before it and one from just after.
// The editor's own history covers fader and curve edits; this stack takes over once that runs out.

export interface ServerStep {
  label: string
  /** Checkpoint to restore to undo the step. */
  before: string
  /** Checkpoint to restore to redo it. */
  after: string
}

export interface StepStack {
  past: ServerStep[]
  future: ServerStep[]
}

export const STEP_LIMIT = 30
export const EMPTY_STEPS: StepStack = { past: [], future: [] }

/** A new step clears the redo side, as any new edit does. */
export function pushStep(s: StepStack, step: ServerStep): StepStack {
  return { past: [...s.past, step].slice(-STEP_LIMIT), future: [] }
}

export function nextUndo(s: StepStack): ServerStep | null {
  return s.past[s.past.length - 1] ?? null
}

export function nextRedo(s: StepStack): ServerStep | null {
  return s.future[0] ?? null
}

/** After the step's "before" checkpoint was restored. */
export function stepUndone(s: StepStack): StepStack {
  const step = nextUndo(s)
  return step ? { past: s.past.slice(0, -1), future: [step, ...s.future] } : s
}

/** After the step's "after" checkpoint was restored. */
export function stepRedone(s: StepStack): StepStack {
  const step = nextRedo(s)
  return step ? { past: [...s.past, step], future: s.future.slice(1) } : s
}

/** A local edit made after undoing a server step means that step can no longer be redone. */
export function dropRedo(s: StepStack): StepStack {
  return s.future.length ? { ...s, future: [] } : s
}

function isStep(v: unknown): v is ServerStep {
  const s = v as Partial<ServerStep> | null
  return typeof s?.label === 'string' && typeof s.before === 'string' && typeof s.after === 'string'
}

export function parseSteps(raw: unknown): StepStack {
  const v = raw as Partial<StepStack> | null
  if (!v || !Array.isArray(v.past) || !Array.isArray(v.future)) return EMPTY_STEPS
  return { past: v.past.filter(isStep).slice(-STEP_LIMIT), future: v.future.filter(isStep).slice(0, STEP_LIMIT) }
}
