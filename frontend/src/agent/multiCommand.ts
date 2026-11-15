import type { AgentPlan, MultiStep } from '../api/backend'
import type { FxStem } from '../fxCommand'
import { chatRoute } from './chatRoute'

/**
 * Several edits in one prompt ("add reverb to the vocals, make the drums sound like a phone and put a vine boom
 * at 0:12"). The AI planner answers these with kind 'multi'; this is the clause splitter (kept in step with
 * app/multi.py split_clauses) and the rule-based fallback for when the planner is unavailable.
 */
export const MAX_STEPS = 8
export const MAX_GENERATE = 3

const VERBS = 'add|put|insert|drop|throw|stick|slap|play|place|make|turn|pitch|shift|transpose|give|apply|boost|cut|lower|raise|' +
  'mute|unmute|solo|duck|normali[sz]e|bring|set|fade|remove|take|reduce|increase|pan|widen|mix|layer|change|run|' +
  'compress|distort|send|drown|bump|crank|quiet|silence|speed|slow|bleep|censor|isolate|separate|split|replace|swap'
const LEAD = '(?:(?:please|can you|could you|would you|also|then|just|now|and)\\s+)*'
const SPLIT = new RegExp(
  `\\s*(?:;|,|\\.(?=\\s))\\s*(?:and\\s+then\\s+|and\\s+also\\s+|and\\s+|then\\s+|also\\s+|plus\\s+)?(?=${LEAD}(?:${VERBS})\\b)` +
  `|\\s+(?:and\\s+then|and\\s+also|and|then|also|plus|after\\s+that)\\s+(?=${LEAD}(?:${VERBS})\\b)`, 'i')
const QUOTED = /"[^"]*"|“[^”]*”|‘[^’]*’|(?<![\w])'[^']*'(?![\w])/g
/** Asks that run as their own flow and cannot be one step of a combined preview. */
const OWN_FLOW = /\b(?:bleep|censor|curse|swear|cuss|replace|swap|substitute|isolate|separate|split|karaoke|instrumental|cinematic|nightcore|slowed|sped[- ]up|speed (?:it |this )?up|slow (?:it|this) down)\b/i

/** Clauses that each start a new edit verb; quoted words never split. */
export function splitClauses(text: string): string[] {
  const quotes: string[] = []
  const masked = text.trim().replace(QUOTED, (m) => { quotes.push(m); return `\u0000${quotes.length - 1}\u0000` })
  return masked.split(SPLIT)
    .map((p) => p.replace(/^[\s,;.]+|[\s,;.]+$/g, ''))
    .filter(Boolean)
    .map((p) => p.replace(/\u0000(\d+)\u0000/g, (_, i: string) => quotes[Number(i)]))
}

export function isCompound(text: string): boolean {
  return splitClauses(text).length >= 2
}

function clarify(message: string): AgentPlan {
  return { kind: 'clarify', description: null, remove: false, actions: [], message }
}

const STEP_KINDS = new Set(['fx', 'duck', 'loudness', 'insert', 'generate', 'edit'])

/**
 * Rule-based multi plan: each clause through the local precedence (chatRoute), effects parsed locally, inserts and
 * new parts from the server's offline parsers via `planClause` (one clause at a time, so they need no AI). Any clause
 * that cannot be planned, or that has its own flow (bleeping, replacement, separation, speed), becomes a question.
 */
export async function localMultiPlan(text: string, stems: FxStem[], planClause: (clause: string) => Promise<AgentPlan | null>): Promise<AgentPlan> {
  const clauses = splitClauses(text)
  if (clauses.length > MAX_STEPS) return clarify(`That is a lot of edits at once. I can do up to ${MAX_STEPS} per message; please split the request.`)
  const steps: MultiStep[] = []
  for (const [i, clause] of clauses.entries()) {
    const name = `Edit ${i + 1} ("${clause}")`
    if (OWN_FLOW.test(clause)) return clarify(`${name} has to be done on its own. Ask for it separately, then the rest.`)
    const route = chatRoute(clause, stems)
    if (route.replacement && !route.sfx) return clarify(`${name} replaces a sound; ask for it on its own.`)
    const base = { label: clause.slice(0, 60), fx: null, inserts: null, generate: null, actions: [] as unknown[] }
    if (route.fx && !route.sfx && !route.generate) {
      if (route.fx.kind === 'clarify') return clarify(`${name}: ${route.fx.message}`)
      steps.push({ ...base, kind: route.fx.kind, fx: route.fx.fx })
      continue
    }
    const plan = await planClause(clause)
    if (!plan || plan.kind === 'clarify') return clarify(`${name}: ${plan?.message || 'I could not plan that part. Try it on its own.'}`)
    if (!STEP_KINDS.has(plan.kind)) return clarify(`${name} has to be done on its own. Ask for it separately, then the rest.`)
    if (plan.kind === 'insert' && plan.inserts?.length) steps.push({ ...base, kind: 'insert', inserts: plan.inserts })
    else if (plan.kind === 'generate' && plan.generate) steps.push({ ...base, kind: 'generate', generate: plan.generate })
    else if ((plan.kind === 'fx' || plan.kind === 'duck' || plan.kind === 'loudness') && plan.fx) steps.push({ ...base, kind: plan.kind, fx: plan.fx })
    else if (plan.kind === 'edit' && Array.isArray(plan.actions) && plan.actions.length) steps.push({ ...base, kind: 'edit', actions: plan.actions as unknown[] })
    else return clarify(`${name}: I could not plan that part. Try it on its own.`)
  }
  if (steps.filter((s) => s.kind === 'generate').length > MAX_GENERATE) return clarify(`I can add up to ${MAX_GENERATE} new parts per message; please split the request.`)
  return { kind: 'multi', description: null, remove: false, actions: [], steps, message: `Previewing ${steps.length} edits: ${steps.map((s) => s.label).join('; ')}.` }
}
