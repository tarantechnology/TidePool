import { parsePrompt } from '../edits'
import type { EditAction, Stem, TrackState } from '../types'
import { validateActions } from './validate'

export interface AgentRequest {
  prompt: string
  stems: Stem[]
  tracks: TrackState[]
  duration: number
}

export interface AgentResult {
  actions: EditAction[]
  message: string
}

export interface Agent {
  /** Short description shown under the prompt bar. */
  readonly label: string
  run(req: AgentRequest): Promise<AgentResult>
}

export const AGENT_TIMEOUT_MS = 15_000

/** The regex parser behind the Agent interface. */
export class RegexAgent implements Agent {
  readonly label = 'rule-based parser, LLM agent not connected'

  run({ prompt, stems, tracks }: AgentRequest): Promise<AgentResult> {
    return Promise.resolve(parsePrompt(prompt, stems, tracks))
  }
}

/**
 * Calls the agent server (server/agent-server.mjs). The model's output is
 * validated here, and any failure falls back to the regex parser so the prompt
 * bar keeps working. The reason for a fallback is always in the message.
 */
export class HttpAgent implements Agent {
  readonly label = 'LLM agent'
  private url: string
  private fallback: Agent

  constructor(url: string, fallback: Agent = new RegexAgent()) {
    this.url = url
    this.fallback = fallback
  }

  async run(req: AgentRequest): Promise<AgentResult> {
    try {
      return await this.request(req)
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err)
      const res = await this.fallback.run(req)
      return { ...res, message: `Agent unavailable (${reason}), used the rule-based parser. ${res.message}` }
    }
  }

  private async request({ prompt, stems, tracks, duration }: AgentRequest): Promise<AgentResult> {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), AGENT_TIMEOUT_MS)
    try {
      const res = await fetch(this.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Stem URLs are local blobs or token-protected, so they stay out of the request.
        body: JSON.stringify({
          prompt,
          stems: stems.map(({ id, label, kind, duration: d }) => ({ id, label, kind, duration: d })),
          tracks,
          duration,
        }),
        signal: ctrl.signal,
      })
      if (!res.ok) {
        const failure = (await res.json().catch(() => null)) as { error?: unknown } | null
        const reason = typeof failure?.error === 'string' ? `: ${failure.error}` : ''
        throw new Error(`HTTP ${res.status}${reason}`)
      }
      const body = (await res.json()) as { actions?: unknown; message?: unknown } | null
      const checked = validateActions(body?.actions, stems, duration)
      if (!checked.ok) throw new Error(`invalid output: ${checked.errors[0]}`)
      const fallback = checked.actions.length > 0 ? 'Agent applied the edit.' : 'The agent made no changes.'
      const message = typeof body?.message === 'string' && body.message ? body.message : fallback
      return { actions: checked.actions, message }
    } catch (err) {
      throw ctrl.signal.aborted ? new Error('timed out') : err
    } finally {
      clearTimeout(timer)
    }
  }
}

export function createAgent(url: string | undefined): Agent {
  return url ? new HttpAgent(url) : new RegexAgent()
}
