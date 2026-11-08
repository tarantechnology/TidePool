// Prompt-to-edit agent server. Keeps the Anthropic API key out of the browser.
// Run with: node server/agent-server.mjs (needs ANTHROPIC_API_KEY and AGENT_MODEL).
// The browser validates everything returned here (src/agent/validate.ts).
import { createServer } from 'node:http'
import { pathToFileURL } from 'node:url'

const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'
const ANTHROPIC_VERSION = '2026-10-21'
const ALLOWED_ORIGIN = 'http://localhost:5173'
const DEFAULT_PORT = 8787
const MAX_BODY_BYTES = 1_000_000
const MAX_TOKENS = 16000
const UPSTREAM_TIMEOUT_MS = 15_000
const TOOL_NAME = 'apply_edits'

/** An error whose text is safe to show. Anything else can quote the API key. */
class SafeError extends Error {}

const stemId = { type: 'string', description: 'id of a stem from the list in the system prompt' }
const gain = { type: 'number', minimum: 0, maximum: 2 }

const actionVariant = (type, properties) => ({
  type: 'object',
  properties: { type: { const: type }, stemId, ...properties },
  required: ['type', 'stemId', ...Object.keys(properties).filter((k) => !k.endsWith('Value'))],
  additionalProperties: false,
})

/** Mirrors the EditAction union in src/types.ts. */
export const EDIT_TOOL = {
  name: TOOL_NAME,
  description:
    'Apply edits to the audio stems. Use one call with every edit the request needs. Gains are multipliers: 1 is unchanged, 0 is silent, 2 is the maximum.',
  input_schema: {
    type: 'object',
    properties: {
      actions: {
        type: 'array',
        maxItems: 20,
        items: {
          oneOf: [
            actionVariant('set_gain', { value: gain }),
            actionVariant('set_mute', { muted: { type: 'boolean' } }),
            actionVariant('set_solo', { solo: { type: 'boolean' } }),
            actionVariant('set_automation', {
              keyframes: {
                type: 'array',
                minItems: 1,
                maxItems: 64,
                items: {
                  type: 'object',
                  properties: { t: { type: 'number', minimum: 0 }, value: gain },
                  required: ['t', 'value'],
                  additionalProperties: false,
                },
              },
            }),
            actionVariant('clear_automation', {}),
            actionVariant('follow_intensity', {
              sourceStemIds: { type: 'array', minItems: 1, items: { type: 'string' } },
              minValue: gain,
              maxValue: gain,
            }),
          ],
        },
      },
    },
    required: ['actions'],
    additionalProperties: false,
  },
}

export function buildSystemPrompt({ stems, tracks, duration }) {
  const stemLines = stems.map((s) => `- ${s.id}: "${s.label}" (${s.kind}, ${s.duration}s)`).join('\n')
  return `You edit a multitrack audio mix by calling the ${TOOL_NAME} tool.

Timeline duration: ${duration} seconds.
Stems:
${stemLines}

Current track state as JSON:
${JSON.stringify(tracks)}

Guidelines:
- Automation is a gain curve, linearly interpolated between keyframes, multiplying the fader gain. Keyframe times are seconds within the timeline.
- set_automation replaces a stem's whole curve, so include existing keyframes that should stay.
- follow_intensity makes a stem's gain follow how loud the source stems are. This measures loudness, it does not detect emotion, so describe it that way. minValue is the gain when sources are quiet, maxValue when they are loud. Set maxValue below minValue to duck.
- If the request is ambiguous or no stem matches, do not call the tool. Ask one short question instead.
- Before calling the tool, say in one short sentence what you are changing. The turn ends at the tool call.`
}

/** Turn a Messages API response into { actions, message }. Throws if it is unusable. */
export function parseMessage(data) {
  if (data?.stop_reason === 'refusal') throw new SafeError('model declined the request')
  if (data?.stop_reason === 'max_tokens') throw new SafeError('model output was cut off')
  const content = Array.isArray(data?.content) ? data.content : []
  const calls = content.filter((b) => b.type === 'tool_use' && b.name === TOOL_NAME)
  // The text beside a call describes the edit, so a bad call must not pass as a reply.
  if (calls.some((b) => !Array.isArray(b.input?.actions))) throw new SafeError('model returned malformed edits')
  const actions = calls.flatMap((b) => b.input.actions)
  const message = content
    .filter((b) => b.type === 'text')
    .map((b) => b.text)
    .join(' ')
    .trim()
  return { actions, message }
}

export async function callAgent(body, { apiKey, model }) {
  const res = await fetch(ANTHROPIC_URL, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    body: JSON.stringify({
      model,
      max_tokens: MAX_TOKENS,
      system: buildSystemPrompt(body),
      tools: [EDIT_TOOL],
      tool_choice: { type: 'auto' },
      messages: [{ role: 'user', content: body.prompt }],
    }),
  })
  // Status only: an upstream error body is not forwarded to the browser.
  if (!res.ok) throw new SafeError(`Anthropic API returned ${res.status}`)
  return parseMessage(await res.json())
}

function isAgentRequest(b) {
  return (
    typeof b?.prompt === 'string' &&
    b.prompt.length > 0 &&
    Array.isArray(b.stems) &&
    Array.isArray(b.tracks) &&
    typeof b.duration === 'number' &&
    Number.isFinite(b.duration)
  )
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (c) => {
      size += c.length
      // Keep draining, so the client can still read the response.
      if (size > MAX_BODY_BYTES) reject(new SafeError('body too large'))
      else chunks.push(c)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

export function createAgentServer(config) {
  return createServer(async (req, res) => {
    const send = (status, body) => {
      res.writeHead(status, { 'content-type': 'application/json' })
      res.end(JSON.stringify(body))
    }
    const origin = req.headers.origin
    // CORS alone only hides the response. Refusing here stops other sites from
    // spending the API key through a visitor's browser.
    if (origin !== undefined && origin !== ALLOWED_ORIGIN) return send(403, { error: 'origin not allowed' })
    if (origin === ALLOWED_ORIGIN) {
      res.setHeader('Access-Control-Allow-Origin', ALLOWED_ORIGIN)
      res.setHeader('Vary', 'Origin')
    }
    if (req.url !== '/agent') return send(404, { error: 'not found' })
    if (req.method === 'OPTIONS') {
      res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
      res.setHeader('Access-Control-Allow-Headers', 'content-type')
      res.writeHead(204)
      return res.end()
    }
    if (req.method !== 'POST') return send(405, { error: 'method not allowed' })

    let body
    try {
      body = JSON.parse(await readBody(req))
    } catch (err) {
      return send(err instanceof SafeError ? 413 : 400, { error: 'invalid request body' })
    }
    if (!isAgentRequest(body)) return send(400, { error: 'invalid request body' })

    try {
      send(200, await callAgent(body, config))
    } catch (err) {
      // err.message is excluded: it can quote the key. The name and code cannot.
      const code = err?.cause?.code ? ` ${err.cause.code}` : ''
      const message = err instanceof SafeError ? err.message : `Anthropic API request failed (${err?.name}${code})`
      console.error('agent request failed:', message)
      send(502, { error: message })
    }
  })
}

function main() {
  const missing = ['ANTHROPIC_API_KEY', 'AGENT_MODEL'].filter((name) => !process.env[name])
  if (missing.length > 0) {
    console.error(`Refusing to start: set ${missing.join(' and ')} in the environment.`)
    process.exit(1)
  }
  const port = Number(process.env.AGENT_PORT) || DEFAULT_PORT
  const server = createAgentServer({ apiKey: process.env.ANTHROPIC_API_KEY, model: process.env.AGENT_MODEL })
  server.listen(port, '127.0.0.1', () => console.log(`Agent server on http://127.0.0.1:${port}/agent`))
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main()
