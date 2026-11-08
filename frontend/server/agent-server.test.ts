import { spawnSync } from 'node:child_process'
import type { AddressInfo } from 'node:net'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
// @ts-expect-error plain .mjs server, no type declarations
import { buildSystemPrompt, createAgentServer, EDIT_TOOL, parseMessage } from './agent-server.mjs'

const KEY = 'test-key-value-not-real'
const MODEL = 'test-model'
const body = {
  prompt: 'mute the siren',
  stems: [{ id: 's', label: 'Siren', kind: 'sfx', duration: 20 }],
  tracks: [{ stemId: 's', gain: 1, muted: false, solo: false, automation: [] }],
  duration: 20,
}

const realFetch = globalThis.fetch
const anthropic = vi.fn<(url: string, init: RequestInit) => Promise<Response>>()

const reply = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status })
const toolReply = (actions: unknown[]) =>
  reply({
    stop_reason: 'tool_use',
    content: [
      { type: 'text', text: 'Muted the siren.' },
      { type: 'tool_use', id: 'toolu_1', name: 'apply_edits', input: { actions } },
    ],
  })

describe('agent server over HTTP (Anthropic API mocked)', () => {
  const server = createAgentServer({ apiKey: KEY, model: MODEL })
  let base = ''

  const post = (data: unknown, headers: Record<string, string> = {}) =>
    realFetch(`${base}/agent`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: typeof data === 'string' ? data : JSON.stringify(data),
    })

  beforeAll(async () => {
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
    vi.stubGlobal('fetch', anthropic)
  })
  afterEach(() => anthropic.mockReset())
  afterAll(async () => {
    vi.unstubAllGlobals()
    await new Promise((r) => server.close(r))
  })

  it('sends a Messages API request with key, version, model and tools', async () => {
    anthropic.mockResolvedValue(toolReply([]))
    await post(body)
    const [url, init] = anthropic.mock.calls[0]
    expect(url).toBe('https://api.anthropic.com/v1/messages')
    const headers = init.headers as Record<string, string>
    expect(headers['x-api-key']).toBe(KEY)
    expect(headers['anthropic-version']).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    const sent = JSON.parse(init.body as string)
    expect(sent.model).toBe(MODEL)
    expect(sent.tools[0].name).toBe('apply_edits')
    expect(sent.tool_choice).toEqual({ type: 'auto' })
    expect(sent.messages).toEqual([{ role: 'user', content: body.prompt }])
    expect(sent.system).toContain('Siren')
    expect(sent.system).toContain('20 seconds')
    expect(sent.temperature).toBeUndefined()
  })

  it('converts tool_use blocks into actions and text into the message', async () => {
    const actions = [{ type: 'set_mute', stemId: 's', muted: true }]
    anthropic.mockResolvedValue(toolReply(actions))
    const res = await post(body)
    expect(res.status).toBe(200)
    expect(await res.json()).toEqual({ actions, message: 'Muted the siren.' })
  })

  it('allows the dev origin only', async () => {
    anthropic.mockResolvedValue(toolReply([]))
    const ok = await post(body, { origin: 'http://localhost:5173' })
    expect(ok.status).toBe(200)
    expect(ok.headers.get('access-control-allow-origin')).toBe('http://localhost:5173')

    const bad = await post(body, { origin: 'http://evil.test' })
    expect(bad.status).toBe(403)
    expect(bad.headers.get('access-control-allow-origin')).toBeNull()
    expect(anthropic).toHaveBeenCalledTimes(1)
  })

  it('answers a preflight without calling the API', async () => {
    const res = await realFetch(`${base}/agent`, { method: 'OPTIONS', headers: { origin: 'http://localhost:5173' } })
    expect(res.status).toBe(204)
    expect(res.headers.get('access-control-allow-methods')).toContain('POST')
    expect(anthropic).not.toHaveBeenCalled()
  })

  it('rejects bad requests without calling the API', async () => {
    expect((await post('not json')).status).toBe(400)
    expect((await post({ ...body, prompt: '' })).status).toBe(400)
    expect((await post({ ...body, stems: 'x' })).status).toBe(400)
    expect((await post({ ...body, duration: null })).status).toBe(400)
    expect((await realFetch(`${base}/agent`)).status).toBe(405)
    expect((await realFetch(`${base}/other`)).status).toBe(404)
    expect(anthropic).not.toHaveBeenCalled()
  })

  it('returns 502 with the status only when the API fails', async () => {
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => {})
    anthropic.mockResolvedValue(reply({ error: { message: `leaky ${KEY}` } }, 401))
    const res = await post(body)
    const text = await res.text()
    expect(res.status).toBe(502)
    expect(text).toContain('401')
    expect(text).not.toContain(KEY)
    quiet.mockRestore()
  })

  it('never repeats a thrown error, which can quote the key', async () => {
    const log = vi.spyOn(console, 'error').mockImplementation(() => {})
    anthropic.mockRejectedValue(new TypeError(`Headers.append: "${KEY}" is an invalid header value.`))
    const res = await post(body)
    expect(res.status).toBe(502)
    expect(await res.text()).not.toContain(KEY)
    expect(JSON.stringify(log.mock.calls)).not.toContain(KEY)
    log.mockRestore()
  })

  it('bounds the upstream call with a timeout', async () => {
    anthropic.mockResolvedValue(toolReply([]))
    await post(body)
    expect(anthropic.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal)
  })

  it('answers an oversized body with 413 instead of dropping the connection', async () => {
    const res = await post('x'.repeat(1_100_000))
    expect(res.status).toBe(413)
    expect(anthropic).not.toHaveBeenCalled()
  })
})

describe('parseMessage', () => {
  it('merges actions across tool_use blocks and ignores other tools', () => {
    const res = parseMessage({
      stop_reason: 'tool_use',
      content: [
        { type: 'tool_use', name: 'apply_edits', input: { actions: [{ a: 1 }] } },
        { type: 'tool_use', name: 'apply_edits', input: { actions: [{ a: 2 }] } },
        { type: 'tool_use', name: 'other', input: { actions: [{ a: 3 }] } },
      ],
    })
    expect(res.actions).toEqual([{ a: 1 }, { a: 2 }])
  })

  it('returns text with no actions for a clarifying question', () => {
    const res = parseMessage({ stop_reason: 'end_turn', content: [{ type: 'text', text: 'Which stem?' }] })
    expect(res).toEqual({ actions: [], message: 'Which stem?' })
  })

  it('skips thinking blocks', () => {
    const res = parseMessage({ stop_reason: 'end_turn', content: [{ type: 'thinking', thinking: '' }] })
    expect(res).toEqual({ actions: [], message: '' })
  })

  it('throws on a tool call with malformed input, rather than claiming an edit', () => {
    const data = {
      stop_reason: 'tool_use',
      content: [
        { type: 'text', text: 'Ducking the music.' },
        { type: 'tool_use', name: 'apply_edits', input: { actions: 'nope' } },
      ],
    }
    expect(() => parseMessage(data)).toThrow('malformed')
  })

  it('throws on refusal and on truncated output', () => {
    expect(() => parseMessage({ stop_reason: 'refusal', content: [] })).toThrow('declined')
    expect(() => parseMessage({ stop_reason: 'max_tokens', content: [] })).toThrow('cut off')
  })
})

describe('EDIT_TOOL', () => {
  it('mirrors every EditAction type', () => {
    const variants = EDIT_TOOL.input_schema.properties.actions.items.oneOf as Array<{
      properties: { type: { const: string } }
      required: string[]
    }>
    expect(variants.map((v) => v.properties.type.const).sort()).toEqual(
      ['clear_automation', 'follow_intensity', 'set_automation', 'set_gain', 'set_mute', 'set_solo'].sort(),
    )
    const follow = variants.find((v) => v.properties.type.const === 'follow_intensity')
    expect(follow?.required).toEqual(['type', 'stemId', 'sourceStemIds'])
  })
})

describe('buildSystemPrompt', () => {
  it('lists stems and says intensity is loudness', () => {
    const text = buildSystemPrompt(body)
    expect(text).toContain('s: "Siren"')
    expect(text).toContain('does not detect emotion')
  })
})

describe('startup', () => {
  const start = (env: Record<string, string>) =>
    spawnSync(process.execPath, ['server/agent-server.mjs'], {
      env: { PATH: process.env.PATH ?? '', ...env },
      encoding: 'utf8',
      timeout: 5000,
    })

  it('refuses to start without the key and does not print values', () => {
    const res = start({ AGENT_MODEL: 'visible-model-value' })
    expect(res.status).toBe(1)
    expect(res.stderr).toContain('ANTHROPIC_API_KEY')
    expect(res.stderr + res.stdout).not.toContain('visible-model-value')
  })

  it('refuses to start without the model and does not print the key', () => {
    const res = start({ ANTHROPIC_API_KEY: KEY })
    expect(res.status).toBe(1)
    expect(res.stderr).toContain('AGENT_MODEL')
    expect(res.stderr + res.stdout).not.toContain(KEY)
  })
})
