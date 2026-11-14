import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { initialTracks } from '../edits'
import type { Stem } from '../types'
import { AGENT_TIMEOUT_MS, createAgent, HttpAgent, RegexAgent } from './index'

const STEMS: Stem[] = [{ id: 's', label: 'Siren', kind: 'sfx', url: 'blob:secret', duration: 20 }]
const req = { prompt: 'mute the siren', stems: STEMS, tracks: initialTracks(STEMS), duration: 20 }

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

describe('RegexAgent', () => {
  it('wraps parsePrompt', async () => {
    const res = await new RegexAgent().run(req)
    expect(res.actions).toEqual([{ type: 'set_mute', stemId: 's', muted: true }])
    expect(res.message).toBe('Muted Siren.')
  })
})

describe('createAgent', () => {
  it('picks HttpAgent only when a URL is set', () => {
    expect(createAgent('http://x/agent')).toBeInstanceOf(HttpAgent)
    expect(createAgent('')).toBeInstanceOf(RegexAgent)
    expect(createAgent(undefined)).toBeInstanceOf(RegexAgent)
  })
})

describe('HttpAgent', () => {
  const fetchMock = vi.fn<typeof fetch>()
  const agent = new HttpAgent('http://agent.test/agent')

  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('returns validated actions and the server message', async () => {
    fetchMock.mockResolvedValue(json({ actions: [{ type: 'set_gain', stemId: 's', value: 1.5 }], message: 'Done.' }))
    const res = await agent.run(req)
    expect(res).toEqual({ actions: [{ type: 'set_gain', stemId: 's', value: 1.5 }], message: 'Done.' })
  })

  it('accepts an empty action list, for example a clarifying question', async () => {
    fetchMock.mockResolvedValue(json({ actions: [], message: 'Which stem?' }))
    expect(await agent.run(req)).toEqual({ actions: [], message: 'Which stem?' })
  })

  it('does not claim an edit when the server sent neither actions nor text', async () => {
    fetchMock.mockResolvedValue(json({ actions: [], message: '' }))
    expect((await agent.run(req)).message).toBe('The agent made no changes.')
  })

  it('posts stems without their URLs', async () => {
    fetchMock.mockResolvedValue(json({ actions: [] }))
    await agent.run(req)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://agent.test/agent')
    expect(init?.method).toBe('POST')
    const body = JSON.parse(init?.body as string)
    expect(body.prompt).toBe(req.prompt)
    expect(body.duration).toBe(20)
    expect(body.stems[0]).toEqual({ id: 's', label: 'Siren', kind: 'sfx', duration: 20 })
  })

  it.each([
    ['network failure', () => Promise.reject(new Error('down')), 'down'],
    ['non-2xx status', () => Promise.resolve(json({}, 500)), 'HTTP 500'],
    ['non-JSON body', () => Promise.resolve(new Response('oops')), 'Agent unavailable'],
    ['missing actions', () => Promise.resolve(json({ message: 'hi' })), 'invalid output'],
    ['null body', () => Promise.resolve(json(null)), 'invalid output'],
    ['invalid actions', () => Promise.resolve(json({ actions: [{ type: 'set_gain', stemId: 'x', value: 1 }] })), 'invalid output'],
  ])('falls back to the regex parser on %s', async (_, impl, reason) => {
    fetchMock.mockImplementation(impl as typeof fetch)
    const res = await agent.run(req)
    expect(res.actions).toEqual([{ type: 'set_mute', stemId: 's', muted: true }])
    expect(res.message).toContain('used the rule-based parser')
    expect(res.message).toContain(reason)
  })

  it('shows the reason the server gave for a failure', async () => {
    fetchMock.mockResolvedValue(json({ error: 'model declined the request' }, 502))
    const res = await agent.run(req)
    expect(res.message).toContain('model declined the request')
  })

  it('aborts after the timeout and falls back', async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(
      (_url, init) =>
        new Promise((_res, rej) => {
          init?.signal?.addEventListener('abort', () => rej(new DOMException('Aborted', 'AbortError')))
        }),
    )
    const pending = agent.run(req)
    await vi.advanceTimersByTimeAsync(AGENT_TIMEOUT_MS)
    const res = await pending
    expect(res.message).toContain('timed out')
    expect(res.actions).toHaveLength(1)
  })
})
