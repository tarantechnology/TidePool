import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SavedStem } from './backend'
import { startRefine } from './backend'
import {
  clampPasses,
  passChoices,
  passLabel,
  refineBlockReason,
  refineConflictMessage,
  refinementBadge,
  runRefine,
} from './refine'

const config = { baseUrl: 'http://api.test:8000', token: 'tok' }
const fetchMock = vi.fn<typeof fetch>()
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

const stem = (over: Partial<SavedStem> & { id: string }): SavedStem => ({
  name: over.id,
  duration: 4,
  audio_url: `/audio/${over.id}`,
  revision: 0,
  volume: 1,
  muted: false,
  solo: false,
  automation: [],
  start_offset: 0,
  mute_ranges: [],
  ...over,
})

describe('pass choices', () => {
  it('offers 1 to 3 when the backend does not say', () => {
    expect(passChoices(undefined)).toEqual([1, 2, 3])
    expect(passChoices('3')).toEqual([1, 2, 3])
    expect(passChoices(0)).toEqual([1, 2, 3])
  })

  it("follows the backend's max_passes, capped at the contract's 3", () => {
    expect(passChoices(2)).toEqual([1, 2])
    expect(passChoices(1)).toEqual([1])
    expect(passChoices(5)).toEqual([1, 2, 3])
  })

  it('labels the ends', () => {
    expect([1, 2, 3].map(passLabel)).toEqual(['1 (fast)', '2', '3 (thorough)'])
  })

  it('clamps a choice the backend has since lowered its limit below', () => {
    expect(clampPasses(3, [1, 2])).toBe(2)
    expect(clampPasses(2, [1, 2, 3])).toBe(2)
  })
})

describe('refinementBadge', () => {
  const r = { passesRequested: 2, passesRun: 2, perPassEnergyDb: [-12.4, -30.6], stoppedEarly: false }

  it('says nothing for a single pass or a stem without refinement', () => {
    expect(refinementBadge(undefined)).toBeNull()
    expect(refinementBadge({ ...r, passesRequested: 1, passesRun: 1, perPassEnergyDb: [-10] })).toBeNull()
  })

  it('counts passes and lists per-pass energy rounded to 1 dB', () => {
    expect(refinementBadge(r)).toEqual({ text: '2 passes', title: 'Ran 2 of 2 passes. Energy per pass: -12 dB, -31 dB' })
  })

  it('flags an early stop, even after one pass', () => {
    expect(refinementBadge({ ...r, passesRequested: 3, stoppedEarly: true })?.text).toBe('2 passes · stopped early')
    expect(refinementBadge({ passesRequested: 3, passesRun: 1, perPassEnergyDb: [], stoppedEarly: true })).toEqual({
      text: '1 pass · stopped early',
      title: 'Ran 1 of 3 passes.',
    })
  })

  it('does not print -0 dB', () => {
    expect(refinementBadge({ ...r, perPassEnergyDb: [-0.4, -40] })?.title).toContain('0 dB, -40 dB')
    expect(refinementBadge({ ...r, perPassEnergyDb: [-0.4, -40] })?.title).not.toContain('-0 dB')
  })
})

describe('refineBlockReason', () => {
  const target = stem({ id: 't', role: 'target' })
  const residual = stem({ id: 'r', role: 'residual' })

  it('allows a target with one residual when nothing is running', () => {
    expect(refineBlockReason(target, [target, residual], { busy: false })).toBeNull()
    expect(refineBlockReason(target, [target, residual], { busy: false, supported: true })).toBeNull()
  })

  it('blocks the residual and role-less stems', () => {
    expect(refineBlockReason(residual, [target, residual], { busy: false })).toMatch(/Only an extracted sound/)
    const plain = stem({ id: 'p' })
    expect(refineBlockReason(plain, [plain, residual], { busy: false })).toMatch(/Only an extracted sound/)
  })

  it('blocks without exactly one residual', () => {
    expect(refineBlockReason(target, [target], { busy: false })).toMatch(/no residual/)
    expect(refineBlockReason(target, [target, residual, stem({ id: 'r2', role: 'residual' })], { busy: false })).toMatch(/more than one residual/)
  })

  it('blocks while a job is busy, and on a backend that says it cannot refine', () => {
    expect(refineBlockReason(target, [target, residual], { busy: true })).toMatch(/Wait/)
    expect(refineBlockReason(target, [target, residual], { busy: false, supported: false })).toMatch(/cannot/)
  })
})

describe('refine requests', () => {
  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('posts to the stem refine route with bearer auth and no body by default', async () => {
    fetchMock.mockResolvedValue(json({ job_id: 'j9' }, 202))
    await expect(startRefine(config, 'p 1', 's/1')).resolves.toBe('j9')
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toBe('http://api.test:8000/api/projects/p%201/stems/s%2F1/refine')
    expect(init?.method).toBe('POST')
    expect(init?.body).toBeUndefined()
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer tok')
  })

  it('sends passes as JSON when given', async () => {
    fetchMock.mockResolvedValue(json({ job_id: 'j9' }, 202))
    await startRefine(config, 'p1', 's1', 2)
    const init = fetchMock.mock.calls[0][1]
    expect(JSON.parse(String(init?.body))).toEqual({ passes: 2 })
    expect(new Headers(init?.headers).get('Content-Type')).toBe('application/json')
  })

  it('polls the job to completion and reports progress', async () => {
    fetchMock
      .mockResolvedValueOnce(json({ job_id: 'j9' }, 202))
      .mockResolvedValueOnce(json({ status: 'processing', progress: 40 }))
      .mockResolvedValueOnce(json({ status: 'completed', progress: 100 }))
    const updates: string[] = []
    await expect(
      runRefine(config, 'p1', 's1', { intervalMs: 0, onUpdate: (s) => updates.push(`${s.state}:${s.progress}`) }),
    ).resolves.toBe('j9')
    expect(String(fetchMock.mock.calls[1][0])).toBe('http://api.test:8000/api/jobs/j9')
    expect(updates).toEqual(['processing:40', 'done:100'])
  })

  it("rejects with the job's error when it fails", async () => {
    fetchMock.mockResolvedValueOnce(json({ job_id: 'j9' }, 202)).mockResolvedValueOnce(json({ status: 'failed', error: 'GPU lost' }))
    await expect(runRefine(config, 'p1', 's1', { intervalMs: 0 })).rejects.toThrow('GPU lost')
  })

  it('surfaces a 409 detail word for word, and nothing for other failures', async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: 'A job is already running for this project' }, 409))
    const err = await runRefine(config, 'p1', 's1').catch((e: unknown) => e)
    expect(refineConflictMessage(err)).toBe('A job is already running for this project')
    expect(refineConflictMessage(new Error('Backend returned 500: boom'))).toBeNull()
    expect(refineConflictMessage(new Error('Backend returned 409'))).toMatch(/refused/)
  })

  it('stops polling when cancelled', async () => {
    const controller = new AbortController()
    fetchMock.mockResolvedValueOnce(json({ job_id: 'j9' }, 202)).mockImplementation(async () => {
      controller.abort()
      return json({ status: 'processing' })
    })
    await expect(runRefine(config, 'p1', 's1', { signal: controller.signal, intervalMs: 0 })).rejects.toHaveProperty('name', 'AbortError')
  })
})
