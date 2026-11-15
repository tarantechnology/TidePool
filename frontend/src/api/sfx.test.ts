import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createSfxPreview } from './backend'
import type { SfxInsert } from './backend'

const config = { baseUrl: 'http://api.test:8000', token: 'tok' }
const fetchMock = vi.fn<typeof fetch>()

describe('createSfxPreview', () => {
  beforeEach(() => {
    fetchMock.mockReset()
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ candidate_id: 'c1', status: 'queued' })))
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('posts only the insert fields the backend validates, with the request text', async () => {
    const insert: SfxInsert = { sfx_query: 'vine boom', sfx_id: 'vine_boom', title: 'Deep boom (vine-boom style)', anchor: { type: 'word', value: 'what', relation: 'at', occurrence: null }, offset: 0, gain_db: -3, duck: true }
    expect(await createSfxPreview(config, 'p1', [insert], "vine boom when he says 'what'")).toBe('c1')
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://api.test:8000/api/projects/p1/sfx-preview')
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      inserts: [{ sfx_query: 'vine boom', sfx_id: 'vine_boom', anchor: { type: 'word', value: 'what', relation: 'at', occurrence: null }, offset: 0, gain_db: -3, duck: true }],
      request: "vine boom when he says 'what'",
    })
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get('Authorization')).toBe('Bearer tok')
  })
})
