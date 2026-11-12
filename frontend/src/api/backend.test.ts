import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BackendStem } from './backend'
import { processProject, toStem } from './backend'
import { buildVisualPrompt, NO_VISUAL_PROMPT } from './separation'

const config = { baseUrl: 'http://api.test:8000', token: 'tok' }
const fetchMock = vi.fn<typeof fetch>()

const bodyOf = (call = 0) => JSON.parse(String(fetchMock.mock.calls[call][1]?.body)) as Record<string, unknown>

describe('processProject', () => {
  beforeEach(() => {
    fetchMock.mockReset()
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ job_id: 'j1' })))
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('always sends visual_prompt, so the backend cannot substitute the detections document', async () => {
    // The field being absent is what made the backend fall back to the whole
    // stored detections document, which is megabytes over the HTTP adapter.
    await processProject(config, 'p1', ['dog barking'], NO_VISUAL_PROMPT)
    expect(bodyOf()).toEqual({ prompts: ['dog barking'], visual_prompt: {} })
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://api.test:8000/api/projects/p1/process')
  })

  it('sends the selection when there is one', async () => {
    const selection = { detection_id: 'd1', track_id: 2, label: 'dog' }
    await processProject(config, 'p1', ['dog barking'], buildVisualPrompt(selection, '1', '3'))
    expect(bodyOf()).toEqual({
      prompts: ['dog barking'],
      visual_prompt: { detection_id: 'd1', track_id: 2, label: 'dog', interval: { start_seconds: 1, end_seconds: 3 } },
    })
  })
})

describe('toStem', () => {
  const row = (over: Partial<BackendStem>): BackendStem => ({
    id: 's1',
    name: 'dog barking',
    duration: 4,
    audio_url: '/audio/s1',
    ...over,
  })

  it('gives the residual its own kind, by role', () => {
    expect(toStem(row({ role: 'residual', name: 'residual' }), 'blob:x').kind).toBe('residual')
  })

  it('leaves a target with the default kind, which is not guessed from the name', () => {
    expect(toStem(row({ role: 'target', name: 'dialogue' }), 'blob:x').kind).toBe('sfx')
  })

  it('prefers the backend label for display, and falls back to the name', () => {
    expect(toStem(row({ label: 'Dog barking' }), 'blob:x').label).toBe('Dog barking')
    expect(toStem(row({ label: '   ' }), 'blob:x').label).toBe('dog barking')
    expect(toStem(row({}), 'blob:x').label).toBe('dog barking')
  })

  it('works unchanged for a backend that sends none of the new fields', () => {
    expect(toStem(row({}), 'blob:x')).toEqual({
      id: 's1',
      label: 'dog barking',
      kind: 'sfx',
      url: 'blob:x',
      duration: 4,
      startOffset: 0,
      muteRanges: [],
    })
  })
})
