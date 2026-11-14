import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BackendStem } from './backend'
import { createCandidate, describeRouting, tagLine, fetchStemBlobUrl, getRoute, processProject, splitTracks, toCapabilities, toRefinement, toStem } from './backend'
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

describe('passes on separation requests', () => {
  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('process sends passes above one, and leaves the default out', async () => {
    fetchMock.mockImplementation(async () => new Response(JSON.stringify({ job_id: 'j1' })))
    await processProject(config, 'p1', ['dog'], NO_VISUAL_PROMPT, false, 3)
    await processProject(config, 'p1', ['dog'], NO_VISUAL_PROMPT)
    await processProject(config, 'p1', ['dog'], NO_VISUAL_PROMPT, false, 1)
    expect(bodyOf(0)).toEqual({ prompts: ['dog'], passes: 3, visual_prompt: {} })
    expect(bodyOf(1)).not.toHaveProperty('passes')
    expect(bodyOf(2)).not.toHaveProperty('passes')
  })

  it('candidates send passes above one, and leave the default out', async () => {
    fetchMock.mockImplementation(async () => new Response(JSON.stringify({ candidate_id: 'c1' })))
    await createCandidate(config, 'p1', null, 'dog', 1, { profile: 'preview', passes: 2 })
    await createCandidate(config, 'p1', 's1', 'dog', 1, { profile: 'preview', passes: 1 })
    await createCandidate(config, 'p1', null, 'dog', 1)
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://api.test:8000/api/projects/p1/candidates')
    expect(bodyOf(0)).toEqual({ stem_id: null, prompt: 'dog', reranking_candidates: 1, profile: 'preview', passes: 2 })
    expect(bodyOf(1)).toEqual({ stem_id: 's1', prompt: 'dog', reranking_candidates: 1, profile: 'preview' })
    expect(bodyOf(2)).not.toHaveProperty('passes')
  })
})

describe('fetchStemBlobUrl', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('bypasses the HTTP cache, since Extract more rewrites audio at the same URL', async () => {
    fetchMock.mockReset()
    fetchMock.mockResolvedValue(new Response(new Blob([new Uint8Array([1, 2])])))
    vi.stubGlobal('fetch', fetchMock)
    const url = await fetchStemBlobUrl(config, '/api/stems/s1/audio')
    expect(fetchMock.mock.calls[0][1]?.cache).toBe('no-store')
    URL.revokeObjectURL(url)
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

  it('keeps instrument tags, best first, and shows the ones the name does not already say', () => {
    const stem = toStem(row({ name: 'Piano', label: 'other', role: 'target', tags: [{ label: 'Piano', score: 0.7 }, { label: ' Strings ', score: 0.3 }, { label: '', score: 0.1 }] }), 'blob:x')
    expect(stem.label).toBe('Piano')
    expect(stem.tags).toEqual(['Piano', 'Strings'])
    expect(tagLine(stem.label, stem.tags)).toBe('Strings')
    expect(tagLine('Vocals', ['Female singing', 'Choir'])).toBe('Female singing · Choir')
    expect(toStem(row({ tags: null }), 'blob:x')).not.toHaveProperty('tags')
  })

  it('shows a music stem by its display name, not its canonical label', () => {
    expect(toStem(row({ name: 'Vocals', label: 'vocals', role: 'target' }), 'blob:x').label).toBe('Vocals')
    expect(toStem(row({ name: 'singing', label: 'vocals', role: 'target' }), 'blob:x').label).toBe('singing')
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

  it('maps refinement when the backend sends it', () => {
    const refinement = { passes_requested: 3, passes_run: 2, per_pass_energy_db: [-12.3, -30.8], stopped_early: true }
    expect(toStem(row({ refinement }), 'blob:x').refinement).toEqual({
      passesRequested: 3,
      passesRun: 2,
      perPassEnergyDb: [-12.3, -30.8],
      stoppedEarly: true,
    })
  })

  it('has no refinement key when it is absent or null', () => {
    expect(toStem(row({}), 'blob:x')).not.toHaveProperty('refinement')
    expect(toStem(row({ refinement: null }), 'blob:x')).not.toHaveProperty('refinement')
  })
})

describe('toRefinement', () => {
  it('drops a block without a usable passes_run', () => {
    expect(toRefinement({ passes_requested: 2 })).toBeUndefined()
    expect(toRefinement({ passes_run: 0 })).toBeUndefined()
    expect(toRefinement('2 passes')).toBeUndefined()
  })

  it('fills in what is missing or malformed instead of failing the load', () => {
    expect(toRefinement({ passes_run: 2, per_pass_energy_db: [-10, null, 'x', -20] })).toEqual({
      passesRequested: 2,
      passesRun: 2,
      perPassEnergyDb: [-10, -20],
      stoppedEarly: false,
    })
  })
})

describe('music splits and routing', () => {
  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('asks /process for music stems with no prompts, and a subset with its edit', async () => {
    fetchMock.mockImplementation(async () => new Response(JSON.stringify({ job_id: 'j1' })))
    expect(await splitTracks(config, 'p1')).toBe('j1')
    expect(bodyOf(0)).toEqual({ prompts: [], visual_prompt: {}, mode: 'music_stems' })
    await splitTracks(config, 'p1', ['vocals'], { action: 'mute' })
    expect(bodyOf(1)).toEqual({ prompts: [], visual_prompt: {}, mode: 'music_stems', stems: ['vocals'], edit_scope: { action: 'mute' } })
    // Everyone else leaves the mode to the service.
    await processProject(config, 'p1', ['dog'], NO_VISUAL_PROMPT)
    expect(bodyOf(2)).not.toHaveProperty('mode')
  })

  it('reads the route decision and describes it', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ mode: 'music_stems', engine: 'demucs', model_id: 'htdemucs_ft', reason: 'music 0.97', scores: null, stems: null })))
    const route = await getRoute(config, 'p1')
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://api.test:8000/api/projects/p1/route')
    expect(route).toEqual({ mode: 'music_stems', engine: 'demucs', modelId: 'htdemucs_ft', reason: 'music 0.97', stems: null })
    expect(describeRouting(route!)).toBe('Music stems · demucs (htdemucs_ft): music 0.97')
    expect(describeRouting({ mode: 'prompted', engine: null, modelId: null, reason: null, stems: null })).toBe('Prompted separation')
  })

  it('reads the music_stems capability, and treats a missing one as absent', () => {
    expect(toCapabilities({ music_stems: { available: true, stems: ['vocals', 'drums', 3] } }).musicStems).toEqual({ available: true, stems: ['vocals', 'drums'] })
    expect(toCapabilities({}).musicStems).toBeUndefined()
  })
})
