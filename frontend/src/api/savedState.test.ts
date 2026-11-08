import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { applyAction } from '../edits'
import type { Stem, TrackState } from '../types'
import { toTrackState } from './backend'
import type { SavedStem } from './backend'
import { loadFromSource } from './loadStems'
import { FileStemSource, HttpStemSource } from './stemSources'
import type { StemSource } from './stemSources'
import { StemSyncRunner, SYNC_DEBOUNCE_MS } from './stemSync'

const config = { baseUrl: 'http://api.test:8000', token: 'tok' }
const AUTOMATION = [
  { time: 1, value: 0.5 },
  { time: 3, value: 1.5 },
]
const saved: SavedStem = {
  id: 's1',
  revision: 0,
  name: 'dog',
  duration: 4,
  audio_url: '/audio/s1',
  volume: 0.4,
  muted: true,
  solo: true,
  automation: AUTOMATION,
  start_offset: 2,
  mute_ranges: [[1, 1.5]],
}
// What the job holds from processing, which is older than the project.
const snapshot = { ...saved, volume: 1, muted: false, solo: false, automation: [], start_offset: 0, mute_ranges: [] }

const fetchMock = vi.fn<typeof fetch>()
const requests = (method = 'GET') =>
  fetchMock.mock.calls.filter(([, init]) => (init?.method ?? 'GET') === method).map(([url]) => String(url))
const json = (body: unknown) => new Response(JSON.stringify(body))

function backend({ job = {}, stems = [saved] as unknown }: { job?: object; stems?: unknown } = {}) {
  fetchMock.mockImplementation(async (input, init) => {
    const url = String(input)
    if (init?.method === 'PATCH') return json({ revision: 1 })
    if (url.endsWith('/api/jobs/j1')) return json({ status: 'completed', project_id: 'p1', stems: [snapshot], ...job })
    if (url.endsWith('/api/projects/p1/stems')) {
      return stems instanceof Response ? stems : json(stems)
    }
    return new Response('wav')
  })
}

async function load() {
  const source = new HttpStemSource(config, 'j1')
  const stems = await source.load()
  return { source, stems }
}

/** Follows a loaded source the way useStemSync does. */
function follow(source: StemSource, tracks: TrackState[]) {
  const errors: string[] = []
  const runner = new StemSyncRunner((m) => errors.push(m), (m) => errors.push(m))
  runner.apply(source.syncTarget ?? null, tracks)
  return { runner, errors }
}

beforeEach(() => {
  vi.useFakeTimers()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} }))
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('toTrackState', () => {
  it('maps every saved field', () => {
    expect(toTrackState(saved)).toEqual({
      stemId: 's1',
      gain: 0.4,
      muted: true,
      solo: true,
      automation: [
        { t: 1, value: 0.5 },
        { t: 3, value: 1.5 },
      ],
    })
  })

  it('keeps volume as is, since backend volume and gain share the range 0 to 2', () => {
    expect(toTrackState({ ...saved, volume: 2 }).gain).toBe(2)
    expect(toTrackState({ ...saved, volume: 0 }).gain).toBe(0)
  })

  it('does not rewrite a saved curve that is closer together than the editor would make it', () => {
    const automation = [
      { time: 1, value: 1 },
      { time: 1.0002, value: 0 },
    ]
    expect(toTrackState({ ...saved, automation }).automation).toEqual([
      { t: 1, value: 1 },
      { t: 1.0002, value: 0 },
    ])
  })
})

describe('restoring saved state', () => {
  it('reads the project, not the job snapshot', async () => {
    backend()
    const { source } = await load()
    expect(requests()).toContain('http://api.test:8000/api/projects/p1/stems')
    expect(source.savedTracks).toEqual([toTrackState(saved)])
    expect(source.syncTarget).toEqual({ config, projectId: 'p1', revisions: { s1: 0 } })
    expect(source.syncIssue).toBeUndefined()
  })

  it('sends nothing on load', async () => {
    backend()
    const { source } = await load()
    follow(source, source.savedTracks!)
    await vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS * 2)
    expect(requests('PATCH')).toEqual([])
  })

  it('a fader change sends the saved automation, mute and solo unchanged', async () => {
    backend()
    const { source } = await load()
    const { runner } = follow(source, source.savedTracks!)
    runner.apply(
      source.syncTarget!,
      applyAction(source.savedTracks!, { type: 'set_gain', stemId: 's1', value: 0.8 }),
    )
    await vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS)
    const patch = fetchMock.mock.calls.find(([, init]) => init?.method === 'PATCH')
    expect(JSON.parse(patch?.[1]?.body as string)).toEqual({
      volume: 0.8,
      expected_revision: 0,
      muted: true,
      solo: true,
      automation: AUTOMATION,
    })
  })

  it('takes offsets, ranges and audio from the saved row, not the snapshot', async () => {
    backend({ stems: [{ ...saved, audio_url: '/audio/current' }] })
    const { stems } = await load()
    expect(requests()).toContain('http://api.test:8000/audio/current')
    expect(stems[0]).toMatchObject({ startOffset: 2, muteRanges: [[1, 1.5]] })
  })

  it('never sends the offset or the ranges', async () => {
    backend()
    const { source } = await load()
    const { runner } = follow(source, source.savedTracks!)
    runner.apply(source.syncTarget!, applyAction(source.savedTracks!, { type: 'set_gain', stemId: 's1', value: 0.8 }))
    await vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS)
    const patch = fetchMock.mock.calls.find(([, init]) => init?.method === 'PATCH')
    expect(Object.keys(JSON.parse(patch?.[1]?.body as string)).sort()).toEqual(['automation', 'expected_revision', 'muted', 'solo', 'volume'])
  })
})

describe('when saved state cannot be trusted', () => {
  const expectPlaybackOnly = async (source: HttpStemSource, stems: Stem[], reason: string) => {
    expect(stems[0]).toMatchObject({ id: 's1', label: 'dog' })
    expect(source.syncTarget).toBeUndefined()
    expect(source.savedTracks).toBeUndefined()
    expect(source.syncIssue).toContain(reason)
    expect(source.syncIssue).toContain('stay in this browser')
    const { runner } = follow(source, [toTrackState(saved)])
    runner.apply(null, [{ ...toTrackState(saved), gain: 0.1 }])
    await vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS * 2)
    expect(requests('PATCH')).toEqual([])
  }

  it('keeps playing and never patches when the fetch fails', async () => {
    backend({ stems: new Response(JSON.stringify({ detail: 'nope' }), { status: 500 }) })
    const { source, stems } = await load()
    await expectPlaybackOnly(source, stems, 'Backend returned 500')
  })

  it('does the same when the network fails', async () => {
    backend()
    const real = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation((input, init) =>
      String(input).endsWith('/api/projects/p1/stems') ? Promise.reject(new Error('offline')) : real(input, init),
    )
    const { source, stems } = await load()
    await expectPlaybackOnly(source, stems, 'offline')
  })

  it('never patches when the job has no project id', async () => {
    backend({ job: { project_id: undefined } })
    const { source, stems } = await load()
    await expectPlaybackOnly(source, stems, 'project_id')
    expect(requests().some((u) => u.endsWith('/stems'))).toBe(false)
  })

  it('never patches when processing replaced the stems of this job', async () => {
    backend({ stems: [{ ...saved, id: 'other' }] })
    const { source, stems } = await load()
    await expectPlaybackOnly(source, stems, 'no longer')
  })

  const malformed: Array<[string, unknown]> = [
    ['a missing field', [{ ...saved, muted: undefined }]],
    ['volume out of range', [{ ...saved, volume: 3 }]],
    ['volume that is not a number', [{ ...saved, volume: '1' }]],
    ['automation out of range', [{ ...saved, automation: [{ time: 1, value: 5 }] }]],
    ['automation out of order', [{ ...saved, automation: [{ time: 2, value: 1 }, { time: 2, value: 0 }] }]],
    ['something that is not a list', { stems: [] }],
    ['a missing offset', [{ ...saved, start_offset: undefined }]],
    ['a negative offset', [{ ...saved, start_offset: -1 }]],
    ['a malformed mute range', [{ ...saved, mute_ranges: [[1]] }]],
  ]
  for (const [name, stems] of malformed) {
    it(`never patches on ${name}`, async () => {
      backend({ stems })
      const { source, stems: loaded } = await load()
      await expectPlaybackOnly(source, loaded, 'malformed')
    })
  }
})

describe('loadFromSource', () => {
  const stem: Stem = { id: 's1', label: 'dog', kind: 'sfx', url: 'blob:x', duration: 4 }
  const engine = { replaceStems: vi.fn(async (stems: Stem[]) => stems) }
  const flush = vi.fn(async () => {})
  beforeEach(() => {
    engine.replaceStems.mockClear()
    flush.mockClear()
  })

  it('starts from saved tracks', async () => {
    backend()
    const res = await loadFromSource(new HttpStemSource(config, 'j1'), engine, () => true, flush)
    expect(res?.tracks).toEqual([toTrackState(saved)])
  })

  it('starts from defaults when nothing is saved, for playback', async () => {
    const source: StemSource = { load: async () => [stem] }
    const res = await loadFromSource(source, engine, () => true, flush)
    expect(res?.tracks).toEqual([{ stemId: 's1', gain: 1, muted: false, solo: false, automation: [] }])
  })

  it('lands queued edits before reading saved state', async () => {
    const order: string[] = []
    backend()
    const real = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation((input, init) => {
      order.push('read')
      return real(input, init)
    })
    flush.mockImplementationOnce(async () => {
      order.push('flush')
    })
    await loadFromSource(new HttpStemSource(config, 'j1'), engine, () => true, flush)
    expect(order[0]).toBe('flush')
    expect(order[1]).toBe('read')
  })

  it('keeps the current editor when flushing unsaved edits fails', async () => {
    const source = new HttpStemSource(config, 'j1')
    const read = vi.spyOn(source, 'load')
    flush.mockRejectedValueOnce(new Error('Unsaved edits'))
    await expect(loadFromSource(source, engine, () => true, flush)).rejects.toThrow('Unsaved edits')
    expect(read).not.toHaveBeenCalled()
    expect(engine.replaceStems).not.toHaveBeenCalled()
  })

  it('flushes pending edits before replacing them with file imports', async () => {
    await loadFromSource(new FileStemSource([new File(['x'], 'a.wav')]), engine, () => true, flush)
    expect(flush).toHaveBeenCalledOnce()
  })

  it('a replaced load changes nothing, wherever it is overtaken', async () => {
    backend()
    for (const overtakenAfter of [1, 2, 3]) {
      engine.replaceStems.mockClear()
      let checks = 0
      const isLatest = () => ++checks < overtakenAfter
      const res = await loadFromSource(new HttpStemSource(config, 'j1'), engine, isLatest, flush)
      expect(res, `overtaken at check ${overtakenAfter}`).toBeNull()
      expect(engine.replaceStems).toHaveBeenCalledTimes(overtakenAfter === 3 ? 1 : 0)
    }
  })

  it('returns null when the engine reports a newer load', async () => {
    const source: StemSource = { load: async () => [stem] }
    const res = await loadFromSource(source, { replaceStems: async () => null }, () => true, flush)
    expect(res).toBeNull()
  })
})
