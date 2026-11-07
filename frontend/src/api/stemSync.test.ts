import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { validateActions } from '../agent/validate'
import { resolveActions } from '../audio/resolve'
import { applyAction, initialTracks, parsePrompt } from '../edits'
import { addKeyframe, moveKeyframe, removeKeyframe } from '../keyframes'
import type { Stem, TrackState } from '../types'
import { hasStepKeyframes, toSyncPatch } from './backend'
import { FileStemSource, HttpStemSource, MockStemSource } from './stemSources'
import type { StemSource } from './stemSources'
import { connectStemSync, StemSync, StemSyncRunner, SYNC_DEBOUNCE_MS } from './stemSync'

const target = { config: { baseUrl: 'http://api.test:8000', token: 'tok' }, projectId: 'p1' }

const track = (stemId: string, over: Partial<TrackState> = {}): TrackState => ({
  stemId,
  gain: 1,
  muted: false,
  solo: false,
  automation: [],
  ...over,
})

const ok = () => new Response('{}', { status: 200 })
const fail = (status: number, detail: string) => new Response(JSON.stringify({ detail }), { status })

const fetchMock = vi.fn<typeof fetch>()
const calls = () =>
  fetchMock.mock.calls.map(([url, init]) => ({
    url: String(url),
    method: init?.method,
    body: JSON.parse(init?.body as string) as unknown,
  }))
const gains = () => calls().map((c) => (c.body as { volume: number }).volume)
const settle = () => vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS)

beforeEach(() => {
  vi.useFakeTimers()
  fetchMock.mockReset()
  fetchMock.mockImplementation(async () => ok())
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('payload', () => {
  it('matches the documented PATCH shape exactly', async () => {
    const sync = new StemSync(target, [track('s1')], () => {})
    sync.update([
      track('s1', {
        gain: 0.5,
        muted: true,
        solo: true,
        automation: [
          { t: 0, value: 1 },
          { t: 10, value: 0.5 },
        ],
      }),
    ])
    await settle()
    expect(calls()).toEqual([
      {
        url: 'http://api.test:8000/api/projects/p1/stems/s1',
        method: 'PATCH',
        body: {
          volume: 0.5,
          muted: true,
          solo: true,
          automation: [
            { time: 0, value: 1 },
            { time: 10, value: 0.5 },
          ],
        },
      },
    ])
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers)
    expect(headers.get('Authorization')).toBe('Bearer tok')
    expect(headers.get('Content-Type')).toBe('application/json')
  })

  it('sends an empty list, so clearing automation reaches the backend', async () => {
    const curve = track('s1', {
      automation: [
        { t: 0, value: 1 },
        { t: 1, value: 0 },
      ],
    })
    const sync = new StemSync(target, [curve], () => {})
    sync.update([track('s1')])
    await settle()
    expect(calls()[0].body).toMatchObject({ automation: [] })
  })
})

describe('no edit can reach that guard', () => {
  const STEMS: Stem[] = [
    { id: 'a', label: 'Speaker A', kind: 'dialogue', url: 'blob:a', duration: 20 },
    { id: 's', label: 'Siren', kind: 'sfx', url: 'blob:s', duration: 20 },
  ]
  const expectSendable = (tracks: TrackState[]) => {
    for (const t of tracks) {
      expect(hasStepKeyframes(t.automation), JSON.stringify(t.automation)).toBe(false)
      expect(toSyncPatch(t).automation).toEqual(t.automation.map((k) => ({ time: k.t, value: k.value })))
    }
  }

  it('holds for parser output on every boundary, alone and in sequence', () => {
    const prompts = [
      'mute the siren from 0 to 5 seconds',
      'mute the siren from 15 to 20 seconds',
      'mute the siren from 15 to 99 seconds',
      'mute the siren at 12 seconds',
      'mute the siren from 0 to 20 seconds',
      'make the siren louder from 0 to 4 seconds',
      'make the siren quieter from 16 to 20 seconds',
      'set the siren to 30% from 0 to 20 seconds',
      'lower the siren by 20% at 18 seconds',
    ]
    let tracks = initialTracks(STEMS)
    for (const prompt of prompts) {
      const alone = parsePrompt(prompt, STEMS, initialTracks(STEMS)).actions
      expect(alone.length, prompt).toBe(1)
      expectSendable(alone.reduce(applyAction, initialTracks(STEMS)))
      tracks = parsePrompt(prompt, STEMS, tracks).actions.reduce(applyAction, tracks)
      expectSendable(tracks)
    }
  })

  it('holds for random drag edits', () => {
    // A fixed seed, so a failure reproduces.
    let seed = 12345
    const random = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648
      return seed / 2147483648
    }
    let tracks = initialTracks(STEMS)
    for (let i = 0; i < 2000; i++) {
      const curve = tracks[0].automation
      const point = { t: Math.round(random() * 40) / 2, value: random() * 2 }
      const index = Math.floor(random() * Math.max(1, curve.length))
      const op = random()
      const keyframes =
        op < 0.5 ? addKeyframe(curve, point, 20) : op < 0.9 ? moveKeyframe(curve, index, point, 20) : removeKeyframe(curve, index)
      tracks = applyAction(tracks, { type: 'set_automation', stemId: 'a', keyframes })
      expectSendable(tracks)
    }
  })

  it('holds for intensity output', () => {
    const data = Float32Array.from({ length: 48000 * 20 }, (_, i) => (i % 96000 < 48000 ? Math.sin(i / 8) : 0))
    const source = { getSamples: () => ({ data, sampleRate: 48000 }) }
    const action = { type: 'follow_intensity' as const, stemId: 's', sourceStemIds: ['a'] }
    const { actions } = resolveActions([action], source, 20)
    expect(actions[0].type).toBe('set_automation')
    expectSendable(actions.reduce(applyAction, initialTracks(STEMS)))
  })

  it('holds for agent output that contains a step', () => {
    const keyframes = [
      { t: 5, value: 1 },
      { t: 5, value: 0 },
      { t: 20, value: 0 },
      { t: 20, value: 1 },
    ]
    const res = validateActions([{ type: 'set_automation', stemId: 'a', keyframes }], STEMS, 20)
    expect(res.ok).toBe(true)
    if (res.ok) expectSendable(res.actions.reduce(applyAction, initialTracks(STEMS)))
  })
})

describe('flush', () => {
  const realTimers = async <T>(run: () => Promise<T>) => {
    vi.useRealTimers()
    return run()
  }

  it('sends a write waiting on the debounce and resolves once it landed', async () => {
    await realTimers(async () => {
      const sync = new StemSync(target, [track('s1')], () => {})
      sync.update([track('s1', { gain: 0.5 })])
      expect(fetchMock).not.toHaveBeenCalled()
      await sync.flush()
      expect(calls().map((c) => c.body)).toMatchObject([{ volume: 0.5 }])
    })
  })

  it('also sends an edit made while a write was in flight', async () => {
    await realTimers(async () => {
      const sync = new StemSync(target, [track('s1')], () => {})
      sync.update([track('s1', { gain: 0.5 })])
      const first = sync.flush()
      sync.update([track('s1', { gain: 0.25 })])
      await first
      expect(calls().map((c) => c.body)).toMatchObject([{ volume: 0.5 }, { volume: 0.25 }])
    })
  })

  it('finishes when a write fails, and the failure still shows', async () => {
    await realTimers(async () => {
      fetchMock.mockResolvedValue(new Response(null, { status: 500 }))
      const onError = vi.fn()
      const sync = new StemSync(target, [track('s1')], onError)
      sync.update([track('s1', { gain: 0.5 })])
      await sync.flush()
      expect(onError).toHaveBeenLastCalledWith(expect.stringContaining('500'))
    })
  })

  it('does nothing without a sync', async () => {
    await new StemSyncRunner(() => {}, () => {}).flush()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('the guard against curves the backend would reject', () => {
  // Nothing in the editor builds this any more. See the test after this block.
  const stepped = [
    { t: 11.95, value: 1 },
    { t: 12, value: 0 },
    { t: 20, value: 0 },
    { t: 20, value: 1 },
  ]

  it('finds equal-time keyframes', () => {
    expect(hasStepKeyframes(stepped)).toBe(true)
    expect(hasStepKeyframes(stepped.slice(0, 3))).toBe(false)
    expect(hasStepKeyframes([])).toBe(false)
  })

  it('leaves automation out rather than reshaping it, and keeps the rest', () => {
    expect(toSyncPatch(track('s', { gain: 0.5, automation: stepped }))).toEqual({
      volume: 0.5,
      muted: false,
      solo: false,
    })
  })

  it('still syncs the fader of a stem whose curve has a step', async () => {
    const sync = new StemSync(target, [track('s1', { automation: stepped })], () => {})
    sync.update([track('s1', { automation: stepped, gain: 0.25 })])
    await settle()
    expect(calls().map((c) => c.body)).toEqual([{ volume: 0.25, muted: false, solo: false }])
  })

  it('makes no call for a change that only touches the unsendable curve', async () => {
    const sync = new StemSync(target, [track('s1', { automation: stepped })], () => {})
    sync.update([track('s1', { automation: [...stepped] })])
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('debounce and coalescing', () => {
  it('sends nothing for the baseline or for unchanged tracks', async () => {
    const tracks = [track('s1'), track('s2')]
    const sync = new StemSync(target, tracks, () => {})
    sync.update(tracks)
    sync.update([track('s1'), track('s2')])
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('waits for the debounce', async () => {
    const sync = new StemSync(target, [track('s1')], () => {})
    sync.update([track('s1', { gain: 0.5 })])
    await vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS - 1)
    expect(fetchMock).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('coalesces a slider drag into one write of the final value', async () => {
    const sync = new StemSync(target, [track('s1')], () => {})
    for (let i = 1; i <= 20; i++) {
      sync.update([track('s1', { gain: i / 40 })])
      await vi.advanceTimersByTimeAsync(50)
    }
    await settle()
    expect(gains()).toEqual([0.5])
  })

  it('debounces each stem on its own', async () => {
    const quiet = track('s2', { gain: 0.3 })
    const sync = new StemSync(target, [track('s1'), track('s2')], () => {})
    sync.update([track('s1', { gain: 0.1 }), quiet])
    for (let i = 2; i <= 10; i++) {
      await vi.advanceTimersByTimeAsync(100)
      sync.update([track('s1', { gain: i / 10 }), quiet])
    }
    expect(calls().map((c) => c.url)).toEqual(['http://api.test:8000/api/projects/p1/stems/s2'])
  })

  it('skips the write when undo lands back on what the backend holds', async () => {
    const start = track('s1')
    const sync = new StemSync(target, [start], () => {})
    sync.update([track('s1', { gain: 0.5 })])
    sync.update([start])
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('last write wins', () => {
  it('keeps one request in flight per stem and then sends only the newest state', async () => {
    let release = () => {}
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => (release = () => resolve(ok()))))
    const sync = new StemSync(target, [track('s1')], () => {})

    sync.update([track('s1', { gain: 0.1 })])
    await settle()
    sync.update([track('s1', { gain: 0.2 })])
    await settle()
    sync.update([track('s1', { gain: 0.3 })])
    await settle()
    expect(gains()).toEqual([0.1])

    release()
    await settle()
    expect(gains()).toEqual([0.1, 0.3])
  })

  it('undoing back to the original value during a write still reaches the backend', async () => {
    let release = () => {}
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => (release = () => resolve(ok()))))
    const start = track('s1')
    const sync = new StemSync(target, [start], () => {})

    sync.update([track('s1', { gain: 0.1 })])
    await settle()
    sync.update([start])
    await settle()
    release()
    await settle()
    expect(gains()).toEqual([0.1, 1])
  })
})

describe('failure', () => {
  it('reports the error and leaves local tracks untouched', async () => {
    fetchMock.mockImplementationOnce(async () => fail(401, 'Invalid project token'))
    const errors: string[] = []
    const edited = [track('s1', { gain: 0.5 })]
    const before = structuredClone(edited)
    const sync = new StemSync(target, [track('s1')], (e) => errors.push(e))
    sync.update(edited)
    await settle()
    expect(errors).toEqual(['Sync failed: Backend returned 401: Invalid project token'])
    expect(edited).toEqual(before)
  })

  it('reports a network error', async () => {
    fetchMock.mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')))
    const errors: string[] = []
    const sync = new StemSync(target, [track('s1')], (e) => errors.push(e))
    sync.update([track('s1', { gain: 0.5 })])
    await settle()
    expect(errors).toEqual(['Sync failed: Failed to fetch'])
  })

  it('does not retry on its own', async () => {
    fetchMock.mockImplementation(async () => fail(500, 'boom'))
    const sync = new StemSync(target, [track('s1')], () => {})
    sync.update([track('s1', { gain: 0.5 })])
    await vi.advanceTimersByTimeAsync(SYNC_DEBOUNCE_MS * 20)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('retries on the next change and clears the error', async () => {
    fetchMock.mockImplementationOnce(async () => fail(500, 'boom'))
    const errors: string[] = []
    const sync = new StemSync(target, [track('s1')], (e) => errors.push(e))
    sync.update([track('s1', { gain: 0.5 })])
    await settle()
    sync.update([track('s1', { gain: 0.6 })])
    await settle()
    expect(gains()).toEqual([0.5, 0.6])
    expect(errors.at(-1)).toBe('')
  })

  it('retries a failed stem when a different stem changes', async () => {
    fetchMock.mockImplementationOnce(async () => fail(500, 'boom'))
    const failed = track('s1', { gain: 0.5 })
    const sync = new StemSync(target, [track('s1'), track('s2')], () => {})
    sync.update([failed, track('s2')])
    await settle()
    sync.update([failed, track('s2', { gain: 0.25 })])
    await settle()
    expect(gains().sort()).toEqual([0.25, 0.5, 0.5])
  })

  it('sends the state again after undo, since a failed write may still have applied', async () => {
    fetchMock.mockImplementationOnce(() => Promise.reject(new Error('signal timed out')))
    const errors: string[] = []
    const start = track('s1')
    const sync = new StemSync(target, [start], (e) => errors.push(e))
    sync.update([track('s1', { gain: 0.5 })])
    await settle()
    sync.update([start])
    await settle()
    expect(gains()).toEqual([0.5, 1])
    expect(errors.at(-1)).toBe('')
  })
})

describe('close', () => {
  it('sends what the debounce was holding', async () => {
    const sync = new StemSync(target, [track('s1')], () => {})
    sync.update([track('s1', { gain: 0.5 })])
    sync.close()
    await vi.advanceTimersByTimeAsync(0)
    expect(gains()).toEqual([0.5])
  })

  it('still reports a failed final write', async () => {
    fetchMock.mockImplementationOnce(async () => fail(500, 'boom'))
    const errors: string[] = []
    const sync = new StemSync(target, [track('s1')], (e) => errors.push(e))
    sync.update([track('s1', { gain: 0.5 })])
    sync.close()
    await vi.advanceTimersByTimeAsync(0)
    expect(errors.at(-1)).toBe('Sync failed: Backend returned 500: boom')
  })

  it('sends nothing when already synced, and clears an old error', async () => {
    const errors: string[] = []
    const sync = new StemSync(target, [track('s1')], (e) => errors.push(e))
    sync.close()
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(errors).toEqual([''])
  })
})

describe('replacing the stems', () => {
  const other = { config: target.config, projectId: 'p2' }
  const runner = (errors: string[] = [], late: string[] = []) =>
    new StemSyncRunner(
      (e) => errors.push(e),
      (e) => late.push(e),
    )

  it('never sends new stems to the old project', async () => {
    const r = runner()
    r.apply(target, [track('a')])
    r.apply(other, [track('b', { gain: 0.5 })])
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('clears the live error, so it does not outlive its stems', async () => {
    fetchMock.mockImplementationOnce(async () => fail(500, 'boom'))
    const errors: string[] = []
    const r = runner(errors)
    r.apply(target, [track('a')])
    r.apply(target, [track('a', { gain: 0.5 })])
    await settle()
    expect(errors.at(-1)).toContain('boom')
    fetchMock.mockImplementation(async () => ok())
    r.apply(null, [track('f')])
    expect(errors.at(-1)).toBe('')
  })

  it('reports a failed final write separately, not as a live error', async () => {
    fetchMock.mockImplementationOnce(async () => fail(500, 'boom'))
    const errors: string[] = []
    const late: string[] = []
    const r = runner(errors, late)
    r.apply(target, [track('a')])
    r.apply(target, [track('a', { gain: 0.5 })])
    r.apply(null, [track('f')])
    await vi.advanceTimersByTimeAsync(0)
    expect(errors.filter(Boolean)).toEqual([])
    expect(late).toEqual(['Previous stems: Sync failed: Backend returned 500: boom'])
  })

  it('does not let the old sync clear the new sync error', async () => {
    let release = () => {}
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => (release = () => resolve(ok()))))
    fetchMock.mockImplementationOnce(async () => fail(500, 'boom'))
    const errors: string[] = []
    const r = runner(errors)
    r.apply(target, [track('a')])
    r.apply(target, [track('a', { gain: 0.5 })])
    r.apply(other, [track('b')])
    await vi.advanceTimersByTimeAsync(0)
    release()
    r.apply(other, [track('b', { gain: 0.25 })])
    await settle()
    expect(errors.at(-1)).toContain('boom')
  })

  it('holds the new sync until the old one has finished, so a reload cannot overlap writes', async () => {
    let release = () => {}
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => (release = () => resolve(ok()))))
    const r = runner()
    r.apply(target, [track('a')])
    r.apply(target, [track('a', { gain: 0.3 })])
    const reload = { ...target }
    r.apply(reload, [track('a')])
    r.apply(reload, [track('a', { gain: 0.9 })])
    await settle()
    expect(gains()).toEqual([0.3])
    release()
    await settle()
    expect(gains()).toEqual([0.3, 0.9])
  })

  it('sends nothing for mock or file stems', async () => {
    const r = runner()
    r.apply(null, [track('m')])
    r.apply(null, [track('m', { gain: 0.5 })])
    await settle()
    r.close()
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('only job stems sync', () => {
  const edit = (target: Parameters<typeof connectStemSync>[0]) => {
    const sync = connectStemSync(target, [track('s1')], () => {})
    sync?.update([track('s1', { gain: 0.5 })])
    return sync
  }

  it('makes no calls for mock stems', async () => {
    const source: StemSource = new MockStemSource()
    await source.load()
    expect(edit(source.syncTarget)).toBeNull()
    await settle()
    source.dispose?.()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('makes no calls for file stems, even with a backend configured', async () => {
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} }))
    const source: StemSource = new FileStemSource([new File(['x'], 'a.wav')])
    await source.load()
    expect(edit(source.syncTarget)).toBeNull()
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('makes no calls when no job is loaded', async () => {
    expect(edit(new HttpStemSource(target.config, 'j1').syncTarget)).toBeNull()
    expect(edit(null)).toBeNull()
    await settle()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('syncs to the project named by the loaded job', async () => {
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} }))
    const row = { id: 's1', name: 'dog', duration: 2, audio_url: '/a' }
    const saved = { ...row, volume: 1, muted: false, solo: false, automation: [], start_offset: 0, mute_ranges: [] }
    fetchMock.mockImplementation(async (input) => {
      if (String(input).endsWith('/api/jobs/j1')) {
        return new Response(JSON.stringify({ status: 'completed', project_id: 'p9', stems: [row] }))
      }
      return String(input).endsWith('/api/projects/p9/stems') ? new Response(JSON.stringify([saved])) : ok()
    })
    const source = new HttpStemSource(target.config, 'j1')
    await source.load()
    expect(source.syncTarget).toEqual({ config: target.config, projectId: 'p9' })
    expect(source.syncIssue).toBeUndefined()
    fetchMock.mockClear()
    edit(source.syncTarget)
    await settle()
    expect(calls()[0].url).toBe('http://api.test:8000/api/projects/p9/stems/s1')
  })

  it('says so when a job does not name its project', async () => {
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} }))
    fetchMock.mockImplementation(async (input) =>
      String(input).endsWith('/api/jobs/j1')
        ? new Response(
            JSON.stringify({ status: 'completed', stems: [{ id: 's1', name: 'dog', duration: 2, audio_url: '/a' }] }),
          )
        : ok(),
    )
    const source = new HttpStemSource(target.config, 'j1')
    await source.load()
    expect(source.syncTarget).toBeUndefined()
    expect(source.syncIssue).toContain('project_id')
  })
})
