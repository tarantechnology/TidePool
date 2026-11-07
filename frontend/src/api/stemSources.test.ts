import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { TrackState } from '../types'
import { getJob, patchStem, trackToStemPatch } from './backend'
import {
  FileStemSource,
  HttpStemSource,
  isAudioFile,
  isVideoFile,
  labelFromFilename,
  stemIdFromFilename,
} from './stemSources'

const config = { baseUrl: 'http://api.test:8000', token: 'tok' }
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

describe('labelFromFilename', () => {
  it.each([
    ['dog_bark-01.wav', 'dog bark 01'],
    ['Siren.MP3', 'Siren'],
    ['a.b.c.flac', 'a.b.c'],
    ['noext', 'noext'],
    ['___.wav', '___.wav'],
  ])('%s becomes %s', (name, label) => {
    expect(labelFromFilename(name)).toBe(label)
  })
})

describe('stemIdFromFilename', () => {
  it('stays unique for repeated names', () => {
    expect(stemIdFromFilename('take.wav', 0)).not.toBe(stemIdFromFilename('take.wav', 1))
  })

  it('slugs and survives names with no usable characters', () => {
    expect(stemIdFromFilename('Dog Bark!.wav', 2)).toBe('file-2-dog-bark')
    expect(stemIdFromFilename('???.wav', 0)).toBe('file-0-audio')
  })
})

describe('isAudioFile', () => {
  it('accepts by MIME type or extension', () => {
    expect(isAudioFile({ name: 'x', type: 'audio/wav' })).toBe(true)
    expect(isAudioFile({ name: 'x.FLAC', type: '' })).toBe(true)
  })

  it('rejects other files', () => {
    expect(isAudioFile({ name: 'x.mp4', type: 'video/mp4' })).toBe(false)
    expect(isAudioFile({ name: 'notes.txt', type: 'text/plain' })).toBe(false)
  })
})

describe('isVideoFile', () => {
  it('accepts by MIME type or extension', () => {
    expect(isVideoFile({ name: 'x', type: 'video/quicktime' })).toBe(true)
    expect(isVideoFile({ name: 'x.MOV', type: '' })).toBe(true)
  })

  it('rejects audio', () => {
    expect(isVideoFile({ name: 'x.wav', type: 'audio/wav' })).toBe(false)
  })
})

describe('FileStemSource', () => {
  it('makes blob URLs, defaults kind to sfx and revokes on dispose', async () => {
    const source = new FileStemSource([new File(['x'], 'dog_bark.wav'), new File(['y'], 'dialogue.wav')])
    const stems = await source.load()
    expect(stems.map((s) => s.label)).toEqual(['dog bark', 'dialogue'])
    expect(stems.every((s) => s.kind === 'sfx')).toBe(true)
    expect(stems[0].url).toMatch(/^blob:/)
    source.dispose()
    await expect(fetch(stems[0].url)).rejects.toThrow()
  })

  it('uses an explicit label', async () => {
    const stems = await new FileStemSource([new File(['x'], 'clip.mp4')], 'Original audio').load()
    expect(stems[0].label).toBe('Original audio')
  })
})

describe('backend client', () => {
  const fetchMock = vi.fn<typeof fetch>()
  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  const authOf = (call: number) => new Headers(fetchMock.mock.calls[call][1]?.headers).get('Authorization')

  it('HttpStemSource reads the job, then fetches each stem with the bearer token', async () => {
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith('/api/jobs/j1')) {
        return json({
          status: 'completed',
          stems: [{ id: 's1', name: 'dog', duration: 4.5, audio_url: '/api/projects/p/stems/s1/audio' }],
        })
      }
      return new Response('wav')
    })
    const stems = await new HttpStemSource(config, 'j1').load()
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://api.test:8000/api/jobs/j1')
    expect(String(fetchMock.mock.calls[1][0])).toBe('http://api.test:8000/api/projects/p/stems/s1/audio')
    expect(authOf(0)).toBe('Bearer tok')
    expect(authOf(1)).toBe('Bearer tok')
    expect(stems[0]).toMatchObject({ id: 's1', label: 'dog', kind: 'sfx', duration: 4.5 })
    expect(stems[0].url).toMatch(/^blob:/)
  })

  it('HttpStemSource fails clearly on a job with no stems', async () => {
    fetchMock.mockResolvedValue(json({ status: 'processing', stems: [] }))
    await expect(new HttpStemSource(config, 'j1').load()).rejects.toThrow('no stems')
  })

  it('revokes a stem that finishes downloading after dispose', async () => {
    let finish: (res: Response) => void = () => {}
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith('/api/jobs/j1')) {
        return json({
          stems: [
            { id: 'a', name: 'a', duration: 1, audio_url: '/fail' },
            { id: 'b', name: 'b', duration: 1, audio_url: '/slow' },
          ],
        })
      }
      if (String(url).endsWith('/fail')) return new Response('', { status: 500 })
      return new Promise<Response>((r) => (finish = r))
    })
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    const source = new HttpStemSource(config, 'j1')
    await expect(source.load()).rejects.toThrow('500')
    source.dispose()
    finish(new Response('wav'))
    await vi.waitFor(() => expect(revoke).toHaveBeenCalledTimes(1))
    revoke.mockRestore()
  })

  it('gives every request a timeout signal', async () => {
    fetchMock.mockResolvedValue(json({ status: 'queued' }))
    await getJob(config, 'j1')
    expect(fetchMock.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal)
  })

  it('forwards a caller abort to the request', async () => {
    const ctrl = new AbortController()
    fetchMock.mockResolvedValue(json({ status: 'queued' }))
    await getJob(config, 'j1', ctrl.signal)
    ctrl.abort()
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
  })

  it('surfaces FastAPI error details', async () => {
    fetchMock.mockResolvedValue(json({ detail: 'Invalid project token' }, 401))
    await expect(getJob(config, 'j1')).rejects.toThrow('Backend returned 401: Invalid project token')
  })

  it('reports the status for validation arrays and non-JSON errors', async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: [{ msg: 'bad' }] }, 422))
    await expect(getJob(config, 'j1')).rejects.toThrow(/^Backend returned 422$/)
    fetchMock.mockResolvedValueOnce(new Response('boom', { status: 500 }))
    await expect(getJob(config, 'j1')).rejects.toThrow(/^Backend returned 500$/)
  })

  it('escapes ids in paths', async () => {
    fetchMock.mockResolvedValue(json({ status: 'queued' }))
    await getJob(config, 'a/b')
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/jobs/a%2Fb')
  })

  it('accepts a same-origin base URL, as used with the dev proxy', async () => {
    vi.stubGlobal('location', { href: 'http://localhost:5173/#/app' })
    fetchMock.mockResolvedValue(json({ status: 'queued' }))
    await getJob({ baseUrl: '/api', token: 'tok' }, 'j1')
    expect(String(fetchMock.mock.calls[0][0])).toBe('http://localhost:5173/api/jobs/j1')
  })

  it('patches a stem with the backend field names', async () => {
    fetchMock.mockResolvedValue(json({}))
    const track: TrackState = { stemId: 's1', gain: 0.5, muted: true, solo: false, automation: [{ t: 1, value: 2 }] }
    await patchStem(config, 'p1', track)
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toBe('http://api.test:8000/api/projects/p1/stems/s1')
    expect(init?.method).toBe('PATCH')
    expect(JSON.parse(init?.body as string)).toEqual({
      volume: 0.5,
      muted: true,
      solo: false,
      automation: [{ time: 1, value: 2 }],
    })
    expect(new Headers(init?.headers).get('Content-Type')).toBe('application/json')
    expect(authOf(0)).toBe('Bearer tok')
  })
})

describe('trackToStemPatch', () => {
  it('collapses equal-time keyframes, since the backend rejects them', () => {
    const patch = trackToStemPatch({
      stemId: 's',
      gain: 1,
      muted: false,
      solo: false,
      automation: [
        { t: 1, value: 1 },
        { t: 5, value: 1 },
        { t: 5, value: 0 },
        { t: 9, value: 0 },
      ],
    })
    expect(patch.automation).toEqual([
      { time: 1, value: 1 },
      { time: 5, value: 0 },
      { time: 9, value: 0 },
    ])
  })
})
