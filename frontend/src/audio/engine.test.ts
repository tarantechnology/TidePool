import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Stem, TrackState } from '../types'
import { AudioEngine } from './engine'
import type { EngineEvent } from './engine'

// Just enough of Web Audio to drive the engine's bookkeeping in Node. Sound
// itself cannot be observed here.
class FakeParam {
  value = 1
  cancelScheduledValues() {}
  setTargetAtTime() {}
  setValueAtTime() {}
  linearRampToValueAtTime() {}
}

class FakeNode {
  gain = new FakeParam()
  disconnected = false
  connect() {}
  disconnect() {
    this.disconnected = true
  }
}

class FakeSource extends FakeNode {
  buffer: unknown = null
  onended: (() => void) | null = null
  started = false
  stopped = false
  start() {
    this.started = true
  }
  stop() {
    this.stopped = true
  }
}

const made = {
  gains: [] as FakeNode[],
  sources: [] as FakeSource[],
  contexts: 0,
  resume: (): Promise<void> => Promise.resolve(),
}

class FakeContext {
  currentTime = 0
  destination = new FakeNode()
  constructor() {
    made.contexts++
  }
  createGain() {
    const g = new FakeNode()
    made.gains.push(g)
    return g
  }
  createBufferSource() {
    const s = new FakeSource()
    made.sources.push(s)
    return s
  }
  async decodeAudioData(bytes: ArrayBuffer) {
    const seconds = new Float32Array(bytes)[0]
    if (seconds === 0) throw new Error('decode failed')
    const data = new Float32Array(seconds * 100)
    return { duration: seconds, sampleRate: 100, length: data.length, getChannelData: () => data }
  }
  resume = () => made.resume()
  async close() {}
}

/** The URL encodes the fake duration, which the fake decoder reads back. */
const stem = (id: string, seconds: number): Stem => ({
  id,
  label: id,
  kind: 'sfx',
  url: `http://fake/${seconds}`,
  duration: 0,
})

const track = (stemId: string): TrackState => ({ stemId, gain: 1, muted: false, solo: false, automation: [] })

describe('AudioEngine', () => {
  let engine: AudioEngine
  let events: EngineEvent[]

  beforeEach(async () => {
    made.gains = []
    made.sources = []
    made.contexts = 0
    made.resume = () => Promise.resolve()
    vi.stubGlobal('AudioContext', FakeContext)
    vi.stubGlobal('fetch', async (url: string) => {
      const seconds = Number(url.split('/').pop())
      if (seconds < 0) return new Response('', { status: 404 })
      return new Response(new Float32Array([seconds]).buffer)
    })
    engine = new AudioEngine()
    events = []
    engine.subscribe((e) => events.push(e))
    await engine.replaceStems([stem('a', 10), stem('b', 20)])
    events.length = 0
  })
  afterEach(() => vi.unstubAllGlobals())

  it('reports the measured duration of each stem', async () => {
    const stems = await engine.replaceStems([stem('a', 7)])
    expect(stems?.[0].duration).toBe(7)
    expect(engine.duration).toBe(7)
  })

  it('exposes samples and peaks per stem, and null for unknown ones', () => {
    expect(engine.getSamples('a')?.sampleRate).toBe(100)
    expect(engine.getSamples('a')?.data).toHaveLength(1000)
    expect(engine.getPeaks('a')).not.toBeNull()
    expect(engine.getSamples('nope')).toBeNull()
    expect(engine.getPeaks('nope')).toBeNull()
  })

  it('notifies every subscriber, and stops after unsubscribe', async () => {
    const second: EngineEvent[] = []
    const off = engine.subscribe((e) => second.push(e))
    await engine.play()
    off()
    engine.pause()
    expect(events.map((e) => e.type)).toEqual(['play', 'pause'])
    expect(second.map((e) => e.type)).toEqual(['play'])
  })

  it('emits seek with the target time, clamped to the duration', () => {
    engine.seek(5)
    engine.seek(99)
    expect(events).toEqual([
      { type: 'seek', time: 5 },
      { type: 'seek', time: 20 },
    ])
  })

  it('replaces stems: old graph disconnected, position reset, same context', async () => {
    engine.seek(5)
    await engine.play()
    const oldGains = [...made.gains]
    await engine.replaceStems([stem('c', 3)])

    expect(made.contexts).toBe(1)
    expect(engine.isPlaying()).toBe(false)
    expect(engine.getTime()).toBe(0)
    expect(engine.duration).toBe(3)
    expect(engine.getPeaks('a')).toBeNull()
    expect(engine.getPeaks('c')).not.toBeNull()
    expect(made.sources.every((s) => s.stopped)).toBe(true)
    expect(oldGains.some((g) => g.disconnected)).toBe(true)
    expect(events.at(-1)?.type).toBe('load')
  })

  it('keeps the current stems if the new ones fail to load', async () => {
    engine.seek(5)
    await expect(engine.replaceStems([stem('c', -1)])).rejects.toThrow('404')
    await expect(engine.replaceStems([stem('c', 0)])).rejects.toThrow('decode failed')
    expect(engine.getPeaks('a')).not.toBeNull()
    expect(engine.getTime()).toBe(5)
    expect(engine.duration).toBe(20)
  })

  it('drops a decode in progress once a newer load has started', async () => {
    await engine.replaceStems([stem('a', 10)])
    const superseded = engine.replaceStems([stem('x', 1)])
    engine.invalidateLoads()
    expect(await superseded).toBeNull()
    expect(engine.duration).toBe(10)
  })

  it('lets the newest of two overlapping replacements win', async () => {
    const [first, second] = await Promise.all([
      engine.replaceStems([stem('x', 1)]),
      engine.replaceStems([stem('y', 2)]),
    ])
    expect(first).toBeNull()
    expect(second?.[0].id).toBe('y')
    expect(engine.getPeaks('x')).toBeNull()
    expect(engine.duration).toBe(2)
  })

  it('returns null, not an error, for a replaced call that fails', async () => {
    const [first, second] = await Promise.all([
      engine.replaceStems([stem('x', -1)]),
      engine.replaceStems([stem('y', 2)]),
    ])
    expect(first).toBeNull()
    expect(second?.[0].id).toBe('y')
  })

  it('starts one set of sources when play is called twice during resume', async () => {
    let release = () => {}
    const resumed = new Promise<void>((r) => (release = r))
    made.resume = () => resumed
    const a = engine.play()
    const b = engine.play()
    release()
    await Promise.all([a, b])
    expect(made.sources.filter((s) => s.started)).toHaveLength(2)
    engine.pause()
    expect(made.sources.every((s) => s.stopped)).toBe(true)
  })

  it('honours a pause that arrives during resume', async () => {
    let release = () => {}
    made.resume = () => new Promise((r) => (release = r))
    const started = engine.play()
    engine.pause()
    release()
    await started
    expect(engine.isPlaying()).toBe(false)
    expect(made.sources).toHaveLength(0)
  })

  it('accepts tracks that reference no loaded stem', async () => {
    await engine.replaceStems([stem('c', 3)])
    expect(() => engine.setTracks([track('gone')])).not.toThrow()
  })
})
