import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Stem, TrackState } from '../types'
import { AudioEngine, describeStemFailures, sourceStart } from './engine'
import type { EngineEvent } from './engine'

// Just enough of Web Audio to drive the engine's bookkeeping in Node. Sound
// itself cannot be observed here.
class FakeParam {
  value = 1
  events: Array<[string, number, number]> = []
  cancelScheduledValues() {
    this.events = []
  }
  setTargetAtTime(value: number, time: number) {
    this.events.push(['target', value, time])
  }
  setValueCurveAtTime(values: Float32Array, time: number, duration: number) {
    this.curves.push({ values: Array.from(values), time, duration })
  }
  curves: Array<{ values: number[]; time: number; duration: number }> = []
  setValueAtTime(value: number, time: number) {
    this.events.push(['set', value, time])
  }
  linearRampToValueAtTime(value: number, time: number) {
    this.events.push(['ramp', value, time])
  }
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
  startArgs: number[] = []
  start(...args: number[]) {
    this.started = true
    this.startArgs = args
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
    const data = new Float32Array(seconds * 100).fill(0.1)
    return { duration: seconds, sampleRate: 100, numberOfChannels: 1, length: data.length, getChannelData: () => data }
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

describe('sourceStart', () => {
  it('waits for a position before the offset', () => {
    expect(sourceStart(0, 2, 5)).toEqual({ delay: 2, bufferOffset: 0 })
    expect(sourceStart(1.5, 2, 5)).toEqual({ delay: 0.5, bufferOffset: 0 })
  })

  it('starts at once exactly at the offset', () => {
    expect(sourceStart(2, 2, 5)).toEqual({ delay: 0, bufferOffset: 0 })
  })

  it('starts inside the buffer past the offset', () => {
    expect(sourceStart(3.5, 2, 5)).toEqual({ delay: 0, bufferOffset: 1.5 })
  })

  it('has nothing to start at or past the end', () => {
    expect(sourceStart(7, 2, 5)).toBeNull()
    expect(sourceStart(9, 2, 5)).toBeNull()
  })

  it('matches the old behaviour with no offset', () => {
    expect(sourceStart(4, 0, 5)).toEqual({ delay: 0, bufferOffset: 4 })
  })
})

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

  it('plays the edited clips automatically, including repeats and restart', async () => {
    vi.useFakeTimers()
    try {
      engine.setSequence([{start:0,end:3},{start:7,end:10},{start:0,end:2}])
      expect(engine.timelineDuration).toBe(8)
      await engine.play()
      expect(made.sources.at(-1)?.startArgs).toEqual([0,0,3])
      await vi.advanceTimersByTimeAsync(3000)
      expect(engine.getTime()).toBe(7)
      expect(engine.getTimelineTime()).toBe(3)
      expect(made.sources.at(-1)?.startArgs).toEqual([0,7,3])
      await vi.advanceTimersByTimeAsync(3000)
      expect(engine.getTime()).toBe(0)
      expect(engine.getTimelineTime()).toBe(6)
      await vi.advanceTimersByTimeAsync(2000)
      expect(engine.isPlaying()).toBe(false)
      expect(engine.getTimelineTime()).toBe(8)
      await engine.play()
      expect(engine.getTime()).toBe(0)
      engine.stop()
      expect(engine.getTimelineTime()).toBe(0)
    } finally { engine.dispose(); vi.useRealTimers() }
  })
  it('restarts a finished sequence even when clip lengths do not sum exactly', async () => {
    vi.useFakeTimers()
    try {
      // These lengths add up to 3.788 one way and 3.7880000000000003 the other.
      engine.setSequence([{start:12.464,end:13.102},{start:0.802,end:3.421},{start:10.067,end:10.598}])
      await engine.play()
      await vi.advanceTimersByTimeAsync(5000)
      expect(engine.isPlaying()).toBe(false)
      await engine.play()
      expect(engine.isPlaying()).toBe(true)
      expect(engine.getTimelineTime()).toBe(0)
    } finally { engine.dispose(); vi.useRealTimers() }
  })
  it('seeks in edited time to the right repeated source occurrence',()=>{
    engine.setSequence([{start:0,end:3},{start:7,end:10},{start:0,end:2}])
    engine.seekTimeline(6.5)
    expect(engine.getTime()).toBe(.5)
    expect(engine.getTimelineTime()).toBe(6.5)
    engine.dispose()
  })
  it('reports the measured duration of each stem', async () => {
    const result = await engine.replaceStems([stem('a', 7)])
    expect(result?.stems[0].duration).toBe(7)
    expect(result?.failures).toEqual([])
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

  it('keeps the current stems when none of the new ones load', async () => {
    engine.seek(5)
    await expect(engine.replaceStems([stem('c', -1)])).rejects.toThrow('404')
    await expect(engine.replaceStems([stem('c', 0)])).rejects.toThrow('decode failed')
    await expect(engine.replaceStems([stem('c', 0), stem('d', -1)])).rejects.toThrow(
      /Could not load any stem: c \(decode failed\), d \(.*404\)/,
    )
    expect(engine.getPeaks('a')).not.toBeNull()
    expect(engine.getTime()).toBe(5)
    expect(engine.duration).toBe(20)
  })

  it('loads the stems that decode and reports the ones that do not', async () => {
    const result = await engine.replaceStems([stem('good', 6), stem('broken', 0), stem('missing', -1)])

    expect(result?.stems.map((s) => s.id)).toEqual(['good'])
    expect(engine.getPeaks('good')).not.toBeNull()
    expect(engine.getPeaks('broken')).toBeNull()
    expect(engine.duration).toBe(6)
    expect(result?.failures.map((f) => f.stem.id)).toEqual(['broken', 'missing'])
    expect(result?.failures[0].reason).toBe('decode failed')
    expect(describeStemFailures(result?.failures ?? [])).toBe(
      'broken (decode failed), missing (Failed to fetch stem missing: 404)',
    )
    expect(events.at(-1)?.type).toBe('load')
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
    expect(second?.stems[0].id).toBe('y')
    expect(engine.getPeaks('x')).toBeNull()
    expect(engine.duration).toBe(2)
  })

  it('returns null, not an error, for a replaced call that fails', async () => {
    const [first, second] = await Promise.all([
      engine.replaceStems([stem('x', -1)]),
      engine.replaceStems([stem('y', 2)]),
    ])
    expect(first).toBeNull()
    expect(second?.stems[0].id).toBe('y')
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

  describe('with offsets and mute ranges', () => {
    const placed = (id: string, seconds: number, over: Partial<Stem>): Stem => ({ ...stem(id, seconds), ...over })
    const sources = () => made.sources.filter((s) => s.started && !s.stopped)
    const envelopeEvents = (stemIndex: number) =>
      // Gains in creation order after master: envelope, fader, gate per stem.
      made.gains.slice(-8)[stemIndex * 4].gain.events.filter((e) => e[0] !== 'target')

    beforeEach(async () => {
      await engine.replaceStems([placed('a', 5, { startOffset: 2 }), placed('b', 3, {})])
      made.sources = []
    })

    it('takes duration from the latest end on the timeline', () => {
      expect(engine.duration).toBe(7)
    })

    it('schedules a later stem and starts the other at once', async () => {
      await engine.play()
      const [a, b] = sources()
      expect(a.startArgs).toEqual([2, 0])
      expect(b.startArgs).toEqual([0, 0])
    })

    it('starts inside the buffer after a seek past the offset', async () => {
      await engine.play()
      made.sources = []
      engine.seek(4)
      const [a] = sources()
      expect(a.startArgs).toEqual([0, 2])
      expect(sources()).toHaveLength(1)
    })

    it('resumes correctly from pause', async () => {
      engine.seek(1)
      await engine.play()
      expect(sources().map((s) => s.startArgs)).toEqual([
        [1, 0],
        [0, 1],
      ])
      engine.pause()
      expect(engine.getTime()).toBe(1)
    })

    it('stops at the end when a seek lands past every stem', async () => {
      await engine.play()
      engine.seek(7)
      expect(engine.isPlaying()).toBe(false)
      expect(engine.getTime()).toBe(7)
    })

    it('does not end when only the shorter stem finishes', async () => {
      await engine.play()
      const [, b] = sources()
      b.onended?.()
      expect(engine.isPlaying()).toBe(true)
    })

    it('schedules mute ranges on top of automation', async () => {
      await engine.replaceStems([placed('a', 10, { muteRanges: [[2, 3]] }), placed('b', 10, {})])
      engine.setTracks([track('a'), track('b')])
      await engine.play()
      expect(envelopeEvents(0)).toEqual([
        ['set', 1, 0],
        ['ramp', 1, 2],
        ['ramp', 0, 2],
        ['ramp', 0, 3],
        ['ramp', 1, 3],
      ])
      expect(envelopeEvents(1)).toEqual([['set', 1, 0]])
    })

    it('reads silent when paused inside a mute range', async () => {
      await engine.replaceStems([placed('a', 10, { muteRanges: [[2, 3]] })])
      engine.setTracks([track('a')])
      engine.seek(2.5)
      expect(made.gains.slice(-4)[0].gain.events.filter((e) => e[0] !== 'target')).toEqual([['set', 0, 0]])
    })

    it('gives analysis the offset', () => {
      expect(engine.getSamples('a')?.startOffset).toBe(2)
    })
  })

  describe('making room for a boosted layer', () => {
    // Per stem the engine makes envelope, fader, room, gate, in that order. Every fake stem is a steady tone.
    const room = (stemIndex: number, count: number) => made.gains.slice(-4 * count)[stemIndex * 4 + 2].gain
    const lastTarget = (p: FakeParam) => p.events.filter((e) => e[0] === 'target').at(-1)?.[1]

    it('turns the other layers down by the boosted layer\'s level while it sounds', async () => {
      await engine.replaceStems([stem('a', 4), stem('b', 4), stem('c', 4)])
      engine.seek(1)
      engine.setTracks([{ ...track('a'), gain: 4 }, track('b'), track('c')])
      expect(lastTarget(room(0, 3))).toBe(1)
      expect(lastTarget(room(1, 3))).toBeCloseTo(0.25)
      expect(lastTarget(room(2, 3))).toBeCloseTo(0.25)
    })

    it('leaves the mix alone at or below 100%, and ignores a muted boost', async () => {
      await engine.replaceStems([stem('a', 4), stem('b', 4)])
      engine.seek(1)
      engine.setTracks([{ ...track('a'), gain: 1 }, track('b')])
      expect(lastTarget(room(1, 2))).toBe(1)
      engine.setTracks([{ ...track('a'), gain: 5, muted: true }, track('b')])
      expect(lastTarget(room(1, 2))).toBe(1)
    })

    it('follows a boosted layer\'s volume curve while playing', async () => {
      await engine.replaceStems([stem('a', 4), stem('b', 4)])
      engine.setTracks([{ ...track('a'), automation: [{ t: 0, value: 1 }, { t: 2, value: 4 }] }, track('b')])
      await engine.play()
      const curve = room(1, 2).curves.at(-1)!
      expect(curve.duration).toBeCloseTo(4)
      expect(curve.values[10]).toBeCloseTo(1 / (1 + 3 * 0.05), 2)
      expect(curve.values[200]).toBeCloseTo(0.25)
      expect(curve.values.at(-2)).toBeCloseTo(0.25)
      expect(curve.values.at(-1)).toBe(1) // the booster has ended
    })
  })

  it('accepts tracks that reference no loaded stem', async () => {
    await engine.replaceStems([stem('c', 3)])
    expect(() => engine.setTracks([track('gone')])).not.toThrow()
  })
})
