import { automationFrom, isAudible } from '../edits'
import type { Stem, TrackState } from '../types'

// Per-stem graph:  source -> envelope (automation) -> fader (volume) -> gate (mute/solo) -> master
// The timeline clock is derived from AudioContext time, so playback position
// and gain automation are scheduled against the same clock.

const PEAK_BINS = 1200
const SMOOTH = 0.01

interface Channel {
  stem: Stem
  buffer: AudioBuffer
  peaks: Float32Array
  envelope: GainNode
  fader: GainNode
  gate: GainNode
  source: AudioBufferSourceNode | null
}

export interface EngineEvent {
  type: 'play' | 'pause' | 'seek' | 'load'
  /** Timeline position when the event fired. */
  time: number
}

export type EngineListener = (event: EngineEvent) => void

export class AudioEngine {
  private ctx: AudioContext
  private master: GainNode
  private channels = new Map<string, Channel>()
  private tracks: TrackState[] = []
  private playing = false
  private startedAt = 0
  private pausedAt = 0
  private listeners = new Set<EngineListener>()
  private loadId = 0
  /** Bumped by anything that cancels a play() still waiting on ctx.resume(). */
  private playRequest = 0
  duration = 0

  constructor() {
    this.ctx = new AudioContext()
    this.master = this.ctx.createGain()
    this.master.gain.value = 0.7
    this.master.connect(this.ctx.destination)
  }

  subscribe(listener: EngineListener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /** A newer load has started, so any decode still running must not swap the stems in. */
  invalidateLoads(): void {
    this.loadId++
  }

  /**
   * Swap the loaded stems, keeping the AudioContext. Decoding happens first, so
   * a failure leaves the current stems playing. Returns the stems with their
   * measured duration, or null if a newer call replaced this one.
   */
  async replaceStems(stems: Stem[]): Promise<Stem[] | null> {
    const id = ++this.loadId
    let loaded: Array<{ stem: Stem; buffer: AudioBuffer }>
    try {
      loaded = await Promise.all(
        stems.map(async (stem) => {
          const res = await fetch(stem.url)
          if (!res.ok) throw new Error(`Failed to fetch stem ${stem.id}: ${res.status}`)
          const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer())
          return { stem: { ...stem, duration: buffer.duration }, buffer }
        }),
      )
    } catch (err) {
      if (id !== this.loadId) return null
      throw err
    }
    if (id !== this.loadId) return null

    this.playRequest++
    this.stopSources()
    this.playing = false
    for (const ch of this.channels.values()) ch.gate.disconnect()
    this.channels.clear()
    this.tracks = []
    this.pausedAt = 0
    this.duration = 0

    for (const { stem, buffer } of loaded) {
      const envelope = this.ctx.createGain()
      const fader = this.ctx.createGain()
      const gate = this.ctx.createGain()
      envelope.connect(fader)
      fader.connect(gate)
      gate.connect(this.master)
      this.channels.set(stem.id, {
        stem,
        buffer,
        peaks: computePeaks(buffer),
        envelope,
        fader,
        gate,
        source: null,
      })
      this.duration = Math.max(this.duration, buffer.duration)
    }
    this.emit('load')
    return loaded.map((l) => l.stem)
  }

  getPeaks(stemId: string): Float32Array | null {
    return this.channels.get(stemId)?.peaks ?? null
  }

  /** Decoded audio of a stem (channel 0), for analysis. */
  getSamples(stemId: string): { data: Float32Array; sampleRate: number } | null {
    const ch = this.channels.get(stemId)
    if (!ch) return null
    return { data: ch.buffer.getChannelData(0), sampleRate: ch.buffer.sampleRate }
  }

  isPlaying(): boolean {
    return this.playing
  }

  getTime(): number {
    if (!this.playing) return this.pausedAt
    return Math.min(this.duration, this.ctx.currentTime - this.startedAt)
  }

  async play(): Promise<void> {
    if (this.playing) return
    const id = ++this.playRequest
    await this.ctx.resume()
    if (id !== this.playRequest) return
    const from = this.pausedAt >= this.duration ? 0 : this.pausedAt
    this.startSources(from)
    this.playing = true
    this.applyTracks()
    this.emit('play')
  }

  pause(): void {
    this.playRequest++
    if (!this.playing) return
    this.pausedAt = this.getTime()
    this.stopSources()
    this.playing = false
    this.applyTracks()
    this.emit('pause')
  }

  stop(): void {
    this.seek(0)
    this.pause()
  }

  seek(t: number): void {
    const target = Math.min(this.duration, Math.max(0, t))
    if (this.playing) {
      this.stopSources()
      this.startSources(target)
    } else {
      this.pausedAt = target
    }
    this.applyTracks()
    this.emit('seek')
  }

  setTracks(tracks: TrackState[]): void {
    this.tracks = tracks
    this.applyTracks()
  }

  dispose(): void {
    this.loadId++
    this.playRequest++
    this.stopSources()
    this.playing = false
    this.listeners.clear()
    void this.ctx.close()
  }

  private startSources(from: number): void {
    this.startedAt = this.ctx.currentTime - from
    for (const ch of this.channels.values()) {
      const src = this.ctx.createBufferSource()
      src.buffer = ch.buffer
      src.connect(ch.envelope)
      src.onended = () => this.handleEnded()
      src.start(0, from)
      ch.source = src
    }
  }

  private stopSources(): void {
    for (const ch of this.channels.values()) {
      if (ch.source) {
        ch.source.onended = null
        try {
          ch.source.stop()
        } catch {
          // already stopped
        }
        ch.source.disconnect()
        ch.source = null
      }
    }
  }

  private handleEnded(): void {
    if (!this.playing || this.getTime() < this.duration - 0.05) return
    this.stopSources()
    this.playing = false
    this.pausedAt = this.duration
    this.applyTracks()
    this.emit('pause')
  }

  private emit(type: EngineEvent['type']): void {
    const event = { type, time: this.getTime() }
    for (const listener of this.listeners) listener(event)
  }

  /** Push fader, mute/solo and automation state onto the audio graph. */
  private applyTracks(): void {
    const now = this.ctx.currentTime
    const t0 = this.getTime()
    for (const track of this.tracks) {
      const ch = this.channels.get(track.stemId)
      if (!ch) continue

      ch.fader.gain.cancelScheduledValues(now)
      ch.fader.gain.setTargetAtTime(track.gain, now, SMOOTH)

      ch.gate.gain.cancelScheduledValues(now)
      ch.gate.gain.setTargetAtTime(isAudible(this.tracks, track) ? 1 : 0, now, SMOOTH)

      const env = ch.envelope.gain
      env.cancelScheduledValues(now)
      const { start, ramps } = automationFrom(track.automation, t0)
      env.setValueAtTime(start, now)
      if (this.playing) {
        for (const kf of ramps) env.linearRampToValueAtTime(kf.value, this.startedAt + kf.t)
      }
    }
  }
}

function computePeaks(buffer: AudioBuffer): Float32Array {
  const data = buffer.getChannelData(0)
  const peaks = new Float32Array(PEAK_BINS)
  const per = Math.max(1, Math.floor(data.length / PEAK_BINS))
  for (let b = 0; b < PEAK_BINS; b++) {
    let max = 0
    const start = b * per
    const end = Math.min(data.length, start + per)
    for (let i = start; i < end; i++) {
      const v = Math.abs(data[i])
      if (v > max) max = v
    }
    peaks[b] = max
  }
  return peaks
}
