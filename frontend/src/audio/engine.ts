import { automationFrom, isAudible, stemEnd, stemStart, withMuteRanges } from '../edits'
import type { Stem, TrackState } from '../types'
import type { SourceClip } from '../state/clipSequence'
import { lengthOf, timelinePosition } from '../state/timeline'

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

/** A stem that could not be fetched or decoded, and the reason it gave. */
export interface StemLoadFailure {
  stem: Stem
  reason: string
}

/** What replaceStems loaded, and what it could not. */
export interface StemLoadResult {
  /** The stems now in the graph, with their measured duration. */
  stems: Stem[]
  failures: StemLoadFailure[]
}

/** One line naming every stem that failed and why, for the status chip. */
export function describeStemFailures(failures: readonly StemLoadFailure[]): string {
  return failures.map(({ stem, reason }) => `${stem.label || stem.id} (${reason})`).join(', ')
}

/**
 * How to start a stem's source for playback from a timeline position: after
 * delay seconds, from bufferOffset into the audio. Null once the stem is over.
 */
export function sourceStart(
  position: number,
  startOffset: number,
  bufferDuration: number,
): { delay: number; bufferOffset: number } | null {
  if (position >= startOffset + bufferDuration) return null
  if (position < startOffset) return { delay: startOffset - position, bufferOffset: 0 }
  return { delay: 0, bufferOffset: position - startOffset }
}

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
  private sequence:SourceClip[]=[]
  private clipIndex=0
  private boundaryTimer:ReturnType<typeof setTimeout>|null=null
  get timelineDuration():number { return this.sequence.length ? lengthOf(this.sequence) : this.duration }
  getTimelineTime():number { return this.sequence.length ? lengthOf(this.sequence.slice(0,this.clipIndex))+Math.max(0,this.getTime()-this.sequence[this.clipIndex].start) : this.getTime() }
  setSequence(clips:SourceClip[]):void {
    if(JSON.stringify(clips)===JSON.stringify(this.sequence))return
    const time=this.getTimelineTime();this.pause();this.sequence=clips.map(c=>({...c}));this.clipIndex=0;this.seekTimeline(Math.min(time,this.timelineDuration))
  }
  seekTimeline(time:number):void {
    if(!this.sequence.length){this.seek(time);return}
    const position=timelinePosition(this.sequence,time);this.clipIndex=position.index;this.seek(position.source)
  }
  private advanceClip():void {
    if(!this.playing)return
    if(this.clipIndex+1>=this.sequence.length){this.finish();return}
    this.stopSources();this.clipIndex++;this.startSources(this.sequence[this.clipIndex].start);this.applyTracks();this.emit('seek')
  }

  constructor() {
    this.ctx = new AudioContext()
    this.master = this.ctx.createGain()
    this.master.gain.value = 0.7
    // Layers go up to 500%, so a brick-wall limiter keeps the mix from clipping
    // (the server export applies the same idea with FFmpeg's alimiter).
    if (typeof this.ctx.createDynamicsCompressor === 'function') {
      const limiter = this.ctx.createDynamicsCompressor()
      limiter.threshold.value = -1
      limiter.knee.value = 0
      limiter.ratio.value = 20
      limiter.attack.value = 0.003
      limiter.release.value = 0.1
      this.master.connect(limiter)
      limiter.connect(this.ctx.destination)
    } else this.master.connect(this.ctx.destination)
  }

  /** Used for A/B playback; does not change or save layer edits. */
  setOutputMuted(muted: boolean): void {
    this.master.gain.value = muted ? 0 : 0.7
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
   * a failure leaves the current stems playing. One stem that cannot be decoded
   * does not cost the others: it comes back in `failures` and the rest load.
   * Throws only when nothing loaded, which keeps the current stems. Returns
   * null if a newer call replaced this one.
   */
  async replaceStems(stems: Stem[]): Promise<StemLoadResult | null> {
    const id = ++this.loadId
    const settled = await Promise.allSettled(
      stems.map(async (stem) => {
        const res = await fetch(stem.url)
        if (!res.ok) throw new Error(`Failed to fetch stem ${stem.id}: ${res.status}`)
        const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer())
        return { stem: { ...stem, duration: buffer.duration }, buffer }
      }),
    )
    if (id !== this.loadId) return null
    const loaded = settled.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : []))
    const failures: StemLoadFailure[] = settled.flatMap((r, i) =>
      r.status === 'rejected' ? [{ stem: stems[i], reason: r.reason instanceof Error ? r.reason.message : String(r.reason) }] : [],
    )
    // Nothing to swap in, so leave the graph as it is and say why.
    if (stems.length > 0 && loaded.length === 0) {
      throw new Error(`Could not load any stem: ${describeStemFailures(failures)}`)
    }

    this.playRequest++
    this.stopSources()
    this.playing = false
    for (const ch of this.channels.values()) ch.gate.disconnect()
    this.channels.clear()
    this.tracks = []
    this.pausedAt = 0
    this.duration = 0
    this.sequence=[];this.clipIndex=0

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
      this.duration = Math.max(this.duration, stemEnd({ ...stem, duration: buffer.duration }))
    }
    this.emit('load')
    return { stems: loaded.map((l) => l.stem), failures }
  }

  getPeaks(stemId: string): Float32Array | null {
    return this.channels.get(stemId)?.peaks ?? null
  }

  /** Decoded audio of a stem (channel 0), for analysis. */
  getSamples(stemId: string): { data: Float32Array; sampleRate: number; startOffset: number } | null {
    const ch = this.channels.get(stemId)
    if (!ch) return null
    return { data: ch.buffer.getChannelData(0), sampleRate: ch.buffer.sampleRate, startOffset: stemStart(ch.stem) }
  }

  isPlaying(): boolean {
    return this.playing
  }

  getTime(): number {
    if (!this.playing) return this.pausedAt
    return Math.min(this.sequence[this.clipIndex]?.end ?? this.duration, this.ctx.currentTime - this.startedAt)
  }

  async play(): Promise<void> {
    if (this.playing) return
    const id = ++this.playRequest
    await this.ctx.resume()
    if (id !== this.playRequest) return
    if(this.getTimelineTime() >= this.timelineDuration-1e-6)this.seekTimeline(0)
    const from = this.pausedAt
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
    this.pause()
    this.seekTimeline(0)
  }

  seek(t: number): void {
    let target = Math.min(this.duration, Math.max(0, t))
    if(this.sequence.length){
      const current=this.sequence[this.clipIndex]
      if(target<current.start || target>current.end){const index=this.sequence.findIndex(c=>target>=c.start&&target<c.end);if(index>=0)this.clipIndex=index;else target=current.start}
    }
    if (this.playing) {
      this.stopSources()
      // Past every stem nothing starts, so no source will ever end the playback.
      if (!this.startSources(target) && !this.sequence.length) this.finish()
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

  /** Returns whether any stem has audio left to play. */
  private startSources(from: number): boolean {
    const now = this.ctx.currentTime
    this.startedAt = now - from
    const boundary=this.sequence[this.clipIndex]?.end
    if(boundary !== undefined)this.boundaryTimer=setTimeout(()=>this.advanceClip(),Math.max(0,(boundary-from)*1000))
    let started = false
    for (const ch of this.channels.values()) {
      const start = sourceStart(from, stemStart(ch.stem), ch.buffer.duration)
      if (!start) continue
      const src = this.ctx.createBufferSource()
      src.buffer = ch.buffer
      src.connect(ch.envelope)
      src.onended = () => this.handleEnded()
      if(boundary !== undefined){const available=Math.min(ch.buffer.duration-start.bufferOffset,boundary-from-start.delay);if(available<=0)continue;src.start(now+start.delay,start.bufferOffset,available)}else src.start(now + start.delay, start.bufferOffset)
      ch.source = src
      started = true
    }
    return started
  }

  private stopSources(): void {
    if(this.boundaryTimer !== null){clearTimeout(this.boundaryTimer);this.boundaryTimer=null}
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
    if(this.sequence.length)return
    if (!this.playing || this.getTime() < this.duration - 0.05) return
    this.finish()
  }

  private finish(): void {
    this.stopSources()
    this.playing = false
    this.pausedAt = this.sequence[this.clipIndex]?.end ?? this.duration
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
      const { start, ramps } = automationFrom(withMuteRanges(track.automation, ch.stem.muteRanges), t0)
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
