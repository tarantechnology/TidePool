import type { Stem } from '../types'

// Synthetic placeholder stems so the editor runs with no assets. They go
// through the same fetch + decode path as real stems (blob URL of a WAV).
// Scene: two "speakers" whose delivery intensifies from ~6s to ~16s, a siren
// and a low ambience bed. These are tones, not speech.

const SAMPLE_RATE = 44100
const DURATION = 20

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

/** 0 (calm) to 1 (heated) across the scene. */
function intensity(t: number): number {
  return smoothstep(6, 16, t)
}

function normalize(data: Float32Array, peak: number): void {
  let max = 0
  for (let i = 0; i < data.length; i++) max = Math.max(max, Math.abs(data[i]))
  if (max === 0) return
  const k = peak / max
  for (let i = 0; i < data.length; i++) data[i] *= k
}

function speaker(f0: number, windows: Array<[number, number]>, seed: number): Float32Array {
  const n = SAMPLE_RATE * DURATION
  const out = new Float32Array(n)
  const rand = mulberry32(seed)
  let phase = 0
  let syllable = 0
  const jitterPhase = rand() * Math.PI * 2
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE
    let gate = 0
    for (const [s, e] of windows) {
      if (t >= s && t <= e) {
        gate = Math.min(1, (t - s) / 0.05, (e - t) / 0.05)
        break
      }
    }
    const inten = intensity(t)
    const f = f0 * (1 + 0.25 * inten) * (1 + 0.02 * Math.sin(2 * Math.PI * 5 * t + jitterPhase))
    phase += (2 * Math.PI * f) / SAMPLE_RATE
    syllable += (2 * Math.PI * (3.5 + 3 * inten)) / SAMPLE_RATE
    if (gate === 0) continue
    const env = Math.pow(0.5 * (1 + Math.sin(syllable)), 1.5)
    let s = 0
    for (let k = 1; k <= 8; k++) s += Math.sin(k * phase) / k
    const amp = 0.25 + 0.6 * inten
    out[i] = s * env * amp * gate
  }
  normalize(out, 0.8)
  return out
}

function siren(): Float32Array {
  const n = SAMPLE_RATE * DURATION
  const out = new Float32Array(n)
  let phase = 0
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE
    const f = 700 + 400 * Math.sin((2 * Math.PI * t) / 3)
    phase += (2 * Math.PI * f) / SAMPLE_RATE
    out[i] = Math.sin(phase) + 0.3 * Math.sin(2 * phase)
  }
  normalize(out, 0.7)
  return out
}

function ambience(seed: number): Float32Array {
  const n = SAMPLE_RATE * DURATION
  const out = new Float32Array(n)
  const rand = mulberry32(seed)
  let lp = 0
  for (let i = 0; i < n; i++) {
    lp += 0.02 * ((rand() * 2 - 1) - lp)
    out[i] = lp
  }
  normalize(out, 0.5)
  return out
}

function encodeWav(samples: Float32Array): Blob {
  const bytes = samples.length * 2
  const buf = new ArrayBuffer(44 + bytes)
  const v = new DataView(buf)
  const str = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i))
  }
  str(0, 'RIFF')
  v.setUint32(4, 36 + bytes, true)
  str(8, 'WAVE')
  str(12, 'fmt ')
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true) // PCM
  v.setUint16(22, 1, true) // mono
  v.setUint32(24, SAMPLE_RATE, true)
  v.setUint32(28, SAMPLE_RATE * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  str(36, 'data')
  v.setUint32(40, bytes, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buf], { type: 'audio/wav' })
}

export interface MockStems {
  stems: Stem[]
  revoke: () => void
}

export function createMockStems(): MockStems {
  const specs: Array<{ id: string; label: string; kind: Stem['kind']; data: Float32Array }> = [
    {
      id: 'speaker-a',
      label: 'Speaker A',
      kind: 'dialogue',
      data: speaker(
        120,
        [
          [0.5, 4.5],
          [8, 12],
          [13, 17],
          [18, 19.8],
        ],
        1,
      ),
    },
    {
      id: 'speaker-b',
      label: 'Speaker B',
      kind: 'dialogue',
      data: speaker(
        210,
        [
          [4.8, 7.8],
          [10, 14.5],
          [15, 19],
        ],
        2,
      ),
    },
    { id: 'siren', label: 'Siren', kind: 'sfx', data: siren() },
    { id: 'ambience', label: 'Ambience', kind: 'ambience', data: ambience(3) },
  ]
  const urls: string[] = []
  const stems = specs.map((s) => {
    const url = URL.createObjectURL(encodeWav(s.data))
    urls.push(url)
    return { id: s.id, label: s.label, kind: s.kind, url, duration: DURATION }
  })
  return { stems, revoke: () => urls.forEach((u) => URL.revokeObjectURL(u)) }
}
