import { useEffect, useState } from 'react'

// Static illustration of the editor for the landing page. The waveform data is
// generated, not real audio.

function rng(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const N = 120

function bars(shape: (x: number) => number, seed: number): number[] {
  const r = rng(seed)
  return Array.from({ length: N }, (_, i) => Math.max(0.05, shape(i / (N - 1)) * (0.5 + 0.5 * r())))
}

function windows(x: number, ws: Array<[number, number]>): number {
  for (const [a, b] of ws) if (x >= a && x <= b) return 0.3 + 0.6 * x
  return 0.06
}

const ROWS = [
  {
    label: 'Speaker A',
    kind: 'Dialogue',
    data: bars((x) => windows(x, [[0.03, 0.2], [0.4, 0.58], [0.64, 0.82], [0.9, 0.98]]), 11),
  },
  {
    label: 'Speaker B',
    kind: 'Dialogue',
    data: bars((x) => windows(x, [[0.22, 0.38], [0.5, 0.72], [0.76, 0.96]]), 22),
  },
  {
    label: 'Siren',
    kind: 'SFX',
    data: bars((x) => (x > 0.4 && x < 0.8 ? 0.92 : 0.42), 33),
  },
  { label: 'Ambience', kind: 'Ambience', data: bars(() => 0.22, 44) },
]

const PHRASES = [
  'Make the siren louder whenever the argument heats up',
  'Mute Speaker B at 12 seconds',
  'Solo the dialogue and drop the ambience',
]

function useTypewriter(phrases: string[]): string {
  const [text, setText] = useState('')
  useEffect(() => {
    let i = 0
    let j = 0
    let deleting = false
    let timer = 0
    const tick = () => {
      const p = phrases[i]
      if (!deleting) {
        j++
        setText(p.slice(0, j))
        if (j >= p.length) {
          deleting = true
          timer = window.setTimeout(tick, 1900)
          return
        }
        timer = window.setTimeout(tick, 36)
      } else {
        j = Math.max(0, j - 2)
        setText(p.slice(0, j))
        if (j === 0) {
          deleting = false
          i = (i + 1) % phrases.length
          timer = window.setTimeout(tick, 380)
          return
        }
        timer = window.setTimeout(tick, 14)
      }
    }
    timer = window.setTimeout(tick, 700)
    return () => window.clearTimeout(timer)
  }, [phrases])
  return text
}

export function HeroPreview() {
  const typed = useTypewriter(PHRASES)
  return (
    <div className="relative overflow-hidden rounded-2xl border border-line-strong bg-surface/80 shadow-[0_40px_120px_-30px_rgb(43_123_214/0.25),0_30px_80px_-20px_rgb(0_0_0/0.8)] backdrop-blur">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <span className="h-2.5 w-2.5 rounded-full bg-tint/10" />
        <span className="h-2.5 w-2.5 rounded-full bg-tint/10" />
        <span className="h-2.5 w-2.5 rounded-full bg-tint/10" />
        <span className="mx-auto font-mono text-[11px] text-subtle">argument-scene</span>
        <span className="w-10" />
      </div>

      <div className="relative">
        {ROWS.map((row, ri) => (
          <div key={row.label} className="grid grid-cols-[7.5rem_1fr] border-b border-line last:border-b-0 sm:grid-cols-[10rem_1fr]">
            <div className="flex flex-col justify-center border-r border-line px-4 py-3">
              <div className="text-[12px] font-medium tracking-tight sm:text-[13px]">{row.label}</div>
              <div className="font-mono text-[9px] uppercase tracking-wider text-subtle">{row.kind}</div>
            </div>
            <div className="relative h-14 px-2 sm:h-[4.5rem]">
              <div className="flex h-full items-center gap-px sm:gap-[2px]">
                {row.data.map((v, i) => (
                  <span
                    key={i}
                    className={`min-w-px flex-1 rounded-full bg-accent/80 ${i % 3 ? 'max-sm:hidden' : ''}`}
                    style={{ height: `${v * 100}%`, opacity: 0.55 + 0.45 * v }}
                  />
                ))}
              </div>
              {ri === 2 && (
                <svg
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  <polyline
                    fill="none"
                    stroke="#0e2440"
                    strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                    points="0,62 38,62 44,14 76,14 82,62 100,62"
                  />
                </svg>
              )}
            </div>
          </div>
        ))}
        <div className="pointer-events-none absolute top-0 right-2 bottom-0 left-[7.5rem] sm:left-[10rem]">
          <div className="absolute top-0 bottom-0 w-px animate-playhead bg-accent shadow-[0_0_12px_2px_rgb(43_123_214/0.5)]" />
        </div>
      </div>

      <div className="border-t border-line p-3">
        <div className="flex items-center gap-3 rounded-xl border border-line-strong bg-canvas/80 px-4 py-3">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" className="shrink-0 text-accent" aria-hidden="true">
            <path
              d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z"
              fill="currentColor"
            />
          </svg>
          <div className="min-w-0 flex-1 truncate text-sm text-fg">
            {typed}
            <span className="ml-0.5 inline-block h-4 w-px animate-caret bg-accent align-middle" />
          </div>
        </div>
      </div>
    </div>
  )
}
