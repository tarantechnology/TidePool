import type { AudioEngine } from '../audio/engine'
import { useEngineTime } from '../hooks/useEngineTime'

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`
}

interface Props {
  engine: AudioEngine | null
  playing: boolean
  duration: number
}

export function Transport({ engine, playing, duration }: Props) {
  const t = useEngineTime(engine)
  const disabled = !engine
  return (
    <div className="flex items-center gap-1.5 rounded-full border border-line bg-tint/[0.03] p-1 pr-5 backdrop-blur">
      <button
        className="flex h-8 w-8 items-center justify-center rounded-full text-muted transition-colors hover:bg-tint/[0.06] hover:text-fg disabled:opacity-40"
        disabled={disabled}
        title="Stop"
        onClick={() => engine?.stop()}
      >
        <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor">
          <rect width="12" height="12" rx="2" />
        </svg>
      </button>
      <button
        className="glow-accent flex h-8 w-8 items-center justify-center rounded-full bg-accent text-white transition-transform hover:scale-105 disabled:opacity-40"
        disabled={disabled}
        title={playing ? 'Pause (Space)' : 'Play (Space)'}
        onClick={() => (playing ? engine?.pause() : void engine?.play())}
      >
        {playing ? (
          <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor">
            <rect x="1" width="3.5" height="12" rx="1" />
            <rect x="7.5" width="3.5" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor">
            <path d="M2 0.5l9.5 5.5L2 11.5z" />
          </svg>
        )}
      </button>
      <div className="ml-3 font-mono text-[13px] tabular-nums text-fg">
        {fmt(t)} <span className="text-faint">/ {fmt(duration)}</span>
      </div>
    </div>
  )
}
