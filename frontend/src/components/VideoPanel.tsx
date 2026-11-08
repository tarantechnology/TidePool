import { useRef } from 'react'
import type { AudioEngine } from '../audio/engine'
import { useEngineTime } from '../hooks/useEngineTime'
import { useVideoSync } from '../hooks/useVideoSync'

function Corner({ cls }: { cls: string }) {
  return <span className={`absolute h-3 w-3 border-white/25 ${cls}`} />
}

interface Props {
  engine: AudioEngine | null
  src: string | null
  onPick: (file: File) => void
  onDuration: (duration: number) => void
  onError: (message: string) => void
}

export function VideoPanel({ engine, src, onPick, onDuration, onError }: Props) {
  const t = useEngineTime(engine)
  const video = useRef<HTMLVideoElement>(null)
  const picker = useRef<HTMLInputElement>(null)

  useVideoSync(engine, video, src, onError)

  const input = (
    <input
      ref={picker}
      type="file"
      accept="video/*"
      hidden
      onChange={(e) => {
        const picked = e.target.files?.[0]
        if (picked) onPick(picked)
        e.target.value = ''
      }}
    />
  )
  const pickButton = (
    <button
      onClick={() => picker.current?.click()}
      className="rounded-full border border-line px-2.5 py-1 font-mono text-[10px] text-zinc-400 transition-colors hover:border-line-strong hover:bg-white/[0.04] hover:text-zinc-100"
    >
      {src ? 'Replace video' : 'Load video'}
    </button>
  )

  if (src) {
    return (
      <div className="group relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-black">
        <video
          ref={video}
          src={src}
          muted
          playsInline
          preload="auto"
          className="h-full w-full object-contain"
          onLoadedMetadata={(e) => onDuration(e.currentTarget.duration)}
          onError={() => onError('This browser could not play the video.')}
        />
        {input}
        <div className="absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100">{pickButton}</div>
      </div>
    )
  }

  return (
    <div className="relative flex min-w-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-black">
      <div className="bg-grid fade-edges absolute inset-0 opacity-70" />
      <div className="relative flex aspect-video h-[88%] max-w-[94%] flex-col items-center justify-center rounded-lg border border-line bg-gradient-to-b from-white/[0.03] to-transparent">
        <Corner cls="top-2 left-2 border-t border-l" />
        <Corner cls="top-2 right-2 border-t border-r" />
        <Corner cls="bottom-2 left-2 border-b border-l" />
        <Corner cls="right-2 bottom-2 border-r border-b" />
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" className="text-zinc-600" aria-hidden="true">
          <rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" strokeWidth="1.5" />
          <path d="M10 9.5v5l4-2.5-4-2.5z" fill="currentColor" />
        </svg>
        <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-600">No video loaded</div>
        <div className="mt-1 font-mono text-3xl tabular-nums text-zinc-300">{t.toFixed(2)}</div>
        <div className="mt-2">{pickButton}</div>
        {input}
      </div>
    </div>
  )
}
