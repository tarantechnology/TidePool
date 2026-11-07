import { useEffect } from 'react'
import type { RefObject } from 'react'
import type { AudioEngine } from '../audio/engine'
import { computeSyncAction } from '../audio/videoSync'

/**
 * Seeking takes time, and a video that lands late would read as drifted again
 * straight away, so corrections are rate-limited instead of firing every frame.
 */
const SEEK_COOLDOWN_MS = 500

/** Mirror engine transport onto a muted video element and correct drift. */
export function useVideoSync(
  engine: AudioEngine | null,
  video: RefObject<HTMLVideoElement | null>,
  src: string | null,
  onError: (message: string) => void,
): void {
  useEffect(() => {
    const el = video.current
    if (!engine || !el || !src) return

    const play = () => {
      el.play().catch((err: unknown) => {
        // An interrupted play() is routine: pause or seek arrived first.
        if (err instanceof DOMException && err.name === 'AbortError') return
        onError(err instanceof Error ? err.message : String(err))
      })
    }

    const off = engine.subscribe((e) => {
      if (e.type === 'play') {
        play()
      } else {
        if (e.type !== 'seek') el.pause()
        el.currentTime = e.time
      }
    })

    let raf = 0
    let lastSeek = 0
    const loop = (now: number) => {
      // readyState 1 means metadata, so duration is known.
      if (el.readyState >= 1 && !el.seeking && now - lastSeek > SEEK_COOLDOWN_MS) {
        // A video shorter than the audio holds on its last frame.
        const target = Math.min(engine.getTime(), el.duration)
        if (computeSyncAction(el.currentTime, target) === 'seek-video') {
          el.currentTime = target
          lastSeek = now
        }
      }
      // A video that failed to load reports through its own error event.
      if (engine.isPlaying() && el.paused && !el.ended && !el.error) play()
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      off()
      el.pause()
    }
  }, [engine, video, src, onError])
}
