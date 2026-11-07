import { useEffect, useState } from 'react'
import type { AudioEngine } from '../audio/engine'

export function useEngineTime(engine: AudioEngine | null): number {
  const [t, setT] = useState(0)
  useEffect(() => {
    if (!engine) return
    let raf = 0
    const loop = () => {
      setT(engine.getTime())
      raf = requestAnimationFrame(loop)
    }
    loop()
    return () => cancelAnimationFrame(raf)
  }, [engine])
  return t
}
