import { memo, useEffect, useRef } from 'react'

interface Props {
  peaks: Float32Array
  color: string
  dimmed: boolean
}

const BAR = 2
const GAP = 1

function WaveformImpl({ peaks, color, dimmed }: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const el = wrap.current
    const cv = canvas.current
    if (!el || !cv) return
    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const w = el.clientWidth
      const h = el.clientHeight
      cv.width = Math.max(1, Math.floor(w * dpr))
      cv.height = Math.max(1, Math.floor(h * dpr))
      const ctx = cv.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      const g = ctx.createLinearGradient(0, 0, 0, h)
      g.addColorStop(0, `${color}99`)
      g.addColorStop(0.5, color)
      g.addColorStop(1, `${color}99`)
      ctx.fillStyle = g
      const mid = h / 2
      for (let x = 0; x < w; x += BAR + GAP) {
        const from = Math.floor((x / w) * peaks.length)
        const to = Math.max(from + 1, Math.floor(((x + BAR) / w) * peaks.length))
        let m = 0
        for (let i = from; i < to && i < peaks.length; i++) m = Math.max(m, peaks[i])
        const amp = Math.max(1, m * mid * 0.88)
        ctx.beginPath()
        ctx.roundRect(x, mid - amp, BAR, amp * 2, 1)
        ctx.fill()
      }
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(el)
    return () => ro.disconnect()
  }, [peaks, color])

  return (
    <div
      ref={wrap}
      className={`absolute inset-0 transition-opacity duration-200 ${dimmed ? 'opacity-20' : 'opacity-100'}`}
    >
      <canvas ref={canvas} className="block h-full w-full" />
    </div>
  )
}

export const Waveform = memo(WaveformImpl)
