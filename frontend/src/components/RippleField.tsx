import { useEffect, useRef } from 'react'

const CELL = 4 // CSS pixels per simulation cell; the canvas is scaled up, which softens the water.
const DAMPING = 0.984

/**
 * A shallow pool behind the hero: a 2D wave equation on a coarse grid. Moving the
 * pointer drags a finger through the water, a click drops a stone, and now and then
 * a drop falls on its own. Paused while off screen or in a hidden tab, and drawn
 * still under prefers-reduced-motion.
 */
export function RippleField({ className = '' }: { className?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const el = canvas.current
    const host = el?.parentElement
    if (!el || !host) return
    const ctx = el.getContext('2d')
    if (!ctx) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let cols = 0, rows = 0
    let cur = new Float32Array(0), prev = new Float32Array(0)
    let image: ImageData | null = null
    const buffer = document.createElement('canvas')
    const bctx = buffer.getContext('2d')

    const resize = () => {
      const r = host.getBoundingClientRect()
      cols = Math.max(8, Math.ceil(r.width / CELL)); rows = Math.max(8, Math.ceil(r.height / CELL))
      cur = new Float32Array(cols * rows); prev = new Float32Array(cols * rows)
      buffer.width = cols; buffer.height = rows
      image = bctx?.createImageData(cols, rows) ?? null
      el.width = cols; el.height = rows
    }

    const drop = (x: number, y: number, radius: number, strength: number) => {
      const cx = Math.round(x / CELL), cy = Math.round(y / CELL)
      for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
        const d2 = dx * dx + dy * dy
        if (d2 > radius * radius) continue
        const gx = cx + dx, gy = cy + dy
        if (gx < 1 || gy < 1 || gx >= cols - 1 || gy >= rows - 1) continue
        prev[gy * cols + gx] += strength * (1 - d2 / (radius * radius + 1))
      }
    }

    const step = () => {
      for (let y = 1; y < rows - 1; y++) {
        const row = y * cols
        for (let x = 1; x < cols - 1; x++) {
          const i = row + x
          const v = (prev[i - 1] + prev[i + 1] + prev[i - cols] + prev[i + cols]) / 2 - cur[i]
          cur[i] = v * DAMPING
        }
      }
      const t = prev; prev = cur; cur = t
    }

    const draw = () => {
      if (!image || !bctx) return
      const px = image.data
      for (let y = 1; y < rows - 1; y++) {
        for (let x = 1; x < cols - 1; x++) {
          const i = y * cols + x
          // Light from the top left: the slope decides crest (white glint) or trough (blue shade).
          const slope = (prev[i - 1] - prev[i + 1]) + (prev[i - cols] - prev[i + cols])
          const o = i * 4
          if (slope > 0) { px[o] = 255; px[o + 1] = 255; px[o + 2] = 255; px[o + 3] = Math.min(255, slope * 16) }
          else { px[o] = 34; px[o + 1] = 112; px[o + 2] = 200; px[o + 3] = Math.min(140, -slope * 8) }
        }
      }
      bctx.putImageData(image, 0, 0)
      ctx.clearRect(0, 0, el.width, el.height)
      ctx.drawImage(buffer, 0, 0)
    }

    resize()
    const observer = new ResizeObserver(resize); observer.observe(host)

    let last = { x: -1, y: -1 }
    const local = (e: PointerEvent) => { const r = host.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top } }
    const onMove = (e: PointerEvent) => {
      const p = local(e)
      if (Math.hypot(p.x - last.x, p.y - last.y) < CELL * 2) return
      last = p; drop(p.x, p.y, 2, 2.4)
    }
    const onDown = (e: PointerEvent) => { const p = local(e); drop(p.x, p.y, 3, 14) }
    host.addEventListener('pointermove', onMove)
    host.addEventListener('pointerdown', onDown)

    // A few drops to start, so the surface is alive before anyone touches it.
    const seed = () => { const r = host.getBoundingClientRect(); drop(r.width * (0.2 + Math.random() * 0.6), r.height * (0.15 + Math.random() * 0.5), 3, 5) }
    seed()

    let frame = 0, visible = true, nextDrop = performance.now() + 900
    const io = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible && !frame) frame = requestAnimationFrame(loop) })
    io.observe(host)
    function loop(now: number) {
      frame = 0
      if (!visible || document.hidden) return
      if (now > nextDrop) { seed(); nextDrop = now + 1400 + Math.random() * 1800 }
      step(); draw()
      frame = requestAnimationFrame(loop)
    }
    const onVisibility = () => { if (!document.hidden && visible && !frame) frame = requestAnimationFrame(loop) }
    document.addEventListener('visibilitychange', onVisibility)

    if (reduced) { for (let i = 0; i < 40; i++) step(); draw() } else frame = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); io.disconnect()
      host.removeEventListener('pointermove', onMove); host.removeEventListener('pointerdown', onDown)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  return <canvas ref={canvas} aria-hidden="true" className={`pointer-events-none absolute inset-0 h-full w-full [image-rendering:auto] ${className}`} />
}
