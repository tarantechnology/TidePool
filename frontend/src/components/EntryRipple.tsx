import { useEffect, useRef, useState } from 'react'
import { Logo } from './Logo'

// The water opens onto the editor at 1.35s (the tp-open delay in index.css) and is gone by 2.3s.
const TOTAL_MS = 2300

// The pool, in world units: a grid of points on a plane below and in front of the camera.
const COLS = 220
const ROWS = 110
const X_SPAN = 6
const Z_NEAR = 1.4
const Z_FAR = 8.5
const CAMERA_HEIGHT = 1.05
const ORIGIN_Z = 4.6

// Blues, from deep water to the lit crest of a ring.
const SHADES = ['#2a64b0', '#3478d0', '#4a92e6', '#6eaef2', '#9ccbf7', '#d6ecff']

/**
 * Entering the app: a pool of points, a glowing drop at its center, and rings
 * that roll outward through the points like a sound wave. Then the water opens
 * from the drop onto the editor, which has been rendering underneath the whole
 * time. Any click or key skips it, and reduced motion skips it entirely.
 */
export function EntryRipple() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [done, setDone] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)

  useEffect(() => {
    if (done) return
    const finish = () => setDone(true)
    const timer = window.setTimeout(finish, TOTAL_MS)
    window.addEventListener('keydown', finish, { once: true })
    return () => { window.clearTimeout(timer); window.removeEventListener('keydown', finish) }
  }, [done])

  useEffect(() => {
    if (done) return
    const el = canvas.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const resize = () => { el.width = window.innerWidth * dpr; el.height = window.innerHeight * dpr }
    resize()
    window.addEventListener('resize', resize)

    const start = performance.now()
    let frame = 0
    const buckets: number[][] = SHADES.map(() => [])

    const draw = (now: number) => {
      const t = (now - start) / 1000
      const w = el.width, h = el.height
      const focal = h * 0.95
      const horizon = h * 0.4
      const cx = w / 2
      ctx.clearRect(0, 0, w, h)
      for (const b of buckets) b.length = 0

      const fadeIn = Math.min(1, t / 0.35)
      // The wave front leaves the drop at t = 0.15s and travels outward.
      const front = Math.max(0, (t - 0.15) * 3.4)
      for (let j = 0; j < ROWS; j++) {
        // Rows evenly spaced on screen, not in depth, so the horizon doesn't bunch into a wall.
        const z = 1 / (1 / Z_FAR + (j / (ROWS - 1)) * (1 / Z_NEAR - 1 / Z_FAR))
        for (let i = 0; i < COLS; i++) {
          const x = (i / (COLS - 1) - 0.5) * X_SPAN * (z / 3.2)
          const dx = x, dz = z - ORIGIN_Z
          const r = Math.sqrt(dx * dx + dz * dz)
          // Rings only exist behind the front, and fade with distance like a real ripple.
          const reached = Math.max(0, Math.min(1, (front - r) * 1.6))
          const ring = Math.sin(r * 2.7 - t * 7.5) * Math.exp(-r * 0.17) * reached
          // Uneven crests, like the peaks of a waveform rather than a perfect sine.
          const grain = 0.55 + 0.45 * Math.sin(x * 5.3 + z * 1.7) * Math.sin(x * 2.1 - z * 3.9 + t)
          const swell = Math.sin(x * 1.3 + t * 1.6) * Math.cos(z * 1.1 - t * 1.2) * 0.05
          const y = ring * 0.5 * grain + swell
          const sx = cx + (x * focal) / z
          const sy = horizon + ((CAMERA_HEIGHT - y) * focal) / z
          if (sx < -4 || sx > w + 4 || sy < -4 || sy > h + 4) continue
          // Far points sink into the dark, like fog over the water.
          const fog = ((z - Z_NEAR) / (Z_FAR - Z_NEAR)) * 4
          // Every point is lit; crests brighten toward white, troughs and distance dim.
          const lit = Math.max(0, Math.round(2.6 + y * 9 - fog * 0.5))
          buckets[Math.min(SHADES.length - 1, lit)].push(sx, sy, Math.max(1 * dpr, (2.5 * dpr) / z))
        }
      }

      ctx.globalCompositeOperation = 'lighter'
      buckets.forEach((pts, k) => {
        ctx.fillStyle = SHADES[k]
        ctx.globalAlpha = (0.62 + k * 0.075) * fadeIn
        ctx.beginPath()
        for (let p = 0; p < pts.length; p += 3) { const s = pts[p + 2]; ctx.rect(pts[p] - s / 2, pts[p + 1] - s / 2, s, s) }
        ctx.fill()
      })

      // The drop: a glowing orb hovering over the center of the rings.
      const ox = cx, oy = horizon + ((CAMERA_HEIGHT - 1.0) * focal) / ORIGIN_Z - h * 0.06
      const pulse = 1 + Math.sin(t * 6) * 0.04
      const radius = (h * 0.05) * Math.min(1, t / 0.3) * pulse
      if (radius > 0) {
        const halo = ctx.createRadialGradient(ox, oy, 0, ox, oy, radius * 5)
        halo.addColorStop(0, 'rgba(140,196,242,0.55)'); halo.addColorStop(1, 'rgba(140,196,242,0)')
        ctx.globalAlpha = 1; ctx.fillStyle = halo
        ctx.beginPath(); ctx.arc(ox, oy, radius * 5, 0, Math.PI * 2); ctx.fill()
        const orb = ctx.createRadialGradient(ox - radius * 0.3, oy - radius * 0.35, radius * 0.1, ox, oy, radius)
        orb.addColorStop(0, '#ffffff'); orb.addColorStop(0.45, '#cfe6fd'); orb.addColorStop(1, '#4f9ae8')
        ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = orb
        ctx.beginPath(); ctx.arc(ox, oy, radius, 0, Math.PI * 2); ctx.fill()
      }
      ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1
      el.parentElement?.style.setProperty('--oy', `${(oy / h) * 100}%`)
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw)
    return () => { cancelAnimationFrame(frame); window.removeEventListener('resize', resize) }
  }, [done])

  if (done) return null
  return (
    <div className="tp-entry fixed inset-0 z-[60] cursor-pointer overflow-hidden bg-[#06142a]" onClick={() => setDone(true)} aria-hidden="true">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_55%_at_50%_45%,rgb(47_127_216/0.35),transparent_70%)]" />
      <canvas ref={canvas} className="absolute inset-0 h-full w-full" />
      <div className="tp-name absolute inset-x-0 top-[13%] flex justify-center text-white [text-shadow:0_0_24px_rgb(140_196_242/0.8)]">
        <Logo className="scale-[2.4] [&_circle]:stroke-[#b8dcfb] [&_circle:last-child]:fill-[#b8dcfb]" />
      </div>
    </div>
  )
}
