import { useEffect, useState } from 'react'
import { Logo } from './Logo'
import { clearEntry, readEntry } from '../entry'

const TOTAL_MS = 1500

/**
 * Entering the app: rings spread from where you clicked, the name surfaces, then the
 * water opens outward and the editor is underneath. The editor renders the whole
 * time; this only covers it. Any click or key skips it, and reduced motion skips it.
 */
export function EntryRipple() {
  const [origin] = useState(readEntry)
  const [done, setDone] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)

  useEffect(() => clearEntry(), [])

  useEffect(() => {
    if (done) return
    const finish = () => setDone(true)
    const timer = window.setTimeout(finish, TOTAL_MS)
    window.addEventListener('keydown', finish, { once: true })
    return () => { window.clearTimeout(timer); window.removeEventListener('keydown', finish) }
  }, [done])

  if (done) return null
  const at = { left: `${origin.x * 100}%`, top: `${origin.y * 100}%` }
  return (
    <div
      className="tp-entry fixed inset-0 z-[60] cursor-pointer overflow-hidden"
      style={{ ['--ox' as string]: `${origin.x * 100}%`, ['--oy' as string]: `${origin.y * 100}%` }}
      onClick={() => setDone(true)}
      aria-hidden="true"
    >
      <div className="absolute inset-0 bg-canvas" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_var(--ox)_var(--oy),rgb(140_196_242/0.35),transparent_60%)]" />
      {[0, 1, 2, 3].map((i) => (
        <span key={i} className="tp-ring absolute rounded-full border-2 border-accent/45 shadow-[0_0_24px_rgb(140_196_242/0.5)]" style={{ ...at, ['--i' as string]: i }} />
      ))}
      <div className="tp-name absolute inset-0 flex items-center justify-center text-fg">
        <Logo className="scale-[2.2]" />
      </div>
    </div>
  )
}
