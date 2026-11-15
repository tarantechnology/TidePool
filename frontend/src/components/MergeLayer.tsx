import { useEffect, useRef, useState } from 'react'

interface Props {
  label: string
  /** The other layers this one can merge into, in lane order. */
  targets: { id: string; label: string }[]
  onMerge: (into: string) => Promise<void>
  className?: string
  side?: 'top' | 'bottom'
}

/**
 * Merge this layer into another: its sound is added to that layer and this lane goes. For voices that
 * Explode split in two but belong to one person. The target keeps its name. Undo brings both back.
 */
export function MergeLayer({ label, targets, onMerge, className = '', side }: Props) {
  const [open, setOpen] = useState(false)
  const [up, setUp] = useState(false)
  const [working, setWorking] = useState(false)
  const box = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', close); document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', esc) }
  }, [open])
  if (!targets.length) return null
  const run = (into: string) => {
    setWorking(true)
    void onMerge(into).catch(() => { /* The panel shows the error. */ }).finally(() => { setWorking(false); setOpen(false) })
  }
  return (
    <span ref={box} className={`relative inline-flex ${className}`}>
      <button
        aria-label={`Merge ${label} into another layer`}
        title="Merge into another layer"
        aria-expanded={open}
        onClick={() => {
          // Low lanes open the list upward so the lane edge doesn't cut it off.
          setUp(side ? side === 'top' : (box.current?.getBoundingClientRect().top ?? 0) > window.innerHeight * 0.55)
          setOpen((o) => !o)
        }}
        className={`flex h-6 w-6 items-center justify-center rounded-md border transition-colors ${open ? 'border-accent/40 bg-accent/10 text-accent' : 'border-line bg-tint/[0.03] text-subtle hover:border-accent/40 hover:text-accent'}`}
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 2.5c0 4 5 4 5 7.5v3.5M13 2.5c0 4-5 4-5 7.5M5.5 11 8 13.5 10.5 11" />
        </svg>
      </button>
      {open && (
        <span role="dialog" aria-label={`Merge ${label}`}
          className={`absolute -right-[30px] z-50 w-52 rounded-xl border border-line bg-surface p-1.5 text-[12px] leading-snug shadow-xl ${up ? 'bottom-full mb-1.5' : 'top-full mt-1.5'}`}>
          <span className="block px-2.5 pt-1 pb-1.5 font-medium text-fg">Merge “{label}” into…</span>
          <span className="block max-h-48 overflow-auto">
            {targets.map((t) => (
              <button key={t.id} disabled={working} onClick={() => run(t.id)}
                className="block w-full truncate rounded-lg px-2.5 py-1.5 text-left font-medium text-fg transition-colors hover:bg-accent/10 disabled:opacity-50">
                {t.label}
              </button>
            ))}
          </span>
          <span className="block px-2.5 pt-1 text-[10px] text-subtle">{working ? 'Merging…' : 'Both sounds become one layer with that name. Undo with ⌘Z.'}</span>
        </span>
      )}
    </span>
  )
}
