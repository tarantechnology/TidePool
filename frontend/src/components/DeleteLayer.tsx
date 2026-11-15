import { useEffect, useRef, useState } from 'react'

interface Props {
  label: string
  /** The layer that holds everything the others don't, if there is one. Its own button is hidden. */
  residualLabel: string | null
  isResidual: boolean
  /** fold: add this layer's sound to the residual (the mix keeps it), or take it out of the mix. */
  onDelete: (fold: boolean) => Promise<void>
  className?: string
  /** Which way the choice opens: below the button (lanes) or above it (mixer strips). */
  side?: 'top' | 'bottom'
}

/**
 * Delete a layer. It asks what happens to the sound: fold it back into the residual, so the mix
 * sounds the same and only the separate layer goes, or remove it from the mix too. Both undo.
 */
export function DeleteLayer({ label, residualLabel, isResidual, onDelete, className = '', side = 'bottom' }: Props) {
  const [open, setOpen] = useState(false)
  const [working, setWorking] = useState(false)
  const box = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', close); document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', esc) }
  }, [open])
  if (isResidual) return null
  const run = (fold: boolean) => {
    setWorking(true)
    void onDelete(fold).catch(() => { /* The panel shows the error. */ }).finally(() => { setWorking(false); setOpen(false) })
  }
  const choice = 'block w-full rounded-lg px-2.5 py-1.5 text-left transition-colors disabled:opacity-50'
  return (
    <span ref={box} className={`relative inline-flex ${className}`}>
      <button
        aria-label={`Delete ${label}`}
        title="Delete this layer"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={`flex h-6 w-6 items-center justify-center rounded-md border transition-colors ${open ? 'border-red-200 bg-red-50 text-red-600' : 'border-line bg-tint/[0.03] text-subtle hover:border-red-200 hover:text-red-600'}`}
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M2.5 4h11M6.5 4V2.5h3V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v4.5M9.2 6.5v4.5" />
        </svg>
      </button>
      {open && (
        <span role="dialog" aria-label={`Delete ${label}`}
          className={`absolute right-0 z-50 w-52 rounded-xl border border-line bg-surface p-1.5 text-[12px] leading-snug shadow-xl ${side === 'bottom' ? 'top-full mt-1.5' : 'bottom-full mb-1.5'}`}>
          <span className="block px-2.5 pt-1 pb-1.5 font-medium text-fg">Delete “{label}”?</span>
          {residualLabel && (
            <button disabled={working} onClick={() => run(true)} className={`${choice} hover:bg-accent/10`}>
              <span className="block font-medium text-fg">Fold into “{residualLabel}”</span>
              <span className="block text-[11px] text-subtle">The layer goes, its sound stays in the mix.</span>
            </button>
          )}
          <button disabled={working} onClick={() => run(false)} className={`${choice} hover:bg-red-50`}>
            <span className="block font-medium text-red-600">Delete its sound too</span>
            <span className="block text-[11px] text-subtle">Takes this sound out of the mix.</span>
          </button>
          <span className="block px-2.5 pt-1 text-[10px] text-subtle">{working ? 'Deleting…' : 'Undo brings it back (⌘Z).'}</span>
        </span>
      )}
    </span>
  )
}
