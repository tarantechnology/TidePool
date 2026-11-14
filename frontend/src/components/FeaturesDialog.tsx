import { useEffect, useRef } from 'react'
import { FEATURE_GUIDE } from '../features'

/** The features, named and explained, in a centered dialog. Escape, the close button or a click outside closes it. */
export function FeaturesDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = dialog.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
      aria-labelledby="features-title"
      className="tp-dialog m-auto w-[min(40rem,calc(100vw-2rem))] rounded-3xl border border-line bg-surface p-0 text-fg shadow-[0_30px_80px_-20px_rgb(14_36_64/0.35)] backdrop:bg-fg/30 backdrop:backdrop-blur-sm"
    >
      <div className="p-6 sm:p-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="features-title" className="font-display text-2xl font-bold tracking-[-0.02em]">What's in the pool</h2>
            <p className="mt-1 text-sm text-muted">Each part of Tidepool is named after something you'd find by the water.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="-mt-1 -mr-2 flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-muted transition-colors hover:bg-surface-2 hover:text-fg">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
          </button>
        </div>
        <dl className="mt-6 grid gap-x-6 gap-y-4 sm:grid-cols-[8rem_1fr]">
          {FEATURE_GUIDE.map((f, i) => (
            <div key={f.name} className="contents" style={{ ['--i' as string]: i }}>
              <dt className="tp-feature flex items-center gap-2 font-display text-[15px] font-bold text-accent">
                <svg width="14" height="14" viewBox="0 0 12 12" fill="none" aria-hidden="true"><circle cx="6" cy="6" r="5" stroke="currentColor" strokeOpacity="0.4" /><circle cx="6" cy="6" r="2" fill="currentColor" /></svg>
                {f.name}
              </dt>
              <dd className="tp-feature -mt-3 text-sm leading-relaxed text-fg-2 sm:mt-0">{f.what}</dd>
            </div>
          ))}
        </dl>
      </div>
    </dialog>
  )
}
