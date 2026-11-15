import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { navigate } from '../hooks/useHashRoute'
import { projectsSnapshot, subscribeProjects } from '../projects/registry'


/**
 * The open project's name in the top bar. Opens a menu to start a new project, jump to a
 * recent one, or see them all. Pending edits are saved before leaving.
 */
export function ProjectSwitcher({ projectId, fallback, disabled, beforeLeave }: {
  projectId: string | null
  /** What to call the editor when no project is open. */
  fallback: string
  /** True while the project is busy (uploading, separating), when leaving would lose the work. */
  disabled: boolean
  beforeLeave: () => Promise<void>
}) {
  const projects = useSyncExternalStore(subscribeProjects, projectsSnapshot)
  const [open, setOpen] = useState(false)
  const [leaving, setLeaving] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const current = projects.find((p) => p.id === projectId)
  const others = projects.filter((p) => p.id !== projectId).slice(0, 5)

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('pointerdown', onDown); window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open])

  const go = async (hash: string) => {
    setLeaving(true)
    try { await beforeLeave() } catch { /* the edits were already sent once; leave anyway */ }
    setOpen(false); setLeaving(false)
    navigate(hash)
  }

  return (
    <div ref={root} className="relative min-w-0">
      <button type="button" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        className="flex h-8 max-w-56 min-w-0 items-center gap-1.5 rounded-full border border-line bg-surface px-3 text-xs text-fg-2 transition-colors hover:border-line-strong hover:text-fg">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${current ? 'bg-accent' : 'bg-faint'}`} />
        <span className="truncate font-medium">{current?.name ?? fallback}</span>
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" className={`shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden="true"><path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && (
        <div role="menu" className="rise-in shadow-card absolute top-10 left-0 z-40 w-72 rounded-xl border border-line bg-surface p-1.5 text-xs">
          {disabled && <p className="px-2 py-1.5 text-[11px] text-amber-800">Wait for the current step to finish before switching projects.</p>}
          <button type="button" role="menuitem" disabled={disabled || leaving} onClick={() => void go('#/app/new')}
            className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left font-medium text-fg hover:bg-surface-2 disabled:opacity-40">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent text-white"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" /></svg></span>
            New project
          </button>
          {others.length > 0 && <p className="mt-1 border-t border-line px-2 pt-2 pb-1 text-[11px] text-subtle">Recent</p>}
          {others.map((p) => (
            <button key={p.id} type="button" role="menuitem" disabled={disabled || leaving} onClick={() => void go(`#/app/p/${encodeURIComponent(p.id)}`)}
              className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left hover:bg-surface-2 disabled:opacity-40">
              <span className="h-6 w-10 shrink-0 overflow-hidden rounded bg-surface-2 ring-1 ring-line">{p.thumb && <img src={p.thumb} alt="" className="h-full w-full object-cover" />}</span>
              <span className="min-w-0 flex-1 truncate text-fg">{p.name}</span>
              {p.layers ? <span className="shrink-0 text-[10px] text-subtle">{p.layers} layers</span> : null}
            </button>
          ))}
          <button type="button" role="menuitem" disabled={leaving} onClick={() => void go('#/app')}
            className="mt-1 flex w-full items-center justify-between rounded-lg border-t border-line px-2 pt-2 pb-1.5 text-left text-accent hover:bg-surface-2">
            All projects
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
        </div>
      )}
    </div>
  )
}
