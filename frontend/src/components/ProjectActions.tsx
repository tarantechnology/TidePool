import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import type { ReactNode } from 'react'
import { FEATURES } from '../features'
import { projectsSnapshot, renameProject, saveProject, shareLink, subscribeProjects } from '../projects/registry'

/** Close a popover on a click outside it or on Escape. */
function useDismiss(open: boolean, close: () => void) {
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) close() }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('pointerdown', onDown); window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open, close])
  return root
}

function Popover({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div role="dialog" className="rise-in shadow-card absolute top-10 right-0 z-40 w-80 rounded-xl border border-line bg-surface p-3 text-xs">
      <div className="mb-2.5 flex items-center gap-2 text-sm font-semibold text-fg">{title}</div>
      {children}
    </div>
  )
}

const CheckIcon = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" /></svg>

type SaveState = { kind: 'idle' } | { kind: 'saving' } | { kind: 'saved'; at: number } | { kind: 'error'; message: string }

export interface SaveHandle { save: () => Promise<void> }

/**
 * Save project. Edits already save on their own; this sends anything still pending, says so
 * plainly, and is where the project is named and shared to another browser.
 */
export function SaveButton({ projectId, flush, disabledReason, saveRef }: {
  projectId: string | null
  flush: () => Promise<void>
  /** Why there is nothing to save yet; null when a project is open. */
  disabledReason: string | null
  /** Lets the editor's ⌘S press the button. */
  saveRef?: { current: SaveHandle | null }
}) {
  const projects = useSyncExternalStore(subscribeProjects, projectsSnapshot)
  const project = projects.find((p) => p.id === projectId) ?? null
  const [open, setOpen] = useState(false)
  const [state, setState] = useState<SaveState>({ kind: 'idle' })
  const [name, setName] = useState('')
  const [copied, setCopied] = useState(false)
  const root = useDismiss(open, () => setOpen(false))
  const flash = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => { if (flash.current) clearTimeout(flash.current) }, [])

  const save = async () => {
    if (!projectId || state.kind === 'saving') return
    setState({ kind: 'saving' })
    try {
      await flush()
      const at = Date.now()
      saveProject({ id: projectId, savedAt: at })
      setState({ kind: 'saved', at })
      if (flash.current) clearTimeout(flash.current)
      flash.current = setTimeout(() => setState((s) => (s.kind === 'saved' ? { kind: 'idle' } : s)), 2400)
    } catch (e) {
      setState({ kind: 'error', message: e instanceof Error ? e.message : String(e) })
    }
  }
  useEffect(() => { if (saveRef) saveRef.current = { save } })

  const commitName = () => { if (projectId && name.trim() && name.trim() !== project?.name) renameProject(projectId, name) }
  const copyLink = async () => {
    if (!project) return
    try { await navigator.clipboard.writeText(shareLink(project)); setCopied(true); setTimeout(() => setCopied(false), 2000) } catch { setCopied(false) }
  }

  const label = state.kind === 'saving' ? 'Saving…' : state.kind === 'saved' ? 'Saved' : 'Save'
  return (
    <div ref={root} className="relative">
      <button type="button" aria-haspopup="dialog" aria-expanded={open} disabled={Boolean(disabledReason)}
        title={disabledReason ?? 'Save project (⌘S)'}
        onClick={() => { setName(project?.name ?? ''); setOpen((o) => !o); if (!open) void save() }}
        className={`inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors disabled:opacity-40 ${state.kind === 'saved' ? 'border-accent/40 bg-accent-soft text-accent' : state.kind === 'error' ? 'border-red-200 bg-red-50 text-red-700' : 'border-line bg-surface text-fg-2 hover:border-line-strong hover:text-fg'}`}>
        {state.kind === 'saved' ? <CheckIcon /> : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={state.kind === 'saving' ? 'animate-pulse' : ''}>
            <path d="M5 4h11l3 3v12a1 1 0 01-1 1H6a1 1 0 01-1-1V5a1 1 0 011-1z" /><path d="M8 4v5h7V4M8 20v-6h8v6" />
          </svg>
        )}
        {label}
      </button>
      {open && projectId && (
        <Popover title={state.kind === 'error'
          ? <span className="text-red-700">Couldn’t save</span>
          : <><span className={`flex h-5 w-5 items-center justify-center rounded-full ${state.kind === 'saving' ? 'bg-surface-2 text-muted' : 'bg-accent text-white'}`}>{state.kind === 'saving' ? <span className="animate-pulse-dot h-1.5 w-1.5 rounded-full bg-faint" /> : <CheckIcon />}</span>{state.kind === 'saving' ? 'Saving project…' : 'Project saved'}</>}>
          {state.kind === 'error' && <p role="alert" className="mb-2 text-red-700">{state.message} Your edits are kept here; try Save again.</p>}
          <label className="block text-[11px] font-medium text-fg-2" htmlFor="project-name">Project name</label>
          <form className="mt-1 flex gap-1.5" onSubmit={(e) => { e.preventDefault(); commitName(); setOpen(false) }}>
            <input id="project-name" value={name} onChange={(e) => setName(e.target.value)} onBlur={commitName} maxLength={80}
              placeholder="Name this project"
              className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-2 py-1.5 text-xs text-fg placeholder:text-subtle focus:border-accent/50 focus:ring-2 focus:ring-accent/15 focus:outline-none" />
            <button type="submit" className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white shadow-sm hover:bg-[#1b62b3]">Done</button>
          </form>
          <p className="mt-2 text-[11px] leading-relaxed text-muted">Your layers and edits are stored on TidePool, and every change also saves on its own as you work.</p>

          <div className="mt-3 border-t border-line pt-2.5">
            <p className="text-[11px] font-medium text-fg-2">Open on another computer</p>
            <div className="mt-1 flex items-center gap-2">
              <button type="button" onClick={() => void copyLink()} disabled={!project?.token}
                className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs text-fg transition-colors hover:border-line-strong hover:bg-surface-2 disabled:opacity-40">
                {copied ? <CheckIcon /> : <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" aria-hidden="true"><path d="M10 14a4 4 0 005.66 0l3-3a4 4 0 00-5.66-5.66l-1 1" /><path d="M14 10a4 4 0 00-5.66 0l-3 3a4 4 0 005.66 5.66l1-1" /></svg>}
                {copied ? 'Link copied' : 'Copy project link'}
              </button>
              <span className="text-[10px] text-subtle">Anyone with the link can edit.</span>
            </div>
          </div>
          <a href="#/app" className="mt-3 block border-t border-line pt-2 text-accent hover:underline">All projects</a>
        </Popover>
      )}
    </div>
  )
}

export type ExportFormat = 'mp4' | 'wav' | 'mp3' | 'json'

const FORMATS: { id: ExportFormat; title: string; detail: string }[] = [
  { id: 'mp4', title: 'Video', detail: 'MP4 with your edited sound, turned as shown' },
  { id: 'wav', title: 'Audio', detail: 'WAV, full quality' },
  { id: 'mp3', title: 'Audio, small', detail: 'MP3, easy to send' },
  { id: 'json', title: 'Edit list', detail: 'JSON of every fader and fade, for other tools' },
]

/** Outflow: the clip with its new sound, as a file. */
export function ExportButton({ ready, hasVideo, onExport }: {
  /** Null when the project can export; otherwise why not yet. */
  ready: string | null
  hasVideo: boolean
  onExport: (format: ExportFormat, onStatus: (text: string) => void) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [picked, setFormat] = useState<ExportFormat>('mp4')
  const format: ExportFormat = !hasVideo && picked === 'mp4' ? 'wav' : picked
  const [status, setStatus] = useState<{ text: string; busy: boolean; error?: boolean } | null>(null)
  const busy = status?.busy === true
  const root = useDismiss(open && !busy, () => setOpen(false))

  const go = async () => {
    setStatus({ text: 'Starting…', busy: true })
    try {
      await onExport(format, (text) => setStatus({ text, busy: true }))
      setStatus({ text: 'Downloaded. Check your downloads folder.', busy: false })
    } catch (e) {
      setStatus({ text: e instanceof Error ? e.message : String(e), busy: false, error: true })
    }
  }
  const action = format === 'mp4' ? 'Export video' : format === 'json' ? 'Download edit list' : `Export ${format.toUpperCase()}`

  return (
    <div ref={root} className="relative">
      <button type="button" aria-haspopup="dialog" aria-expanded={open} onClick={() => { setOpen((o) => !o); if (!busy) setStatus(null) }}
        title="Export the clip with its new sound"
        className="inline-flex h-8 items-center gap-1.5 rounded-full bg-accent px-3.5 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-[#1b62b3]">
        {busy ? <span className="animate-pulse-dot h-1.5 w-1.5 rounded-full bg-white" /> : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19h14" /></svg>
        )}
        {busy ? 'Exporting' : 'Export'}
      </button>
      {open && (
        <Popover title={<>Export <span className="rounded-full bg-accent-soft px-1.5 py-px text-[10px] font-medium text-accent">{FEATURES.export}</span></>}>
          {ready ? (
            <p className="rounded-lg bg-surface-2 px-2.5 py-2 text-[11px] leading-relaxed text-fg-2">{ready}</p>
          ) : (
            <>
              <div role="radiogroup" aria-label="Export as" className="space-y-1">
                {FORMATS.map((f) => {
                  const off = f.id === 'mp4' && !hasVideo
                  return (
                    <button key={f.id} type="button" role="radio" aria-checked={format === f.id} disabled={off || busy} onClick={() => setFormat(f.id)}
                      className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors disabled:opacity-40 ${format === f.id ? 'border-accent/50 bg-accent-soft/60' : 'border-line hover:border-line-strong hover:bg-surface-2'} ${f.id === 'json' ? 'mt-2' : ''}`}>
                      <span className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border ${format === f.id ? 'border-accent' : 'border-line-strong'}`}>
                        {format === f.id && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
                      </span>
                      <span className="min-w-0">
                        <span className="block font-medium text-fg">{f.title}</span>
                        <span className="block text-[11px] text-muted">{off ? 'This project has no video' : f.detail}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
              <button type="button" disabled={busy} onClick={() => void go()}
                className="mt-3 w-full rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-[#1b62b3] disabled:opacity-60">
                {busy ? 'Exporting…' : action}
              </button>
              {status && <p role="status" className={`mt-2 text-[11px] ${status.error ? 'text-red-700' : 'text-muted'}`}>{status.text}</p>}
              {!status && <p className="mt-2 text-[11px] text-subtle">Your latest edits are saved first, so the file matches what you hear.</p>}
            </>
          )}
        </Popover>
      )}
    </div>
  )
}
