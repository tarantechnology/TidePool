import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import type { DragEvent } from 'react'
import { Logo } from '../components/Logo'
import { RippleField } from '../components/RippleField'
import { getProject as fetchProject } from '../api/backend'
import { isAudioFile, isVideoFile } from '../api/stemSources'
import { navigate } from '../hooks/useHashRoute'
import { forgetProject, listProjects, nameFromFile, projectsSnapshot, saveProject, setPendingUpload, subscribeProjects } from '../projects/registry'
import type { SavedProject } from '../projects/registry'
import { formatDuration, relative } from '../projects/format'

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || '/api'
const field = 'w-full rounded-lg border border-line bg-surface px-2.5 py-2 text-sm text-fg placeholder:text-subtle focus:border-accent/50 focus:outline-none focus:ring-2 focus:ring-accent/15'


/** The editor opens a new project with this file already uploading. */
function startWith(file: File) {
  setPendingUpload(file)
  navigate('#/app/new')
}

/** A project this tab opened before the project list existed. */
function adoptTabProject() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(`soundcanvas-project:${API_BASE_URL}`) ?? 'null') as { id?: unknown; token?: unknown } | null
    if (typeof saved?.id === 'string' && typeof saved.token === 'string' && !listProjects().some((p) => p.id === saved.id)) saveProject({ id: saved.id, token: saved.token })
  } catch { /* nothing to adopt */ }
}

export default function Projects() {
  const projects = useSyncExternalStore(subscribeProjects, projectsSnapshot)
  const [dragging, setDragging] = useState(false)
  const [notice, setNotice] = useState('')
  const picker = useRef<HTMLInputElement>(null)
  useEffect(() => { adoptTabProject() }, [])

  const take = (files: File[]) => {
    const media = files.find(isVideoFile) ?? files.find(isAudioFile)
    if (media) startWith(media)
    else setNotice('That file is not a video or audio clip. Try an MP4, MOV, WAV or MP3.')
  }
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDragging(false); take(Array.from(e.dataTransfer.files)) }

  return (
    <div className="relative min-h-screen bg-canvas text-fg"
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={(e) => { if (e.relatedTarget === null) setDragging(false) }}
      onDrop={onDrop}>
      <div className="pointer-events-none absolute -top-40 left-1/2 h-80 w-[60rem] -translate-x-1/2 rounded-full bg-sky/30 blur-[120px]" />
      <header className="relative z-10 border-b border-line bg-surface/70 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
          <a href="#/" className="text-fg transition-opacity hover:opacity-80"><Logo /></a>
          <a href="#/app/new" className="text-xs text-muted hover:text-fg">Open an empty editor</a>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-6xl px-5 pt-10 pb-16">
        <h1 className="font-display text-3xl font-bold tracking-[-0.02em] text-balance">Projects</h1>
        <p className="mt-1 text-sm text-muted">Start from a clip, or pick up where you left off. Every project keeps its layers, edits and undo history.</p>

        <button type="button" onClick={() => picker.current?.click()}
          className={`group relative mt-6 flex w-full items-center gap-5 overflow-hidden rounded-2xl border border-dashed px-6 py-7 text-left transition-colors ${dragging ? 'border-accent bg-accent-soft' : 'border-line-strong bg-surface hover:border-accent/50 hover:bg-accent-soft/50'}`}>
          <RippleField className="opacity-60 [mask-image:linear-gradient(to_right,transparent,#000_45%)]" />
          <span className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-accent text-white shadow-sm transition-transform group-hover:scale-105">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
          </span>
          <span className="relative">
            <span className="block text-base font-semibold text-fg">New project</span>
            <span className="block text-sm text-muted">{dragging ? 'Drop to start' : 'Choose a video or audio clip, or drop one anywhere on this page.'}</span>
          </span>
        </button>
        <input ref={picker} type="file" accept=".mp4,.mov,.webm,.mkv,.wav,.mp3,.m4a,video/*,audio/*" hidden onChange={(e) => { take(Array.from(e.target.files ?? [])); e.target.value = '' }} />
        {notice && <p role="status" className="mt-2 text-sm text-amber-800">{notice}</p>}

        <section className="mt-10" aria-labelledby="recent">
          <div className="flex items-baseline justify-between">
            <h2 id="recent" className="text-sm font-semibold text-fg">Recent</h2>
            {projects.length > 0 && <span className="text-xs text-subtle">{projects.length} in this browser</span>}
          </div>
          {projects.length === 0 ? (
            <p className="mt-3 rounded-2xl border border-line bg-surface/60 px-5 py-6 text-sm text-muted">Projects you open show up here, so you can switch between them without an ID or token.</p>
          ) : (
            <ul className="mt-3 grid gap-x-5 gap-y-7 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
              {projects.map((p, i) => <ProjectTile key={p.id} project={p} index={i} />)}
            </ul>
          )}
        </section>

        <OpenById />
      </main>
    </div>
  )
}

function ProjectTile({ project: p, index }: { project: SavedProject; index: number }) {
  const [confirm, setConfirm] = useState(false)
  const meta = [p.duration ? formatDuration(p.duration) : null, p.layers ? `${p.layers} layer${p.layers === 1 ? '' : 's'}` : 'Not split yet'].filter(Boolean).join(' · ')
  return (
    <li className="rise-in group relative" style={{ animationDelay: `${Math.min(index, 8) * 30}ms` }}>
      <a href={`#/app/p/${encodeURIComponent(p.id)}`} className="block rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40">
        <div className="relative aspect-video overflow-hidden rounded-xl border border-line bg-surface-2 transition-shadow group-hover:shadow-card">
          {p.thumb ? <img src={p.thumb} alt="" className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]" />
            : <PoolArt seed={p.id} audio={p.hasVideo === false} />}
          <span className="absolute inset-0 rounded-xl ring-1 ring-inset ring-black/5" />
        </div>
        <p className="mt-2 truncate text-sm font-medium text-fg">{p.name}</p>
        <p className="truncate text-xs text-subtle">{meta} · {relative(p.openedAt)}</p>
      </a>
      <div className="absolute top-2 right-2 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        {confirm ? (
          <span className="flex items-center gap-1 rounded-full bg-surface/95 p-0.5 text-[11px] shadow-card">
            <button type="button" className="rounded-full px-2 py-1 font-medium text-red-700 hover:bg-red-50" onClick={() => forgetProject(p.id)}>Remove from list</button>
            <button type="button" className="rounded-full px-2 py-1 text-muted hover:text-fg" onClick={() => setConfirm(false)}>Keep</button>
          </span>
        ) : (
          <button type="button" aria-label={`Remove ${p.name} from this list`} title="Remove from this list" onClick={() => setConfirm(true)}
            className="flex h-7 w-7 items-center justify-center rounded-full bg-surface/90 text-muted shadow-card hover:text-fg">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
          </button>
        )}
      </div>
    </li>
  )
}

/** Rings in a pool, placed from the project id, for a clip with no frame to show. */
function PoolArt({ seed, audio }: { seed: string; audio: boolean }) {
  let h = 0
  for (const c of seed) h = (h * 31 + c.charCodeAt(0)) >>> 0
  const x = 30 + (h % 40), y = 35 + ((h >> 8) % 30)
  return (
    <svg viewBox="0 0 160 90" className="h-full w-full" aria-hidden="true">
      <rect width="160" height="90" fill="#eef5fc" />
      {[34, 24, 15, 7].map((r, i) => <circle key={r} cx={x * 1.6} cy={y * 0.9} r={r} fill="none" stroke="#2270c8" strokeOpacity={0.12 + i * 0.12} strokeWidth="1.2" />)}
      <circle cx={x * 1.6} cy={y * 0.9} r="2.6" fill="#2270c8" />
      {audio && <text x="152" y="82" textAnchor="end" fontSize="8" fill="#536a86" fontFamily="Geist Mono, monospace">audio</text>}
    </svg>
  )
}

/** For a project made on another computer or shared by a teammate. */
function OpenById() {
  const [id, setId] = useState('')
  const [token, setToken] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const open = async () => {
    setError(''); setBusy(true)
    try {
      const info = await fetchProject({ baseUrl: API_BASE_URL, token: token.trim() }, id.trim())
      saveProject({ id: info.project_id, token: token.trim(), name: nameFromFile(info.filename), ...(info.metadata?.duration ? { duration: info.metadata.duration } : {}), hasVideo: Boolean(info.metadata?.has_video) }, true)
      navigate(`#/app/p/${encodeURIComponent(info.project_id)}`)
    } catch (e) {
      setError(e instanceof Error && /40[14]/.test(e.message) ? 'That ID and token do not match a project.' : e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }
  return (
    <details className="group mt-12 max-w-md text-sm">
      <summary className="cursor-pointer list-none text-muted hover:text-fg"><span className="inline-block transition-transform group-open:rotate-90">›</span> Open with an ID</summary>
      <form className="mt-3 space-y-2" onSubmit={(e) => { e.preventDefault(); if (id.trim() && token.trim()) void open() }}>
        <p className="text-xs text-muted">For a project from another browser. Its ID and token are under Upload, in Share access.</p>
        <input className={field} value={id} onChange={(e) => setId(e.target.value)} placeholder="Project ID" aria-label="Project ID" />
        <input className={field} type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="Project token" aria-label="Project token" />
        <button type="submit" disabled={busy || !id.trim() || !token.trim()} className="rounded-lg bg-accent px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-[#1b62b3] disabled:opacity-40">{busy ? 'Opening…' : 'Open project'}</button>
        {error && <p role="alert" className="text-xs text-red-600">{error}</p>}
      </form>
    </details>
  )
}
