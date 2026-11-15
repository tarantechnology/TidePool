import { useState } from 'react'

interface Props {
  value: string
  /** Absent when this layer cannot be renamed (mock or local stems). */
  onRename?: (name: string) => Promise<void> | void
  className?: string
  /** Let a long name wrap instead of cutting it off. */
  wrap?: boolean
}

/** A layer name. Double-click it (or press Enter on it) to rename; Enter saves, Escape cancels. */
export function EditableName({ value, onRename, className = '', wrap = false }: Props) {
  const fit = wrap ? 'break-words' : 'truncate'
  const [draft, setDraft] = useState<string | null>(null)
  if (!onRename) return <div className={`${fit} ${className}`}>{value}</div>
  if (draft === null) {
    return (
      <div
        role="button"
        tabIndex={0}
        title="Double-click to rename"
        className={`cursor-text ${fit} rounded px-0.5 -mx-0.5 hover:bg-tint/[0.05] ${className}`}
        onDoubleClick={() => setDraft(value)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === 'F2') { e.preventDefault(); setDraft(value) } }}
      >
        {value}
      </div>
    )
  }
  const commit = () => {
    const next = draft.trim()
    setDraft(null)
    if (next && next !== value) void Promise.resolve(onRename(next)).catch(() => { /* The panel shows the error. */ })
  }
  return (
    <input
      autoFocus
      aria-label="Layer name"
      value={draft}
      maxLength={60}
      onChange={(e) => setDraft(e.target.value)}
      onFocus={(e) => e.currentTarget.select()}
      onBlur={commit}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'Enter') e.currentTarget.blur()
        if (e.key === 'Escape') setDraft(null)
      }}
      className={`w-full min-w-0 rounded border border-accent/50 bg-surface px-1 outline-none ring-2 ring-accent/15 ${className}`}
    />
  )
}
