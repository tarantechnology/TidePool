import type { EditAction } from '../types'

/** What the command bar changed, newest last, in plain words. */
function describe(a: EditAction): string {
  const { type, stemId, ...rest } = a
  const what = type.replace(/_/g, ' ')
  const detail = Object.entries(rest).map(([k, v]) => `${k} ${Array.isArray(v) ? `${v.length} points` : String(v)}`).join(', ')
  return `${what} on ${stemId}${detail ? ` (${detail})` : ''}`
}

export function EditLog({ log }: { log: EditAction[] }) {
  if (log.length === 0) return <p className="text-xs text-subtle">Edits made from the command bar show up here.</p>
  return (
    <ol className="space-y-1.5 text-xs">
      {log.map((a, i) => (
        <li key={i} className="flex gap-2 rounded-lg bg-surface-2 px-2 py-1.5 text-fg-2">
          <span className="w-4 shrink-0 text-right font-medium text-accent">{i + 1}</span>
          <span className="min-w-0 break-words">{describe(a)}</span>
        </li>
      ))}
    </ol>
  )
}
