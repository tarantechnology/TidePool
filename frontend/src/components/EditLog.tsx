import type { EditAction } from '../types'

export function EditLog({ log }: { log: EditAction[] }) {
  return (
    <aside className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-line bg-ink-900/60 backdrop-blur">
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-500">Edit list</span>
        <span className="rounded-full border border-line px-1.5 font-mono text-[10px] text-zinc-400">{log.length}</span>
      </div>
      <div className="min-h-0 flex-1 space-y-2 overflow-auto p-3">
        {log.length === 0 ? (
          <div className="font-mono text-[11px] text-zinc-600">Edits from the prompt bar appear here.</div>
        ) : (
          log.map((a, i) => {
            const { type, stemId, ...rest } = a
            return (
              <div key={i} className="rounded-lg border border-line bg-white/[0.02] px-2.5 py-2 font-mono text-[11px]">
                <div className="flex items-center gap-2">
                  <span className="text-accent">{type}</span>
                  <span className="text-zinc-500">{stemId}</span>
                </div>
                <div className="mt-1 break-all leading-relaxed text-zinc-500">{JSON.stringify(rest)}</div>
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}
