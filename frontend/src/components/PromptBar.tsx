import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'

interface Entry {
  prompt: string
  message: string
  ok: boolean
}

interface Props {
  disabled: boolean
  /** Why the bar is disabled, shown as helper text and as the tooltip. */
  disabledReason?: string
  agentLabel: string
  onRun: (text: string) => Promise<{ message: string; ok: boolean }>
}

const SUGGESTIONS = [
  'Make the siren louder whenever the argument gets intense',
  'Make the siren louder from 8 to 16 seconds',
  'Mute Speaker B at 12 seconds',
  'Solo Speaker A',
  'Make the siren quieter',
]

export function PromptBar({ disabled, disabledReason = '', agentLabel, onRun }: Props) {
  const [text, setText] = useState('')
  const [history, setHistory] = useState<Entry[]>([])
  const [pending, setPending] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const reason = disabled ? disabledReason : ''

  // A disabled input loses focus, so hand it back when the run finishes.
  const wasPending = useRef(false)
  useEffect(() => {
    if (wasPending.current && !pending) input.current?.focus()
    wasPending.current = pending
  }, [pending])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const prompt = text.trim()
    if (!prompt || pending) return
    setPending(true)
    try {
      const res = await onRun(prompt)
      setHistory((h) => [...h.slice(-2), { prompt, ...res }])
      setText('')
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setHistory((h) => [...h.slice(-2), { prompt, message, ok: false }])
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="relative z-10 shrink-0 px-4 pt-1 pb-4">
      <div className="mx-auto max-w-3xl">
        {history.length > 0 && (
          <div className="mb-2 space-y-0.5 px-1 font-mono text-[11px]">
            {history.map((h, i) => (
              <div key={i} className={h.ok ? 'text-zinc-400' : 'text-amber-300'}>
                <span className="text-zinc-600">&gt; {h.prompt}</span>
                <span className="ml-2">{h.message}</span>
              </div>
            ))}
          </div>
        )}
        <form
          onSubmit={submit}
          title={reason}
          className="flex items-center gap-3 rounded-2xl border border-line-strong bg-ink-900/90 py-2 pr-2 pl-4 shadow-2xl shadow-black/60 backdrop-blur transition-all focus-within:border-accent/50 focus-within:shadow-[0_0_0_4px_rgb(61_255_139/0.08),0_0_48px_-12px_rgb(61_255_139/0.4)]"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="shrink-0 text-accent" aria-hidden="true">
            <path
              d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z"
              fill="currentColor"
            />
          </svg>
          <input
            ref={input}
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={disabled || pending}
            placeholder={reason || 'Describe an edit, e.g. make the siren louder from 8 to 16 seconds'}
            className="h-9 flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={disabled || pending || !text.trim()}
            className="flex h-9 items-center gap-2 rounded-xl bg-accent px-3.5 text-sm font-medium text-black transition-all hover:brightness-110 disabled:bg-white/[0.06] disabled:text-zinc-600"
          >
            {pending ? (
              <span className="font-mono text-xs">Thinking...</span>
            ) : (
              <>
                Apply
                <kbd className="rounded border border-black/20 px-1 font-mono text-[10px]">&#8629;</kbd>
              </>
            )}
          </button>
        </form>
        <div className="mt-2.5 flex items-center gap-3 px-1">
          {/* One row, so the timeline keeps its height. The rest scrolls sideways. */}
          <div className="flex min-w-0 flex-1 gap-1.5 overflow-x-auto [scrollbar-width:none]">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                title={reason}
                disabled={disabled || pending}
                onClick={() => {
                  setText(s)
                  input.current?.focus()
                }}
                className="shrink-0 whitespace-nowrap rounded-full border border-line px-2.5 py-1 text-[11px] text-zinc-400 transition-colors hover:border-line-strong hover:bg-white/[0.04] hover:text-zinc-100 disabled:opacity-40"
              >
                {s}
              </button>
            ))}
          </div>
          <span className="shrink-0 font-mono text-[10px] text-zinc-600">{reason || agentLabel}</span>
        </div>
      </div>
    </div>
  )
}
