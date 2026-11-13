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
  onRun: (text: string) => Promise<{ message: string; ok: boolean }>
}

export function PromptBar({ disabled, disabledReason = '', onRun }: Props) {
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
          <div className="mb-2 space-y-0.5 px-1 text-[11px]">
            {history.map((h, i) => (
              <div key={i} className={h.ok ? 'text-muted' : 'text-amber-700'}>
                <span className="text-faint">&gt; {h.prompt}</span>
                <span className="ml-2">{h.message}</span>
              </div>
            ))}
          </div>
        )}
        <form
          onSubmit={submit}
          title={reason}
          className="flex items-center gap-3 rounded-2xl border border-line-strong bg-surface py-2 pr-2 pl-4 shadow-[0_12px_40px_-16px_rgb(29_90_158/0.35)] transition-all focus-within:border-accent/50 focus-within:shadow-[0_0_0_4px_rgb(34_112_200/0.12),0_12px_40px_-16px_rgb(29_90_158/0.45)]"
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
            placeholder={reason || 'Describe a sound to find or an edit to make, e.g. isolate the dog barking'}
            className="h-9 flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-faint disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={disabled || pending || !text.trim()}
            className="flex h-9 items-center gap-2 rounded-xl bg-accent px-3.5 text-sm font-medium text-white transition-all hover:bg-[#1b62b3] disabled:bg-tint/[0.06] disabled:text-faint"
          >
            {pending ? (
              <span className="text-xs">Thinking…</span>
            ) : (
              <>
                Apply
                <kbd className="rounded border border-white/40 px-1 font-mono text-[10px]">&#8629;</kbd>
              </>
            )}
          </button>
        </form>
        {reason && <p className="mt-2 px-1 text-[11px] text-faint">{reason}</p>}
      </div>
    </div>
  )
}
