import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { FEATURES } from '../features'
import { FeatureTag } from './FeatureTag'

interface Entry {
  prompt: string
  message: string
  ok: boolean
}

interface SpeechResultEvent { results: ArrayLike<{isFinal:boolean; [index:number]:{transcript:string}}> }
interface SpeechRecognizer {
  lang: string
  interimResults: boolean
  onresult: ((event:SpeechResultEvent) => void) | null
  onerror: ((event:{error:string}) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}
type SpeechConstructor = new () => SpeechRecognizer

interface Props {
  disabled: boolean
  /** Why the bar is disabled, shown as helper text and as the tooltip. */
  disabledReason?: string
  onRun: (text: string) => Promise<{ message: string; ok: boolean }>
  /**
   * "Split into tracks", while the project can be split. `suggested` is set when
   * the audio service thinks the upload is a song, and puts the action first.
   */
  split?: { suggested: string | null; onSplit: () => void; busy?: boolean } | null
}

export function PromptBar({ disabled, disabledReason = '', onRun, split = null }: Props) {
  const [text, setText] = useState('')
  const [last, setLast] = useState<Entry | null>(null)
  // The reply describes the moment it was made, so it fades once that moment has passed.
  useEffect(() => {
    if (!last) return
    const timer = window.setTimeout(() => setLast(null), last.ok ? 6000 : 12000)
    return () => window.clearTimeout(timer)
  }, [last])
  const [pending, setPending] = useState(false)
  const [listening, setListening] = useState(false)
  const recognition = useRef<SpeechRecognizer|null>(null)
  const Speech = (window as unknown as {SpeechRecognition?:SpeechConstructor; webkitSpeechRecognition?:SpeechConstructor}).SpeechRecognition ?? (window as unknown as {webkitSpeechRecognition?:SpeechConstructor}).webkitSpeechRecognition
  useEffect(() => () => { if (recognition.current) { recognition.current.onresult=null; recognition.current.onerror=null; recognition.current.onend=null; recognition.current.stop(); recognition.current=null } }, [])
  const input = useRef<HTMLInputElement>(null)
  const reason = disabled ? disabledReason : ''

  // A disabled input loses focus, so hand it back when the run finishes.
  const wasPending = useRef(false)
  useEffect(() => {
    if (wasPending.current && !pending) input.current?.focus()
    wasPending.current = pending
  }, [pending])

  const submitPrompt = async (prompt: string) => {
    prompt = prompt.trim()
    if (!prompt || pending) return
    setPending(true)
    try {
      const res = await onRun(prompt)
      setLast({ prompt, ...res })
      setText('')
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setLast({ prompt, message, ok: false })
    } finally {
      setPending(false)
    }
  }
  const submit = (e: FormEvent) => { e.preventDefault(); void submitPrompt(text) }
  const startListening = () => {
    if (!Speech || pending || listening) return
    const listener=new Speech(); recognition.current=listener
    listener.lang='en-US'; listener.interimResults=false
    listener.onresult = event => {
      const result=event.results[0]
      if (!result?.isFinal) return
      recognition.current=null; setListening(false)
      void submitPrompt(result[0].transcript)
    }
    listener.onerror = event => { recognition.current=null; setListening(false); setLast({prompt:'Voice input',message:`Speech recognition failed (${event.error}). Check your microphone permission or type your request.`,ok:false}) }
    listener.onend = () => { if (recognition.current === listener) { recognition.current=null; setListening(false) } }
    try { listener.start(); setListening(true) } catch { recognition.current=null; setListening(false); setLast({prompt:'Voice input',message:'Could not start the microphone. Check permission and try again.',ok:false}) }
  }

  return (
    <div className="relative z-10 shrink-0 px-1 pt-2 pb-3">
      <div className="mx-auto max-w-3xl">
        {split?.suggested && !last && (
          <p className="rise-in mb-2 flex items-center gap-2 px-1 text-[11px] text-muted" role="status">
            <span>{split.suggested}</span>
            <button type="button" onClick={split.onSplit} disabled={disabled || pending || split.busy} className="font-medium text-accent hover:underline disabled:opacity-40">
              Split it into one track per instrument
            </button>
          </p>
        )}
        {last && (
          <p className={`rise-in mb-2 px-1 text-[11px] ${last.ok ? 'text-muted' : 'text-amber-700'}`} role="status">
            <span className="text-subtle">{last.prompt}</span>
            <span className="mx-1.5 text-subtle">·</span>
            {last.message}
          </p>
        )}
        <form
          onSubmit={submit}
          title={reason}
          className="flex items-center gap-3 rounded-2xl border border-line-strong bg-surface py-2 pr-2 pl-4 shadow-[0_12px_40px_-16px_rgb(29_90_158/0.35)] transition-all focus-within:border-accent/50 focus-within:shadow-[0_0_0_4px_rgb(34_112_200/0.12),0_12px_40px_-16px_rgb(29_90_158/0.45)]"
        >
          <FeatureTag name={FEATURES.command} className="shrink-0 py-0.5 text-[11px]" />
          <input
            ref={input}
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={disabled || pending}
            placeholder={reason || 'Ask Current, e.g. trim from 3 to 4 seconds'}
            className="h-9 flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-faint disabled:opacity-50"
          />
          {Speech && <button type="button" aria-label={listening ? 'Stop listening' : 'Speak a command'} title={listening ? 'Stop listening' : 'Speak a command'} disabled={disabled || pending} onClick={() => { if (listening) recognition.current?.stop(); else startListening() }} className="flex h-9 shrink-0 items-center rounded-xl border border-line px-3 text-sm text-fg hover:border-line-strong disabled:opacity-40">{listening ? 'Listening…' : '🎙'}</button>}
          {split && (
            <button
              type="button"
              onClick={split.onSplit}
              disabled={disabled || pending || split.busy}
              title="One editable track per instrument: vocals, drums, bass and the rest"
              className={`flex h-9 shrink-0 items-center rounded-xl px-3 text-sm transition-colors disabled:opacity-40 ${split.suggested ? 'bg-accent-soft font-medium text-accent ring-1 ring-accent/40 hover:bg-accent-soft/70' : 'border border-line text-fg hover:border-line-strong hover:bg-surface-2'}`}
            >
              Split into tracks
            </button>
          )}
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
      </div>
    </div>
  )
}
