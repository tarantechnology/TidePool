import { useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import type { AudioEngine } from '../audio/engine'
import type { BackendConfig, SavedStem } from '../api/backend'
import { FEATURES } from '../features'
import { useEngineTime } from '../hooks/useEngineTime'
import * as api from '../moonpull/api'
import type { Transcript } from '../moonpull/api'
import {
  applyGainPull, cleanMoments, describe, echoGain, echoTimes, findPhrase, isGain, loudMoments, MAX_MOMENTS, newRule, pickLayer,
  RECIPES, RESPONSES, responseFor, responseKey, ruleState, signature,
} from '../moonpull/rules'
import type { Moment, Recipe, Rule } from '../moonpull/rules'
import type { Keyframe, Stem, TrackState } from '../types'

export type WithLayers = <T>(label: string, work: (config: BackendConfig, projectId: string, saved: SavedStem[]) => Promise<T>, reload: boolean) => Promise<T>

const fmt = (t: number) => t.toFixed(1)
const field = 'w-full rounded-lg border border-line bg-surface px-2 py-1.5 text-xs text-fg placeholder:text-subtle focus:border-accent/50 focus:outline-none focus:ring-2 focus:ring-accent/15'
const primary = 'rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-[#1b62b3] disabled:opacity-40'
const secondary = 'rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs text-fg transition-colors hover:border-line-strong hover:bg-surface-2 disabled:opacity-40'
const quiet = 'rounded-md px-1.5 py-0.5 text-[11px] font-medium text-accent transition-colors hover:bg-accent-soft disabled:text-subtle disabled:hover:bg-transparent'

export function MoonIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M13.5 10.2A6 6 0 015.8 2.5a6 6 0 107.7 7.7z" fill="currentColor" fillOpacity="0.18" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  )
}

/** Lets a reload's new tracks reach the ref the panel reads before the next step. */
const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))

const sameCurve = (a: Keyframe[], b: Keyframe[]) =>
  a.length === b.length && a.every((k, i) => Math.abs(k.t - b[i].t) < 2e-3 && Math.abs(k.value - b[i].value) < 1e-3)

/* ------------------------------------------------------------------ Motif strip */

/**
 * One row per rule on top of the Tide chart, marking every moment the rule holds. A mark glows
 * while the playhead is inside it, so the pattern is visible as the clip plays.
 */
export function MotifStrip({ rules, stems, engine, duration, onOpen }: { rules: Rule[]; stems: Stem[]; engine: AudioEngine; duration: number; onOpen: (id: string) => void }) {
  const t = useEngineTime(engine)
  if (rules.length === 0 || duration <= 0) return null
  return (
    <div className="max-h-[94px] shrink-0 overflow-y-auto border-b border-line bg-gradient-to-b from-accent-soft/60 to-surface py-0.5" aria-label={`${FEATURES.symbolism} rules`}>
      {rules.map((rule) => {
        const state = ruleState(rule)
        const moments = cleanMoments(rule.moments, duration)
        return (
          <div key={rule.id} className="grid h-[22px] grid-cols-[15rem_minmax(0,1fr)]">
            <button type="button" onClick={() => onOpen(rule.id)} title={describe(rule, stems)}
              className="flex min-w-0 items-center gap-2 border-r border-line px-4 text-left text-[11px] text-fg-2 transition-colors hover:bg-accent-soft">
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: rule.color }} />
              <span className="truncate font-medium">{rule.meaning}</span>
              <span className={`ml-auto shrink-0 text-[10px] ${state === 'applied' ? 'text-accent' : 'text-amber-700'}`}>
                {state === 'applied' ? `${moments.length}×` : state === 'changed' ? 'edited' : 'not applied'}
              </span>
            </button>
            <div className="relative cursor-pointer overflow-hidden" onPointerDown={(e) => {
              if (e.target !== e.currentTarget || e.button !== 0) return
              const box = e.currentTarget.getBoundingClientRect()
              engine.seek(Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)) * duration)
            }}>
              {moments.map((m, i) => {
                const live = t >= m.start && t <= m.end
                return (
                  <button key={i} type="button" aria-label={`${rule.meaning} at ${fmt(m.start)} seconds${m.note ? `: ${m.note}` : ''}`}
                    title={`${fmt(m.start)} to ${fmt(m.end)} s${m.note ? ` · ${m.note}` : ''}`}
                    onClick={() => engine.seek(m.start)}
                    className={`motif-mark absolute top-[4px] h-[14px] min-w-1 rounded-full transition-[box-shadow,opacity] duration-200 ${state === 'applied' ? '' : 'opacity-45'} ${live ? 'motif-live' : ''}`}
                    style={{ left: `${(m.start / duration) * 100}%`, width: `${((m.end - m.start) / duration) * 100}%`, background: rule.color, ['--motif' as string]: rule.color } as CSSProperties} />
                )
              })}
              <div className="pointer-events-none absolute top-0 bottom-0 w-px bg-accent/60" style={{ left: `${(t / duration) * 100}%` }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ Panel */

interface PanelProps {
  rules: Rule[]
  setRules: (update: (rules: Rule[]) => Rule[]) => void
  selectedId: string | null
  onSelect: (id: string | null) => void
  stems: Stem[]
  getTracks: () => TrackState[]
  engine: AudioEngine
  duration: number
  /** Present when the layers are a separated project, so the server can change their audio. */
  withLayers: WithLayers | null
  onAutomation: (stemId: string, keyframes: Keyframe[]) => void
  onClose: () => void
}

/** Transcripts by project layer and file, so reopening the panel does not read the lines again. */
const transcripts = new Map<string, Transcript>()

export function MoonPullPanel(props: PanelProps) {
  const { rules, setRules, selectedId, onSelect, stems, onClose } = props
  const selected = rules.find((r) => r.id === selectedId) ?? null
  const [adding, setAdding] = useState(rules.length === 0)

  const create = (recipe?: Recipe) => {
    const rule = newRule(stems, rules.length, recipe)
    if (recipe?.source === 'loud' && recipe.loudLayer) {
      const src = pickLayer(stems, recipe.loudLayer)
      const s = props.engine.getSamples(src)
      if (s) rule.moments = loudMoments(s.data, s.sampleRate, s.startOffset, props.duration)
    }
    setRules((rs) => [...rs, rule])
    onSelect(rule.id)
    setAdding(false)
  }

  return (
    <section className="shadow-card flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-line bg-surface" aria-label={FEATURES.symbolism}>
      <header className="flex items-start gap-2 border-b border-line px-4 pt-3 pb-2.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-sm font-semibold text-fg"><span className="text-accent"><MoonIcon size={16} /></span>{FEATURES.symbolism}</div>
          <p className="mt-0.5 text-[11px] leading-snug text-muted">Give a sound a meaning and it keeps it for the whole clip, the way the moon pulls every tide the same way.</p>
        </div>
        <button type="button" onClick={onClose} aria-label={`Close ${FEATURES.symbolism}`} className="-mr-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted hover:bg-tint/[0.06] hover:text-fg">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {rules.length > 0 && (
          <ul className="border-b border-line py-1">
            {rules.map((r) => {
              const state = ruleState(r)
              return (
                <li key={r.id}>
                  <button type="button" onClick={() => { onSelect(r.id === selectedId ? null : r.id); setAdding(false) }} aria-expanded={r.id === selectedId}
                    className={`flex w-full items-center gap-2 px-4 py-1.5 text-left text-xs transition-colors ${r.id === selectedId ? 'bg-accent-soft' : 'hover:bg-surface-2'}`}>
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: r.color }} />
                    <span className="min-w-0 flex-1 truncate font-medium text-fg">{r.meaning}</span>
                    <span className={`shrink-0 rounded-full px-1.5 py-px text-[10px] ${state === 'applied' ? 'bg-accent-soft text-accent' : 'bg-amber-50 text-amber-800'}`}>
                      {state === 'applied' ? 'Holding' : state === 'changed' ? 'Edited' : 'Draft'}
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}

        {selected && !adding ? (
          <RuleEditor key={selected.id} rule={selected} {...props} />
        ) : adding || rules.length === 0 ? (
          <Recipes stems={stems} onPick={create} />
        ) : (
          <p className="px-4 py-4 text-xs text-muted">Pick a rule to edit it, or add another.</p>
        )}
      </div>

      {!adding && rules.length > 0 && (
        <footer className="border-t border-line px-4 py-2.5">
          <button type="button" className={secondary + ' w-full'} onClick={() => { setAdding(true); onSelect(null) }}>Add a rule</button>
        </footer>
      )}
    </section>
  )
}

function Recipes({ stems, onPick }: { stems: Stem[]; onPick: (r?: Recipe) => void }) {
  return (
    <div className="px-4 py-3">
      <p className="mb-2 text-[11px] font-medium text-fg-2">Start from an idea</p>
      <ul className="space-y-1">
        {RECIPES.map((r) => (
          <li key={r.id}>
            <button type="button" onClick={() => onPick(r)} className="group w-full rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-surface-2">
              <span className="block text-xs font-medium text-fg group-hover:text-accent">{r.title}</span>
              <span className="block text-[11px] leading-snug text-muted">{r.example}</span>
              <span className="mt-0.5 block text-[10px] text-subtle">Layer: {stems.find((s) => s.id === pickLayer(stems, r.layer))?.label ?? 'none yet'}</span>
            </button>
          </li>
        ))}
      </ul>
      <button type="button" className={quiet + ' mt-2'} onClick={() => onPick()}>Start a blank rule</button>
    </div>
  )
}

/* ------------------------------------------------------------------ One rule */

function RuleEditor({ rule, rules, setRules, onSelect, stems, getTracks, engine, duration, withLayers, onAutomation }: PanelProps & { rule: Rule }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [source, setSource] = useState<'lines' | 'loud' | null>(null)
  const update = (patch: Partial<Rule>) => setRules((rs) => rs.map((r) => (r.id === rule.id ? { ...r, ...patch } : r)))
  const moments = useMemo(() => cleanMoments(rule.moments, duration), [rule.moments, duration])
  const r = rule.response
  const key = responseKey(r)
  const needsServer = !isGain(r)
  const state = ruleState(rule)
  const target = stems.find((s) => s.id === rule.targetStemId)

  const setMoments = (next: Moment[]) => update({ moments: cleanMoments(next, duration).slice(0, MAX_MOMENTS) })
  const mark = () => {
    const t = engine.getTime()
    setMoments([...rule.moments, { start: t, end: Math.min(duration, t + 1.5) }])
  }

  /** Take this rule's earlier result off the layer. Returns false when it could not be. */
  const takeOffGain = (): boolean => {
    const a = rule.applied
    if (!a || a.kind !== 'gain') return true
    const track = getTracks().find((t) => t.stemId === a.stemId)
    if (!track) return true
    if (sameCurve(track.automation, a.after)) { onAutomation(a.stemId, a.before); return true }
    if (sameCurve(track.automation, a.before)) return true
    setError(`The volume curve of ${stems.find((s) => s.id === a.stemId)?.label ?? 'that layer'} was edited after this rule was applied, so it was left as it is. Undo those edits or clear its curve first.`)
    return false
  }

  const run = async (label: string, work: () => Promise<void>) => {
    setBusy(label); setError(''); setNote('')
    try { await work() } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(null) }
  }

  const apply = () => run(state === 'draft' ? 'Applying the rule…' : 'Applying the rule again…', async () => {
    if (!target) throw new Error('Choose the layer this rule changes.')
    if (moments.length === 0) throw new Error('Add at least one moment first.')
    const sig = signature({ ...rule, moments })
    if (isGain(r)) {
      if (rule.applied?.kind === 'audio') { await removeAudio(); await nextFrame() }
      if (!takeOffGain()) return
      const before = getTracks().find((t) => t.stemId === target.id)?.automation ?? []
      // Taking a different layer's result off does not touch this one; the same layer goes back to `before` above.
      const base = rule.applied?.kind === 'gain' && rule.applied.stemId === target.id ? rule.applied.before : before
      const result = applyGainPull(base, moments, r, duration)
      if (!result.ok) throw new Error(result.error)
      onAutomation(target.id, result.keyframes)
      update({ moments, applied: { kind: 'gain', stemId: target.id, before: base, after: result.keyframes, signature: sig } })
      setNote(`${target.label} ${r.kind === 'swell' ? 'swells' : 'hushes'} at ${moments.length} moment${moments.length > 1 ? 's' : ''}. Undo with ⌘Z.`)
      return
    }
    if (!withLayers) throw new Error('This response changes the audio itself, so it needs separated project layers. Separate the clip first.')
    if (!takeOffGain()) return
    const previous = rule.applied
    if (previous?.kind === 'audio') assertNewest(previous)
    const file = await withLayers(`Moon pull: ${rule.meaning}`, async (config, pid) => {
      if (previous?.kind === 'audio') await api.revert(config, pid, previous.stemId, previous.file)
      if (r.kind === 'effect') {
        return (await api.applyEffect(config, pid, target.id, { effect: r.effect, ranges: moments.map((m) => [m.start, m.end]), amount: r.amount, rule: rule.meaning })).moonpull_file
      }
      if (r.kind === 'echo') {
        return (await api.applyEcho(config, pid, target.id, { source_stem_id: r.sourceStemId, source_range: r.source, times: echoTimes(moments, r.anchor), gain: echoGain(r.amount), rule: rule.meaning })).moonpull_file
      }
      throw new Error('Unknown response')
    }, true)
    update({ moments, applied: { kind: 'audio', stemId: target.id, file, signature: sig, at: Date.now() } })
    setNote(`Applied at ${moments.length} moment${moments.length > 1 ? 's' : ''}. Remove it here to get the earlier audio back.`)
  })

  const removeAudio = async () => {
    const a = rule.applied
    if (!a || a.kind !== 'audio') return
    if (!withLayers) throw new Error('Open the project to remove this rule from its audio.')
    assertNewest(a)
    await withLayers(`Moon pull: removing ${rule.meaning}`, (config, pid) => api.revert(config, pid, a.stemId, a.file), true)
  }

  /** Audio changes stack, so only the newest one on a layer can come off. */
  const assertNewest = (a: { stemId: string; at: number }) => {
    const later = rules.filter((x) => x.id !== rule.id && x.applied?.kind === 'audio' && x.applied.stemId === a.stemId && x.applied.at > a.at)
    if (later.length > 0) throw new Error(`“${later.map((x) => x.meaning).join('”, “')}” changed this layer after this rule. Remove ${later.length > 1 ? 'those' : 'it'} first, then come back to this one.`)
  }

  const remove = () => run('Removing the rule from the audio…', async () => {
    if (rule.applied?.kind === 'audio') await removeAudio()
    else if (!takeOffGain()) return
    update({ applied: undefined })
    setNote('Removed. The layer is back to how it was before this rule.')
  })

  const discard = () => run('Deleting the rule…', async () => {
    if (rule.applied?.kind === 'audio') await removeAudio()
    else if (!takeOffGain()) return
    setRules((rs) => rs.filter((x) => x.id !== rule.id))
    onSelect(null)
  })

  return (
    <div className="space-y-4 px-4 py-3 text-xs">
      <label className="block">
        <span className="mb-1 block text-[11px] font-medium text-fg-2">Whenever</span>
        <input className={field + ' text-sm font-medium'} value={rule.meaning} maxLength={60} placeholder="a character lies"
          onChange={(e) => update({ meaning: e.target.value })} />
      </label>

      <fieldset className="space-y-2">
        <legend className="mb-1 text-[11px] font-medium text-fg-2">the sound</legend>
        <select className={field} value={rule.targetStemId} aria-label="Layer this rule changes"
          onChange={(e) => update({ targetStemId: e.target.value, response: r.kind === 'echo' && r.sourceStemId === rule.targetStemId ? { ...r, sourceStemId: e.target.value } : r })}>
          {!target && <option value={rule.targetStemId}>Choose a layer</option>}
          {stems.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
        <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="What the sound does">
          {RESPONSES.map((x) => (
            <button key={x.kind} type="button" role="radio" aria-checked={key === x.kind}
              onClick={() => update({ response: responseFor(x.kind, r, rule.targetStemId) })}
              className={`rounded-full border px-2.5 py-1 text-[11px] transition-colors ${key === x.kind ? 'border-accent bg-accent text-white' : 'border-line bg-surface text-fg-2 hover:border-line-strong'}`}>
              {x.label}
            </button>
          ))}
        </div>
        {r.kind === 'echo' && <EchoSource rule={rule} stems={stems} engine={engine} withLayers={withLayers} update={update} />}
        <label className="flex items-center gap-2">
          <span className="w-16 shrink-0 text-[11px] text-muted">Strength</span>
          <input type="range" min={0.1} max={1} step={0.05} value={r.amount} aria-label="Strength"
            onChange={(e) => update({ response: { ...r, amount: Number(e.target.value) } })}
            className="fader flex-1" style={{ ['--pos' as string]: `${((r.amount - 0.1) / 0.9) * 100}%` } as CSSProperties} />
          <span className="w-8 text-right font-mono text-[10px] text-muted">{Math.round(r.amount * 100)}</span>
        </label>
        {needsServer && !withLayers && <p className="text-[11px] text-amber-800">Changes the audio itself, so it works on separated project layers.</p>}
      </fieldset>

      <div>
        <div className="mb-1 flex items-center gap-1">
          <span className="text-[11px] font-medium text-fg-2">Moments</span>
          <span className="text-[10px] text-subtle">{moments.length}</span>
          <button type="button" className={quiet + ' ml-auto'} disabled={moments.length === 0} onClick={() => setMoments([])}>Clear</button>
        </div>
        <div className="mb-2 flex flex-wrap gap-1">
          <button type="button" className={secondary} onClick={mark} title="Adds 1.5 seconds from the playhead. Press M while the panel is open.">Mark the playhead</button>
          <button type="button" className={secondary} aria-pressed={source === 'lines'} onClick={() => setSource(source === 'lines' ? null : 'lines')}>From lines…</button>
          <button type="button" className={secondary} aria-pressed={source === 'loud'} onClick={() => setSource(source === 'loud' ? null : 'loud')}>When a layer is loud…</button>
        </div>
        <MarkKey onMark={mark} />
        {source === 'lines' && <LinesPicker stems={stems} withLayers={withLayers} onAdd={(ms) => setMoments([...rule.moments, ...ms])} />}
        {source === 'loud' && <LoudPicker stems={stems} engine={engine} duration={duration} onAdd={(ms) => setMoments(ms)} />}
        {moments.length === 0 ? (
          <p className="rounded-lg bg-surface-2 px-2.5 py-2 text-[11px] text-muted">No moments yet. Play the clip and mark each time the meaning happens, or take them from what is said.</p>
        ) : (
          <ol className="divide-y divide-line rounded-lg border border-line">
            {moments.map((m, i) => (
              <li key={i} className="flex items-center gap-1.5 px-2 py-1">
                <button type="button" className="font-mono text-[11px] text-accent hover:underline" onClick={() => engine.seek(m.start)} title="Jump there">{fmt(m.start)}</button>
                <span className="text-subtle">to</span>
                <input type="number" step={0.1} min={m.start} max={duration} value={Number(m.end.toFixed(2))} aria-label={`Moment ${i + 1} end`}
                  className="w-14 rounded border border-transparent bg-transparent px-1 font-mono text-[11px] hover:border-line focus:border-accent/50 focus:outline-none"
                  onChange={(e) => setMoments(moments.map((x, j) => (j === i ? { ...x, end: Number(e.target.value) } : x)))} />
                <span className="min-w-0 flex-1 truncate text-[11px] text-muted" title={m.note}>{m.note}</span>
                <button type="button" aria-label={`Remove moment ${i + 1}`} className="rounded px-1 text-subtle hover:bg-surface-2 hover:text-fg" onClick={() => setMoments(moments.filter((_, j) => j !== i))}>×</button>
              </li>
            ))}
          </ol>
        )}
      </div>

      <div className="rounded-xl bg-accent-soft/70 px-3 py-2.5">
        <p className="text-[11px] leading-snug text-fg-2">{describe(rule, stems)}, {moments.length === 0 ? 'once you add moments.' : `${moments.length} time${moments.length > 1 ? 's' : ''} in this clip.`}</p>
        <div className="mt-2 flex flex-wrap items-center gap-1.5 whitespace-nowrap">
          <button type="button" className={primary} disabled={Boolean(busy) || moments.length === 0 || state === 'applied' || (needsServer && !withLayers)} onClick={() => void apply()}>
            {state === 'draft' ? 'Apply rule' : state === 'changed' ? 'Apply changes' : 'Holding'}
          </button>
          {rule.applied && <button type="button" className={secondary} disabled={Boolean(busy)} onClick={() => void remove()} title="Take this rule off its layer">Remove</button>}
          <button type="button" className={quiet + ' ml-auto text-red-700 hover:bg-red-50'} disabled={Boolean(busy)} onClick={() => void discard()}>Delete</button>
        </div>
        {busy && <p role="status" className="mt-2 flex items-center gap-1.5 text-[11px] text-accent"><span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent" />{busy}</p>}
        {note && !busy && <p role="status" className="mt-2 text-[11px] text-fg-2">{note}</p>}
        {error && <p role="alert" className="mt-2 text-[11px] text-red-700">{error}</p>}
      </div>
    </div>
  )
}

/** While the panel is open, M marks the playhead, so moments can be tapped in as the clip plays. */
function MarkKey({ onMark }: { onMark: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== 'm' || e.metaKey || e.ctrlKey || e.altKey) return
      const el = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) || el.isContentEditable) return
      e.preventDefault(); onMark()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onMark])
  return null
}

function EchoSource({ rule, stems, engine, withLayers, update }: { rule: Rule; stems: Stem[]; engine: AudioEngine; withLayers: WithLayers | null; update: (p: Partial<Rule>) => void }) {
  const r = rule.response
  // Remembered with the layer it was found on, so picking another layer forgets it.
  const [lookup, setLookup] = useState<{ id: string; start: number; end: number } | null>(null)
  const [error, setError] = useState('')
  if (r.kind !== 'echo') return null
  const found = lookup && lookup.id === r.sourceStemId ? lookup : null
  const find = () => {
    if (!withLayers) return
    setError('')
    void withLayers('Finding the loudest sound…', (c, pid) => api.findMotif(c, pid, r.sourceStemId), false)
      .then((m) => setLookup({ id: r.sourceStemId, ...m })).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }
  const range = r.source ? { start: r.source[0], end: r.source[1] } : found
  return (
    <div className="space-y-1.5 rounded-lg border border-line px-2.5 py-2">
      <label className="flex items-center gap-2">
        <span className="w-16 shrink-0 text-[11px] text-muted">Sound from</span>
        <select className={field} value={r.sourceStemId} aria-label="Layer the motif sound comes from"
          onChange={(e) => update({ response: { ...r, sourceStemId: e.target.value, source: null } })}>
          {stems.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      </label>
      <p className="text-[11px] text-muted">
        {r.source ? 'The 2 seconds you chose' : 'Its loudest 2 seconds'}
        {range ? <>, <button type="button" className="font-mono text-accent hover:underline" onClick={() => engine.seek(range.start)} title="Jump there">{fmt(range.start)} to {fmt(range.end)} s</button></> : '. '}
        {!range && withLayers && <button type="button" className={quiet} onClick={find}>Find it</button>}
      </p>
      {error && <p className="text-[11px] text-red-700">{error}</p>}
      <div className="flex flex-wrap items-center gap-1">
        <button type="button" className={quiet} onClick={() => { const t = engine.getTime(); update({ response: { ...r, source: [Math.round(t * 100) / 100, Math.round((t + 2) * 100) / 100] } }) }}>Use 2 s from the playhead</button>
        {r.source && <button type="button" className={quiet} onClick={() => update({ response: { ...r, source: null } })}>Use the loudest</button>}
      </div>
      <div className="flex rounded-full border border-line bg-surface p-0.5 text-[10px]" role="group" aria-label="When the echo lands">
        {(['start', 'end'] as const).map((a) => (
          <button key={a} type="button" aria-pressed={r.anchor === a} onClick={() => update({ response: { ...r, anchor: a } })}
            className={`flex-1 rounded-full px-2 py-0.5 font-medium transition-colors ${r.anchor === a ? 'bg-accent text-white' : 'text-muted hover:text-fg'}`}>
            {a === 'start' ? 'As the moment starts' : 'Right after it ends'}
          </button>
        ))}
      </div>
    </div>
  )
}

function LinesPicker({ stems, withLayers, onAdd }: { stems: Stem[]; withLayers: WithLayers | null; onAdd: (m: Moment[]) => void }) {
  const [layer, setLayer] = useState(() => pickLayer(stems, ['talk', 'voice', 'dialog', 'speech', 'people', 'speak']))
  const [text, setText] = useState<Transcript | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [phrase, setPhrase] = useState('')
  const [picked, setPicked] = useState<Set<number>>(new Set())
  const read = () => {
    if (!withLayers) return
    setBusy(true); setError('')
    void withLayers('Reading the lines…', async (c, pid, saved) => {
      const row = saved.find((s) => s.id === layer)
      const k = `${pid}:${layer}:${row?.revision ?? 0}`
      const cached = transcripts.get(k)
      if (cached) return cached
      const t = await api.transcribe(c, pid, layer)
      transcripts.set(k, t)
      return t
    }, false).then((t) => { setText(t); setPicked(new Set()) }).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e))).finally(() => setBusy(false))
  }
  if (!withLayers) return <p className="mb-2 text-[11px] text-amber-800">Reading lines needs separated project layers.</p>
  const lines = text?.lines ?? []
  const phraseHits = text && phrase.trim() ? findPhrase(text.words.length > 0 ? text.words : text.lines, phrase) : []
  return (
    <div className="mb-2 space-y-1.5 rounded-lg border border-line p-2">
      <div className="flex gap-1.5">
        <select className={field} value={layer} onChange={(e) => { setLayer(e.target.value); setText(null) }} aria-label="Layer with the dialogue">
          {stems.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
        <button type="button" className={secondary + ' shrink-0'} disabled={busy} onClick={read}>{busy ? 'Reading…' : text ? 'Read again' : 'Read the lines'}</button>
      </div>
      {error && <p className="text-[11px] text-red-700">{error}</p>}
      {text && (lines.length === 0 ? <p className="text-[11px] text-muted">No lines were heard on this layer.</p> : (
        <>
          <ul className="max-h-44 space-y-px overflow-y-auto">
            {lines.map((l, i) => (
              <li key={i}>
                <label className="flex cursor-pointer items-start gap-1.5 rounded px-1 py-0.5 hover:bg-surface-2">
                  <input type="checkbox" className="mt-0.5 accent-[#2270c8]" checked={picked.has(i)} onChange={() => setPicked((p) => { const n = new Set(p); if (n.has(i)) n.delete(i); else n.add(i); return n })} />
                  <span className="w-9 shrink-0 font-mono text-[10px] text-subtle">{fmt(l.start)}</span>
                  <span className="text-[11px] text-fg">{l.text}</span>
                </label>
              </li>
            ))}
          </ul>
          <div className="flex gap-1.5">
            <button type="button" className={primary} disabled={picked.size === 0} onClick={() => { onAdd(lines.filter((_, i) => picked.has(i)).map((l) => ({ start: l.start, end: l.end, note: `“${l.text}”` }))); setPicked(new Set()) }}>
              Add {picked.size || ''} line{picked.size === 1 ? '' : 's'}
            </button>
            <button type="button" className={quiet} onClick={() => onAdd(lines.map((l) => ({ start: l.start, end: l.end, note: `“${l.text}”` })))}>Every line</button>
          </div>
          <div className="flex gap-1.5">
            <input className={field} value={phrase} placeholder="Every time someone says…" onChange={(e) => setPhrase(e.target.value)} aria-label="Phrase" />
            <button type="button" className={secondary + ' shrink-0'} disabled={phraseHits.length === 0} onClick={() => onAdd(phraseHits)}>Add {phraseHits.length || ''}</button>
          </div>
        </>
      ))}
    </div>
  )
}

function LoudPicker({ stems, engine, duration, onAdd }: { stems: Stem[]; engine: AudioEngine; duration: number; onAdd: (m: Moment[]) => void }) {
  const [layer, setLayer] = useState(() => pickLayer(stems, ['talk', 'voice', 'dialog', 'speech', 'people']))
  const [threshold, setThreshold] = useState(0.4)
  const found = useMemo(() => {
    const samples = engine.getSamples(layer)
    return samples ? loudMoments(samples.data, samples.sampleRate, samples.startOffset, duration, { threshold }) : []
  }, [engine, layer, duration, threshold])
  return (
    <div className="mb-2 space-y-1.5 rounded-lg border border-line p-2">
      <select className={field} value={layer} onChange={(e) => setLayer(e.target.value)} aria-label="Layer to listen to">
        {stems.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
      </select>
      <label className="flex items-center gap-2">
        <span className="w-16 shrink-0 text-[11px] text-muted">Only loud</span>
        <input type="range" min={0.15} max={0.85} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} aria-label="How loud counts"
          className="fader flex-1" style={{ ['--pos' as string]: `${((threshold - 0.15) / 0.7) * 100}%` } as CSSProperties} />
      </label>
      <div className="relative h-3 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
        {found.map((m, i) => <span key={i} className="absolute top-0 bottom-0 rounded-full bg-accent/70" style={{ left: `${(m.start / duration) * 100}%`, width: `${((m.end - m.start) / duration) * 100}%` }} />)}
      </div>
      <button type="button" className={primary} disabled={found.length === 0} onClick={() => onAdd(found)}>Use these {found.length} moments</button>
      <p className="text-[10px] text-subtle">Replaces the list. Hears loudness only, so music on the same layer counts too.</p>
    </div>
  )
}
