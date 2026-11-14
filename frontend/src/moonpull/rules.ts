import { clamp, MAX_KEYFRAMES, normalizeKeyframes, spliceAutomation } from '../edits'
import { normalizeEnvelope, rmsEnvelope, smoothEnvelope } from '../audio/analysis'
import type { Keyframe, Stem } from '../types'

// Moon pull: a rule that ties a sound to a meaning and holds for the whole clip, the way the moon
// pulls every tide the same way. "Whenever a character lies, the thunder answers." A rule is a list
// of moments (when the meaning happens) and one response on one layer (what the sound does).

export type Effect = 'deepen' | 'impact' | 'disorient' | 'muffle'

export type PullResponse =
  /** Layer volume rises around each moment. Stays in the editor as automation. */
  | { kind: 'swell'; amount: number }
  /** Layer volume drops around each moment. Stays in the editor as automation. */
  | { kind: 'hush'; amount: number }
  /** The layer's audio is changed inside each moment, on the server. */
  | { kind: 'effect'; effect: Effect; amount: number }
  /** A motif sound (the loudest stretch of a layer, or a chosen stretch) lands at each moment. */
  | { kind: 'echo'; sourceStemId: string; source: [number, number] | null; anchor: 'start' | 'end'; amount: number }

export interface Moment {
  start: number
  end: number
  /** What happens there, for example the line that was said. */
  note?: string
}

export type Applied =
  | { kind: 'gain'; stemId: string; before: Keyframe[]; after: Keyframe[]; signature: string }
  | { kind: 'audio'; stemId: string; file: string; signature: string; at: number }

export interface Rule {
  id: string
  /** The story meaning, for example "A lie". */
  meaning: string
  color: string
  targetStemId: string
  response: PullResponse
  moments: Moment[]
  applied?: Applied
}

export const RESPONSES: Array<{ kind: PullResponse['kind'] | Effect; label: string; verb: string }> = [
  { kind: 'echo', label: 'Echo a sound', verb: 'answers with' },
  { kind: 'swell', label: 'Swell', verb: 'swells' },
  { kind: 'hush', label: 'Hush', verb: 'hushes' },
  { kind: 'deepen', label: 'Deepen', verb: 'deepens' },
  { kind: 'impact', label: 'Impact', verb: 'hits harder' },
  { kind: 'disorient', label: 'Disorient', verb: 'disorients' },
  { kind: 'muffle', label: 'Muffle', verb: 'muffles' },
]

export const COLORS = ['#2b7bd6', '#14a3a3', '#7c6cf0', '#e0874a', '#d65b8a', '#4aa86a', '#8a9bb0']

export const MAX_MOMENTS = 32

/** Seconds a swell or hush starts before a moment and lingers after it. */
export const GAIN_LEAD = { swell: 0.3, hush: 0.25 }
export const GAIN_TAIL = { swell: 1.2, hush: 0.4 }

export const responseKey = (r: PullResponse): PullResponse['kind'] | Effect => (r.kind === 'effect' ? r.effect : r.kind)
export const isGain = (r: PullResponse): r is Extract<PullResponse, { kind: 'swell' | 'hush' }> => r.kind === 'swell' || r.kind === 'hush'

export function responseFor(key: PullResponse['kind'] | Effect, previous: PullResponse, fallbackSource: string): PullResponse {
  const amount = previous.amount
  if (key === 'swell' || key === 'hush') return { kind: key, amount }
  if (key === 'echo') return previous.kind === 'echo' ? previous : { kind: 'echo', sourceStemId: fallbackSource, source: null, anchor: 'start', amount }
  if (key === 'effect') return previous
  return { kind: 'effect', effect: key, amount }
}

/** What a rule's result depends on. A rule whose signature differs from what was applied is out of date. */
export function signature(rule: Pick<Rule, 'targetStemId' | 'response' | 'moments'>): string {
  return JSON.stringify([rule.targetStemId, rule.response, rule.moments.map((m) => [m.start, m.end])])
}

export type RuleState = 'draft' | 'applied' | 'changed'
export function ruleState(rule: Rule): RuleState {
  if (!rule.applied) return 'draft'
  return rule.applied.signature === signature(rule) ? 'applied' : 'changed'
}

/** Sorted, inside the timeline, at least 50 ms long, and joined where they overlap or touch. */
export function cleanMoments(moments: Moment[], duration: number): Moment[] {
  const sorted = moments
    .filter((m) => Number.isFinite(m.start) && Number.isFinite(m.end))
    .map((m) => ({ ...m, start: clamp(Math.min(m.start, m.end), 0, duration), end: clamp(Math.max(m.start, m.end), 0, duration) }))
    .filter((m) => m.end - m.start >= 0.05)
    .sort((a, b) => a.start - b.start)
  const out: Moment[] = []
  for (const m of sorted) {
    const last = out.at(-1)
    if (last && m.start <= last.end) {
      last.end = Math.max(last.end, m.end)
      if (m.note && m.note !== last.note) last.note = last.note ? `${last.note} / ${m.note}` : m.note
    } else out.push({ ...m })
  }
  return out
}

/** Windows a swell or hush covers: each moment widened by the lead and tail, joined where they meet. */
export function gainWindows(moments: Moment[], kind: 'swell' | 'hush', duration: number): Array<[number, number]> {
  const widened = moments.map((m) => ({ start: m.start - GAIN_LEAD[kind], end: m.end + GAIN_TAIL[kind] }))
  return cleanMoments(widened, duration).map((m) => [m.start, m.end])
}

/** The volume multiplier inside each window. */
export function gainScale(r: { kind: 'swell' | 'hush'; amount: number }): number {
  const a = clamp(r.amount, 0, 1)
  return r.kind === 'swell' ? 1 + a : 1 - 0.9 * a
}

export type GainResult = { ok: true; keyframes: Keyframe[] } | { ok: false; error: string }

/**
 * The layer's curve with the rule on top. Outside the windows it is unchanged; inside, the curve
 * under it is scaled, so earlier automation keeps its shape.
 */
export function applyGainPull(curve: Keyframe[], moments: Moment[], response: { kind: 'swell' | 'hush'; amount: number }, duration: number): GainResult {
  let base = normalizeKeyframes(curve, duration)
  // An empty curve means "1 everywhere"; splicing needs real points at both ends.
  if (base.length === 0) base = [{ t: 0, value: 1 }, { t: duration, value: 1 }]
  const windows = gainWindows(moments, response.kind, duration)
  // Each window costs about four points. When there are too many, hold the change through the
  // shortest gaps first (rain that stays down between two close lines), until the curve fits.
  let out: Keyframe[] = base
  for (const bridge of [0, 0.75, 1.5, 3, 6, 12]) {
    out = base
    for (const [from, to] of joinWindows(windows, bridge)) out = spliceAutomation(out, { scale: gainScale(response) }, from, to, duration)
    if (out.length <= MAX_KEYFRAMES) return { ok: true, keyframes: out }
  }
  return { ok: false, error: `This rule needs ${out.length} volume points on one layer, more than the ${MAX_KEYFRAMES} a layer can hold. Join or remove some moments.` }
}

function joinWindows(windows: Array<[number, number]>, bridge: number): Array<[number, number]> {
  const out: Array<[number, number]> = []
  for (const [a, b] of windows) {
    const last = out.at(-1)
    if (last && a - last[1] < bridge) last[1] = b
    else out.push([a, b])
  }
  return out
}

/** Where an echo lands for each moment. */
export function echoTimes(moments: Moment[], anchor: 'start' | 'end'): number[] {
  return moments.map((m) => Math.round((anchor === 'start' ? m.start : m.end) * 1000) / 1000)
}

/** Echo amount (0..1) to the gain the server mixes the motif at. */
export const echoGain = (amount: number) => Math.round((0.3 + clamp(amount, 0, 1)) * 100) / 100

export interface LoudOptions {
  /** 0..1 of the layer's own loud level. */
  threshold?: number
  /** Gaps shorter than this join two moments. */
  bridge?: number
  /** Moments shorter than this are dropped. */
  minLength?: number
}

/**
 * Moments where a layer is loud, relative to its own loud parts: "whenever someone speaks", "every
 * time the impacts hit". Levels only, so it hears a shout and a door slam the same way.
 */
export function loudMoments(samples: Float32Array, sampleRate: number, startOffset: number, duration: number, opts: LoudOptions = {}): Moment[] {
  const { threshold = 0.4, bridge = 0.6, minLength = 0.3 } = opts
  const hop = 0.05
  const env = normalizeEnvelope(smoothEnvelope(rmsEnvelope(samples, sampleRate, hop), 2))
  const raw: Moment[] = []
  let open = -1
  for (let i = 0; i <= env.length; i++) {
    const v = i < env.length ? env[i] : 0
    // Hysteresis: it opens above the threshold and closes only well below it.
    if (open < 0 && v >= threshold) open = i
    else if (open >= 0 && v < threshold * 0.7) {
      raw.push({ start: startOffset + open * hop, end: startOffset + i * hop })
      open = -1
    }
  }
  const joined: Moment[] = []
  for (const m of raw) {
    const last = joined.at(-1)
    if (last && m.start - last.end < bridge) last.end = m.end
    else joined.push({ ...m })
  }
  return cleanMoments(
    joined.filter((m) => m.end - m.start >= minLength).map((m) => ({ start: round(m.start), end: round(m.end) })),
    duration,
  ).slice(0, MAX_MOMENTS)
}

const round = (v: number) => Math.round(v * 100) / 100

/** Every occurrence of a phrase in timed words, as moments. Case and punctuation do not matter. */
export function findPhrase(words: Array<{ start: number; end: number; text: string }>, phrase: string): Moment[] {
  const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}']+/gu, ' ').trim()
  const target = norm(phrase).split(' ').filter(Boolean)
  if (target.length === 0) return []
  const tokens = words.map((w) => ({ ...w, n: norm(w.text) }))
  const out: Moment[] = []
  for (let i = 0; i + target.length <= tokens.length; i++) {
    if (target.every((t, k) => tokens[i + k].n === t)) {
      out.push({ start: tokens[i].start, end: tokens[i + target.length - 1].end, note: `“${phrase.trim()}”` })
    }
  }
  return out
}

/** First layer whose name contains one of the words, else the first layer. */
export function pickLayer(stems: Stem[], words: string[]): string {
  const hit = stems.find((s) => words.some((w) => s.label.toLowerCase().includes(w)))
  return (hit ?? stems[0])?.id ?? ''
}

export interface Recipe {
  id: string
  meaning: string
  title: string
  /** How the rule reads, with the layer filled in. */
  example: string
  layer: string[]
  response: (stems: Stem[]) => PullResponse
  /** Where its moments usually come from. */
  source: 'lines' | 'loud' | 'mark'
  loudLayer?: string[]
}

const THUNDER = ['thunder', 'storm', 'boom']
const RAIN = ['rain', 'weather', 'wind']
const VOICE = ['talk', 'voice', 'dialog', 'speech', 'people', 'speak']

/** Starting points that show what a rule can be. Each one is an ordinary rule once picked. */
export const RECIPES: Recipe[] = [
  { id: 'lie', meaning: 'A lie', title: 'The storm knows a lie', example: 'Whenever a character lies, the thunder answers.', layer: THUNDER,
    response: (s) => ({ kind: 'echo', sourceStemId: pickLayer(s, THUNDER), source: null, anchor: 'end', amount: 0.7 }), source: 'lines' },
  { id: 'words', meaning: 'Words matter', title: 'Words cut through', example: 'Whenever someone speaks, the rain steps back.', layer: RAIN,
    response: () => ({ kind: 'hush', amount: 0.6 }), source: 'loud', loudLayer: VOICE },
  { id: 'power', meaning: 'Power arrives', title: 'Power arrives', example: 'Every time he shows his strength, the thunder swells.', layer: THUNDER,
    response: () => ({ kind: 'swell', amount: 0.8 }), source: 'mark' },
  { id: 'god', meaning: 'Something more than human', title: 'A voice not human', example: 'When he speaks as a god, his voice drops.', layer: VOICE,
    response: () => ({ kind: 'effect', effect: 'deepen', amount: 0.5 }), source: 'lines' },
  { id: 'dread', meaning: 'Losing control', title: 'Losing control', example: 'When the weapon fires, the world spins.', layer: ['residual', 'screech', 'fight'],
    response: () => ({ kind: 'effect', effect: 'disorient', amount: 0.7 }), source: 'mark' },
]

export function newRule(stems: Stem[], index: number, recipe?: Recipe): Rule {
  const target = recipe ? pickLayer(stems, recipe.layer) : stems[0]?.id ?? ''
  return {
    id: Math.random().toString(36).slice(2, 10),
    meaning: recipe?.meaning ?? 'A new rule',
    color: COLORS[index % COLORS.length],
    targetStemId: target,
    response: recipe ? recipe.response(stems) : { kind: 'swell', amount: 0.6 },
    moments: [],
  }
}

/** One sentence that says the whole rule, the way it reads on the motif strip. */
export function describe(rule: Rule, stems: Stem[]): string {
  const name = (id: string) => stems.find((s) => s.id === id)?.label ?? 'a missing layer'
  const target = name(rule.targetStemId)
  const r = rule.response
  const what =
    r.kind === 'echo' ? `${target} answers with ${r.sourceStemId === rule.targetStemId ? 'its' : `${name(r.sourceStemId)}'s`} ${r.source ? 'chosen' : 'loudest'} sound`
      : r.kind === 'effect' ? `${target} ${RESPONSES.find((x) => x.kind === r.effect)?.verb}`
        : `${target} ${r.kind === 'swell' ? 'swells' : 'hushes'}`
  return `Whenever “${rule.meaning}”, ${what}`
}

const KEY = 'tidepool.moonpull.'

export function loadRules(scope: string): Rule[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY + scope) ?? '[]') as unknown
    return Array.isArray(raw) ? (raw.filter((r) => r && typeof r === 'object' && typeof (r as Rule).id === 'string' && Array.isArray((r as Rule).moments)) as Rule[]) : []
  } catch {
    return []
  }
}

export function saveRules(scope: string, rules: Rule[]): void {
  try { localStorage.setItem(KEY + scope, JSON.stringify(rules)) } catch { /* rules just won't survive a reload */ }
}
