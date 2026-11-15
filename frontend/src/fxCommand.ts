import type { FxChip, FxPlan } from './api/backend'

/**
 * Rule-based reading of common effect asks, so layer effects work without the AI planner:
 * "add reverb to the vocals", "make the drums sound like a phone", "slowed and reverb",
 * "duck the music under the voice", "normalize for youtube". Returns null for anything it
 * does not recognise (the caller then tries its other parsers), or a clarify with a question.
 * Preset names match app/fx_schema.py.
 */
export interface FxStem { id: string; label: string; tags?: string[]; fx?: FxChip[] }
export type FxCommand =
  | { kind: 'fx' | 'duck' | 'loudness'; fx: FxPlan; message: string }
  | { kind: 'clarify'; message: string }

const PRESETS: Array<[RegExp, string]> = [
  [/\bslowed\s*(?:\+|and|&|n'?|with|plus)?\s*reverb\w*|\bslowed[-_ ]?reverb/, 'slowed_reverb'],
  [/\bnightcore\b/, 'nightcore'],
  [/\bsped[- ]up\b|\bspeed (?:it |this |the (?:song|track) )?up\b/, 'sped_up'],
  [/\bslowed\b|\bslow (?:it|this|the (?:song|track)) down\b/, 'slowed'],
  [/\bdeep[- ]?fried\b|\bear ?rape\b|\bblown[- ]out\b/, 'deep_fried'],
  [/\bwalkie[- ]?talkie\b|\bintercom\b|\bcb radio\b|\bpolice radio\b/, 'walkie_talkie'],
  [/\b(?:tele|cell ?|i)?phone\b|\bphone call\b|\bfacetime\b/, 'telephone'],
  [/\bold (?:radio|timey radio)\b|\bradio\b/, 'radio'],
  [/\b(?:through|behind) (?:the |a )?walls?\b|\b(?:next|another|other) room\b|\bmuffled\b/, 'through_the_wall'],
  [/\bunder ?water\b/, 'underwater'],
  [/\bmegaphone\b|\bbullhorn\b|\bloudspeaker\b|\bpa system\b/, 'megaphone'],
  [/\bold[- ](?:movie|film|timey)\b|\bsilent film\b|\b19[0-4]0s\b/, 'old_movie'],
  [/\bvinyl\b|\brecord player\b|\bturntable\b|\bcrackl\w*/, 'vinyl'],
  [/\blo-?fi\b/, 'lofi'],
  [/\b8-?bit\b|\bchiptune\b|\bretro game\b|\bbit-?crush\w*/, 'eight_bit'],
  [/\bchipmunks?\b|\bhelium\b/, 'chipmunk'],
  [/\bdeep(?:er)? voice\b|\bvoice deeper\b|\bsound deeper\b|\bpitch(?:ed)? (?:it )?down\b/, 'deep_voice'],
  [/\brobot\w*\b|\bcyborg\b|\bdalek\b/, 'robot'],
  [/\bannouncer\b|\bradio host\b|\bmovie trailer\b|\bvoice of god\b/, 'announcer'],
  [/\bgiant\b|\bmonster\b|\bdemon\w*\b/, 'giant'],
  [/\bdream(?:y|like)\b|\bethereal\b|\bspacey\b/, 'dreamy'],
  [/\bchurch\b|\bcathedral\b/, 'church'],
  [/\bstadium\b|\barena\b/, 'stadium'],
  [/\b(?:concert )?hall\b/, 'hall'],
  [/\bbathroom\b|\bshower\b/, 'bathroom'],
  [/\bsmall room\b|\bin a room\b|\bcloset\b/, 'small_room'],
  [/\bplate\b/, 'plate'],
  [/\becho(?:es|ey|y)?\b|\bdelay\b/, 'echo'],
  [/\breverb\w*\b|\breverberat\w*/, 'reverb'],
  [/\bbass ?boost\w*|\bmore bass\b|\bboost the bass\b|\bbassier\b/, 'bass_boost'],
  [/\bmono\b/, 'mono'],
  [/\bwider\b|\bwiden\b|\bwide\b|\bmore stereo\b/, 'wide'],
  [/\bwarm(?:er|th)?\b/, 'warm'],
  [/\bbright(?:er|en)?\b|\bcrisp(?:er)?\b|\bairy\b/, 'bright'],
  [/\bharsh\w*\b|\bshrill\b|\bsibilan\w*/, 'less_harsh'],
  [/\bmudd(?:y|iness)\b|\bboomy\b/, 'less_muddy'],
  [/\bdistort\w*|\boverdrive\w*|\bcrunch\w*|\bfuzz\w*/, 'overdrive'],
  [/\bchorus\b/, 'chorus'],
  [/\bflang\w*/, 'flanger'],
  [/\bphas(?:er|ing)\b/, 'phaser'],
  [/\btremolo\b/, 'tremolo'],
  [/\bauto-?pan\w*|\bpan(?:ning)? (?:it )?(?:left and right|back and forth|around)\b/, 'autopan'],
]

/** Presets whose words almost always mean the effect. The rest ("phone", "radio", "monster", "chorus",
 * "hall", "warm"…) also name sounds or song sections, so they count only in an effect-shaped phrase. */
const UNAMBIGUOUS = new Set(['slowed_reverb', 'nightcore', 'sped_up', 'slowed', 'deep_fried', 'underwater', 'lofi', 'eight_bit', 'chipmunk',
  'deep_voice', 'reverb', 'bass_boost', 'less_harsh', 'less_muddy', 'overdrive', 'flanger', 'phaser', 'tremolo', 'autopan', 'vinyl'])
/** "phone ringing", "monster roar", "radio chatter": the preset word names a sound, not an effect. */
const SOUND_NOUN = /^\s*(?:roar\w*|ring(?:s|ing)?|ringtones?|barks?|barking|growl\w*|scream\w*|chatter\w*|static|buzz\w*|notifications?|alarms?|noises?|beeps?|beeping|sirens?|footsteps?|crash\w*|explosions?|sfx|sound effects?|clips?|sample)\b/
const LAYER_WORD = /^\s*(?:vocals?|vox|voice|voices|singing|singer|dialogue|dialog|speech|narration|drums?|percussion|bass|guitars?|piano|keys|synths?|strings|music|mix|track|song)\b/
const EFFECT_WORD = /^\s*(?:effects?|filters?|fx|presets?|style|vibe|version|treatment|reverb\w*|echo|delay|voice)\b/

/** Whether a preset word at this spot is asked for as an effect: "sound like a phone", "into a robot",
 * "make the mix wider", "in a hall", "add chorus to…", "chorus on the guitar", "robot voice". */
function effectShaped(before: string, after: string): boolean {
  if (/\bsound(?:s|ing)? like\b/.test(before)) return true
  if (/\binto\s+(?:(?:an?|the|some)\s+)?(?:\w+\s+)?$/.test(before)) return true
  if (/^(?:please\s+)?(?:make|get)\s+(?:it|this|them|him|her|(?:his|her|their|the|my)\s+[\w ]+?)\s+(?:sound\s+)?(?:(?:a (?:bit|little|lot)|much|way|more|even)\s+)*$/.test(before)) return true
  if (/\b(?:in|inside)\s+(?:an?|the)\s+(?:\w+\s+)?$/.test(before)) return true
  if (/\b(?:add|put|apply|use|give)\b(?:.*\s)?$/.test(before) && /^\s*(?:(?:effects?|filters?|fx)\s+)?(?:to|on|onto|over)\b/.test(after)) return true
  if (/^\s*on\s+(?:the|it|this|his|her|my|their|all)\b/.test(after)) return true
  if (EFFECT_WORD.test(after) || LAYER_WORD.test(after)) return true
  return before.trim() === '' && /^\s*(?:$|it\b|this\b|that\b|the\b|my\b|everything\b)/.test(after)
}

/** The preset an ask names. With `anywhere`, any mention counts (for "remove the church reverb", "less echo"). */
function findPreset(text: string, anywhere: boolean): string | null {
  for (const [pattern, name] of PRESETS) {
    for (const m of text.matchAll(new RegExp(pattern.source, 'g'))) {
      const before = text.slice(0, m.index), after = text.slice(m.index! + m[0].length)
      if (SOUND_NOUN.test(after)) continue
      if (anywhere || UNAMBIGUOUS.has(name) || effectShaped(before, after)) return name
    }
  }
  return null
}

/**
 * Semitones for a pitch ask. An explicit amount wins ("pitch the vocals up 2 semitones"); without one,
 * "higher"/"raise the pitch" is +3 and "lower" is -3, "a lot"/"way higher" ±7, "a little"/"slightly" ±1.
 * Null when the text is not a pitch ask or has no direction. "More"/"less" then tweaks it.
 */
export function pitchShift(text: string): number | null {
  const t = text.toLowerCase()
  if (!/\bpitch(?:ed)?\b/.test(t)) return null
  const up = /\b(?:higher|high|up|raise[ds]?|raising|increase[ds]?|boost)\b/.test(t)
  const down = /\b(?:lower|low|down|drop|decrease[ds]?|reduce[ds]?|deeper)\b/.test(t)
  if (up === down) return null
  const sign = up ? 1 : -1
  const amount = /(\d+(?:\.\d+)?)\s*(?:semitones?|half[- ]steps?|steps?|st)\b/.exec(t)
  if (amount) return sign * Number(amount[1])
  if (/\b(?:a lot|lots|way|much|really|super|very|tons?)\b/.test(t)) return sign * 7
  if (/\b(?:a (?:little|bit|touch|tad)|slightly|little|bit|tad)\b/.test(t)) return sign * 1
  return sign * 3
}

/** Asks that belong to sound replacement or separation, never to the effect parser. ("Add a monster roar at 12"
 * falls out because "monster" names a sound there; "add reverb to the vocals at 0:12" is still an effect.) */
const REPLACEMENT_ASK = /\b(?:replace|replacing|swap|substitute)\b|\binstead of\b/
const SEPARATION_ASK = /\b(?:isolate|separate|extract|split|solo)\b|\bpull out\b|\bkeep only\b/

const MUSIC_WORDS = /\b(?:music|instrumental|backing track|bgm|beat)\b/
/** Words for a layer, and what a matching stem's label or tags contain. */
const LAYERS: Array<[RegExp, RegExp]> = [
  [/\b(?:vocals?|vox|singers?|singing|lead vocal)\b/, /vocal|voice|sing|choir/],
  [/\b(?:voice|dialogue|dialog|speech|narrat\w*|narrator|speaker|talking|host|presenter|him|her)\b/, /speech|dialog|voice|narrat|speak|talk|vocal|person|man|woman/],
  [/\b(?:drums?|percussion|kick|snare|hi-?hats?)\b/, /drum|percussion|kick|snare|hat|cymbal/],
  [/\bbass(?:line| guitar)?\b/, /\bbass\b/],
  [/\bguitars?\b/, /guitar/],
  [/\b(?:piano|keys|keyboard)\b/, /piano|keys|keyboard/],
  [/\b(?:synths?|synthesizer)\b/, /synth/],
  [/\bstrings\b/, /string|violin|cello/],
  [MUSIC_WORDS, /music|instrument|song|bgm|residual|everything else/],
]
const WHOLE = /\b(?:whole|entire|everything|all of it|the mix|full mix|the song|this song|the track|this track|the audio|the clip|the video|whole thing)\b/
const VOICE = /speech|dialog|voice|narrat|speak|talk|vocal|sing|person/

const PRONOUN = /^(?:him|her|them|it)$/

function haystack(stem: FxStem): string {
  return [stem.label, ...(stem.tags ?? [])].join(' ').toLowerCase()
}

function seconds(value: string): number {
  const parts = value.split(':').map(Number)
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0]
}

const TIME = String.raw`(\d+(?::\d{1,2})?(?:\.\d+)?)\s*(?:s|secs?|seconds?)?`
function interval(text: string): { start: number; end: number } | null {
  const m = new RegExp(String.raw`\b(?:from|between)\s+${TIME}\s+(?:to|and|until|-|–)\s+${TIME}`).exec(text)
  if (!m) return null
  const start = seconds(m[1]), end = seconds(m[2])
  return end > start ? { start, end } : null
}

function plan(partial: Partial<FxPlan>): FxPlan {
  return { targets: ['mix'], preset: null, chain: null, interval: null, adjust: null, speed: null, duck_source: null, duck_depth_db: null, loudness_lufs: null, ...partial }
}

type Targets = { ids: string[] } | { missing: string } | null

/** The layers a request names, ['mix'] for the whole song, or what it named that does not exist. */
function targets(text: string, stems: FxStem[], ignore: RegExp | null): Targets {
  const cleaned = ignore ? text.replace(new RegExp(ignore.source, 'g'), ' ') : text
  const named = LAYERS.filter(([words]) => words.test(cleaned))
  const word = (w: RegExp) => (w.exec(cleaned)?.[0] ?? 'that layer').replace(/^the /, '')
  if (!named.length) return { ids: ['mix'] }
  const musicOnly = named.every(([w]) => w === MUSIC_WORDS)
  if (!stems.length) return musicOnly ? { ids: ['mix'] } : { missing: word(named[0][0]) }
  const ids = new Set<string>()
  for (const [words, pattern] of named) {
    const name = word(words).replace(/s$/, '')
    // A layer whose own name matches beats one that only matches by a tag or a broader word.
    // Pronouns ("make her louder") only ever match by role, never as a substring of a label ("Other").
    const pronoun = PRONOUN.test(name)
    const whole = new RegExp(String.raw`\b${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:s|es)?\b`)
    const exact = pronoun ? [] : stems.filter(s => whole.test(s.label.toLowerCase()))
    const tagged = pronoun ? [] : stems.filter(s => (s.tags ?? []).some(t => t.toLowerCase().replace(/s$/, '') === name))
    let matched = exact.length ? exact : tagged.length ? tagged : stems.filter(s => pattern.test(haystack(s)))
    if (!matched.length && words === MUSIC_WORDS) matched = stems.filter(s => !VOICE.test(haystack(s)))
    matched.forEach(s => ids.add(s.id))
  }
  if (!ids.size) return { missing: word(named[0][0]) }
  if (WHOLE.test(cleaned) && musicOnly) return { ids: ['mix'] }
  return { ids: stems.filter(s => ids.has(s.id)).map(s => s.id) }
}

function pickTargets(text: string, stems: FxStem[], ignore: RegExp | null): string[] | FxCommand {
  const found = targets(text, stems, ignore) ?? { ids: ['mix'] }
  if ('missing' in found) {
    return { kind: 'clarify', message: stems.length
      ? `No layer matches "${found.missing}". Layers: ${stems.map(s => s.label).join(', ')}.`
      : `Split the song into tracks first to put an effect on the ${found.missing.replace(/^the /, '')}, or ask for the whole mix.` }
  }
  return found.ids
}

const NOT_EFFECTS = /\b(?:add|insert|play|put)\b.*\b(?:vine boom|laugh track|sound effect|sfx|airhorn|record scratch|sad trombone|crickets)\b|\bbleep|\bcurse|\bswear/

export function fxCommand(raw: string, stems: FxStem[] = []): FxCommand | null {
  const text = raw.trim().toLowerCase().replace(/[.!?]+$/, '')
  if (!text || NOT_EFFECTS.test(text) || REPLACEMENT_ASK.test(text) || SEPARATION_ASK.test(text)) return null
  const voiceStem = () => stems.find(s => /speech|dialog|narrat|speak|talk|voice/.test(haystack(s))) ?? stems.find(s => VOICE.test(haystack(s)))

  // Auto-ducking.
  if (/\bduck(?:ing|ed)?\b|\bside-?chain\b/.test(text) || /\b(?:lower|quieter|turn down|dip|drop)\b.*\b(?:music|song|bgm|background|beat)\b.*\b(?:when|while|whenever|under|during)\b.*\b(?:talk\w*|speak\w*|voice|speech|dialog\w*|says?|vocals?|narrat\w*)/.test(text)) {
    const source = voiceStem()
    if (!source) return { kind: 'clarify', message: 'Ducking needs a separate voice layer. Isolate the voice or dialogue first, then ask again.' }
    const named = targets(text.replace(/\b(?:under|below|beneath|when|while|whenever|during)\b.*$/, ' '), stems.filter(s => s.id !== source.id), null)
    const music = named && 'ids' in named && named.ids[0] !== 'mix' ? named.ids : stems.filter(s => s.id !== source.id).map(s => s.id)
    if (!music.length) return { kind: 'clarify', message: 'Ducking needs a music layer besides the voice.' }
    const depth = /\bby (\d+(?:\.\d+)?)\s*db\b/.exec(text)
    return { kind: 'duck', fx: plan({ targets: music, duck_source: source.id, duck_depth_db: depth ? Number(depth[1]) : null }), message: `Ducking under ${source.label}.` }
  }

  // Loudness normalization.
  if (/\bnormali[sz]\w*|\bloudness\b|\blufs\b|\bloud enough for\b|\bmaster(?:ed|ing)? for\b|\b(?:for|ready for|upload(?:ing)? to) (?:youtube|tiktok|spotify|instagram|reels|podcasts?|broadcast|apple music|tv)\b/.test(text)) {
    const explicit = /(-?\d+(?:\.\d+)?)\s*lufs/.exec(text)
    const lufs = explicit ? -Math.abs(Number(explicit[1])) : /\b(?:podcasts?|apple)\b/.test(text) ? -16 : /\b(?:broadcast|tv|ebu)\b/.test(text) ? -23 : -14
    const chosen = pickTargets(text.replace(/\b(?:youtube|tiktok|spotify|instagram|reels|podcasts?|broadcast|apple music|tv)\b/g, ' '), stems, null)
    if (!Array.isArray(chosen)) return chosen
    return { kind: 'loudness', fx: plan({ targets: chosen, loudness_lufs: lufs }), message: `Normalize to ${lufs} LUFS.` }
  }

  const named = findPreset(text, true)
  const namedWords = named ? PRESETS.find(([, name]) => name === named)![0] : null

  // Removing an applied effect, only when a chip is there to remove ("remove the reverb" may otherwise be a separation).
  if (/\b(?:remove|take (?:off|out)|get rid of|strip|turn off|disable|no more|without)\b|\btake\b.+\b(?:off|out)\b/.test(text) && /\b(?:effects?|fx|filters?)\b|\bthe\b/.test(text)) {
    const chosen = targets(text, stems, namedWords)
    const ids = chosen && 'ids' in chosen ? chosen.ids : ['mix']
    const pool = stems.filter(s => ids[0] === 'mix' || ids.includes(s.id))
    const chips = pool.flatMap(s => s.fx ?? []).filter(c => !c.baked)
    const match = chips.filter(c => (named ? c.preset === named : /\b(?:effects?|fx|filters?)\b/.test(text)))
    if (!match.length) return null
    return { kind: 'fx', fx: plan({ targets: ids, adjust: 'remove', preset: named }), message: `Remove ${named ? match[match.length - 1].label : 'the last effect'}.` }
  }

  // "More" / "less" of the last effect.
  const more = /^(?:(?:make it|a (?:bit|little|lot)|slightly|way|much|even|bit)\s+)*(?:more|wetter|stronger|heavier|louder effect|crank it|turn it up)\b/.test(text) || /\bmore (?:of (?:that|it)|effect)\b|\bdouble it\b/.test(text)
  const less = /^(?:(?:make it|a (?:bit|little|lot)|slightly|way|much|even|bit)\s+)*(?:less|drier|weaker|subtler|lighter|softer|gentler|tone it down|turn it down)\b/.test(text) || /\btoo much\b|\btoo (?:strong|wet|heavy|loud)\b|\btone (?:it|that) down\b/.test(text)
  if ((more || less) && !(named && /^(?:add|put|apply|give|make)\b/.test(text))) {
    const chosen = pickTargets(text, stems, namedWords)
    if (!Array.isArray(chosen)) return chosen
    return { kind: 'fx', fx: plan({ targets: chosen, adjust: more ? 'more' : 'less', preset: named }), message: `${more ? 'More' : 'Less'} of ${named ?? 'the last effect'}.` }
  }

  const preset = findPreset(text, false)
  const presetWords = preset ? PRESETS.find(([, name]) => name === preset)![0] : null

  const semitones = pitchShift(text)
  const echoTime = /\b(\d+(?:\.\d+)?)\s*(ms|milliseconds?)\b/.exec(text) ?? /\b(\d+(?:\.\d+)?)\s*(s|seconds?)\s+(?:echo|delay)\b/.exec(text)
  if (!preset && !semitones) return null
  const chosen = pickTargets(text, stems, presetWords)
  if (!Array.isArray(chosen)) return chosen
  const chain: FxPlan['chain'] = []
  if (semitones) chain.push({ type: 'pitch', params: { semitones } })
  const usePreset = semitones && (preset === 'deep_voice') ? null : preset
  if (usePreset === 'echo' && echoTime) {
    chain.push({ type: 'delay', params: { time: Number(echoTime[1]) * (echoTime[2].startsWith('m') ? 1 : 1000) } })
  }
  const tempo = usePreset && ['slowed_reverb', 'slowed', 'nightcore', 'sped_up'].includes(usePreset)
  if (tempo && chosen[0] !== 'mix') return { kind: 'clarify', message: 'Slowed, sped-up and nightcore change the whole song. Ask for it on the whole song.' }
  return {
    kind: 'fx',
    fx: plan({ targets: chosen, preset: usePreset === 'echo' && echoTime ? null : usePreset, chain: chain.length ? chain : null, interval: tempo ? null : interval(text) }),
    message: 'Effect preview.',
  }
}

