import { describe, expect, it } from 'vitest'
import { fxCommand, pitchShift } from './fxCommand'
import type { FxStem } from './fxCommand'
import { fxRequest, toFxChips, toStem } from './api/backend'
import { requestsSoundReplacement } from './agent/replacementIntent'

const SONG: FxStem[] = [
  { id: 'v', label: 'Vocals', tags: ['Female singing'] },
  { id: 'd', label: 'Drums', tags: ['Drum kit'] },
  { id: 'b', label: 'Bass', tags: ['Bass guitar'] },
  { id: 'o', label: 'Other', tags: ['Piano', 'Guitar'] },
]
const VIDEO: FxStem[] = [{ id: 's', label: 'speech' }, { id: 'm', label: 'music' }, { id: 'r', label: 'everything else' }]
const CHIPPED: FxStem[] = SONG.map(s => s.id === 'v' ? { ...s, fx: [{ id: 'c1', label: 'Church', preset: 'church', scope: 'layer', speed: null }] } : s)

type Expected = { kind: string; targets?: string[]; preset?: string | null; adjust?: string | null; interval?: { start: number; end: number } | null; duck?: string; lufs?: number; semitones?: number }

// Natural-language asks and the plan the rule-based parser should produce.
const FIXTURES: Array<[string, FxStem[], Expected]> = [
  ['add reverb to the vocals', SONG, { kind: 'fx', targets: ['v'], preset: 'reverb' }],
  ['add some reverb to the vocals', SONG, { kind: 'fx', targets: ['v'], preset: 'reverb' }],
  ['put the singer in a church', SONG, { kind: 'fx', targets: ['v'], preset: 'church' }],
  ['make the vocals sound like they are in a cathedral', SONG, { kind: 'fx', targets: ['v'], preset: 'church' }],
  ['make the drums sound like a phone', SONG, { kind: 'fx', targets: ['d'], preset: 'telephone' }],
  ["make the drums sound like they're coming through a phone", SONG, { kind: 'fx', targets: ['d'], preset: 'telephone' }],
  ['make the vocals sound like a walkie talkie', SONG, { kind: 'fx', targets: ['v'], preset: 'walkie_talkie' }],
  ['make it sound like an old radio', SONG, { kind: 'fx', targets: ['mix'], preset: 'radio' }],
  ['make the music sound like it is coming through the wall', VIDEO, { kind: 'fx', targets: ['m'], preset: 'through_the_wall' }],
  ['make the music sound like the party is in the next room', VIDEO, { kind: 'fx', targets: ['m'], preset: 'through_the_wall' }],
  ['underwater vocals', SONG, { kind: 'fx', targets: ['v'], preset: 'underwater' }],
  ['make his voice sound like a megaphone', VIDEO, { kind: 'fx', targets: ['s'], preset: 'megaphone' }],
  ['slowed and reverb', SONG, { kind: 'fx', targets: ['mix'], preset: 'slowed_reverb' }],
  ['slowed + reverb the whole song', SONG, { kind: 'fx', targets: ['mix'], preset: 'slowed_reverb' }],
  ['make a slowed reverb version', [], { kind: 'fx', targets: ['mix'], preset: 'slowed_reverb' }],
  ['nightcore this', SONG, { kind: 'fx', targets: ['mix'], preset: 'nightcore' }],
  ['speed the song up', SONG, { kind: 'fx', targets: ['mix'], preset: 'sped_up' }],
  ['slow it down', [], { kind: 'fx', targets: ['mix'], preset: 'slowed' }],
  ['make it lo-fi', [], { kind: 'fx', targets: ['mix'], preset: 'lofi' }],
  ['give the whole song a vinyl crackle', SONG, { kind: 'fx', targets: ['mix'], preset: 'vinyl' }],
  ['make the drums 8-bit', SONG, { kind: 'fx', targets: ['d'], preset: 'eight_bit' }],
  ['make it sound like an old movie', VIDEO, { kind: 'fx', targets: ['mix'], preset: 'old_movie' }],
  ['make him sound like a chipmunk', VIDEO, { kind: 'fx', targets: ['s'], preset: 'chipmunk' }],
  ['give the narrator a deep voice', VIDEO, { kind: 'fx', targets: ['s'], preset: 'deep_voice' }],
  ['make the voice robotic', VIDEO, { kind: 'fx', targets: ['s'], preset: 'robot' }],
  ['turn the vocals into a robot', SONG, { kind: 'fx', targets: ['v'], preset: 'robot' }],
  ['make him sound like an announcer', VIDEO, { kind: 'fx', targets: ['s'], preset: 'announcer' }],
  ['make the voice sound like a giant', VIDEO, { kind: 'fx', targets: ['s'], preset: 'giant' }],
  ['make the guitar dreamy', SONG, { kind: 'fx', targets: ['o'], preset: 'dreamy' }],
  ['make the mix wider', SONG, { kind: 'fx', targets: ['mix'], preset: 'wide' }],
  ['make the bass mono', SONG, { kind: 'fx', targets: ['b'], preset: 'mono' }],
  ['make the vocals warmer', SONG, { kind: 'fx', targets: ['v'], preset: 'warm' }],
  ['make the drums brighter', SONG, { kind: 'fx', targets: ['d'], preset: 'bright' }],
  ['the vocals are too harsh', SONG, { kind: 'fx', targets: ['v'], preset: 'less_harsh' }],
  ['the bass sounds muddy', SONG, { kind: 'fx', targets: ['b'], preset: 'less_muddy' }],
  ['add distortion to the guitar', SONG, { kind: 'fx', targets: ['o'], preset: 'overdrive' }],
  ['add chorus to the guitar', SONG, { kind: 'fx', targets: ['o'], preset: 'chorus' }],
  ['flanger on the drums', SONG, { kind: 'fx', targets: ['d'], preset: 'flanger' }],
  ['phaser on the piano', SONG, { kind: 'fx', targets: ['o'], preset: 'phaser' }],
  ['add tremolo to the guitar', SONG, { kind: 'fx', targets: ['o'], preset: 'tremolo' }],
  ['deep fried', [], { kind: 'fx', targets: ['mix'], preset: 'deep_fried' }],
  ['earrape the drop', [], { kind: 'fx', targets: ['mix'], preset: 'deep_fried' }],
  ['put the drums and bass in a stadium', SONG, { kind: 'fx', targets: ['d', 'b'], preset: 'stadium' }],
  ['echo on the vocals from 10 to 12 seconds', SONG, { kind: 'fx', targets: ['v'], preset: 'echo', interval: { start: 10, end: 12 } }],
  ['add reverb to the vocals between 0:30 and 0:45', SONG, { kind: 'fx', targets: ['v'], preset: 'reverb', interval: { start: 30, end: 45 } }],
  ['bathroom reverb on the voice', VIDEO, { kind: 'fx', targets: ['s'], preset: 'bathroom' }],
  ['pitch the vocals up 3 semitones', SONG, { kind: 'fx', targets: ['v'], preset: null, semitones: 3 }],
  ['duck the music under the voice', VIDEO, { kind: 'duck', targets: ['m'], duck: 's' }],
  ['lower the music when he talks', VIDEO, { kind: 'duck', targets: ['m'], duck: 's' }],
  ['add an echo with 500 ms delay to the vocals', SONG, { kind: 'fx', targets: ['v'], preset: null }],
  ['sidechain the drums to the vocals', SONG, { kind: 'duck', targets: ['d'], duck: 'v' }],
  ['normalize for youtube', SONG, { kind: 'loudness', targets: ['mix'], lufs: -14 }],
  ['make it loud enough for tiktok', [], { kind: 'loudness', targets: ['mix'], lufs: -14 }],
  ['normalize the voice for podcasts', VIDEO, { kind: 'loudness', targets: ['s'], lufs: -16 }],
  ['normalize to -18 LUFS', SONG, { kind: 'loudness', targets: ['mix'], lufs: -18 }],
  ['more', CHIPPED, { kind: 'fx', targets: ['mix'], adjust: 'more' }],
  ['a bit wetter', CHIPPED, { kind: 'fx', targets: ['mix'], adjust: 'more' }],
  ['too much', CHIPPED, { kind: 'fx', targets: ['mix'], adjust: 'less' }],
  ['less reverb on the vocals', CHIPPED, { kind: 'fx', targets: ['v'], adjust: 'less', preset: 'reverb' }],
  ['remove the effects from the vocals', CHIPPED, { kind: 'fx', targets: ['v'], adjust: 'remove', preset: null }],
  ['take the church reverb off the vocals', CHIPPED, { kind: 'fx', targets: ['v'], adjust: 'remove', preset: 'church' }],
  ['add reverb to the vocals', [], { kind: 'clarify' }],
  ['add reverb to the saxophone', SONG, { kind: 'fx', targets: ['mix'], preset: 'reverb' }],
  ['nightcore the vocals', SONG, { kind: 'clarify' }],
  ['duck the music under the voice', [], { kind: 'clarify' }],
]

describe('rule-based effect commands', () => {
  it('has a broad fixture set', () => { expect(FIXTURES.length).toBeGreaterThanOrEqual(40) })
  for (const [text, stems, expected] of FIXTURES) {
    it(text + (stems.length ? '' : ' (no layers)'), () => {
      const result = fxCommand(text, stems)
      expect(result, text).not.toBeNull()
      expect(result!.kind).toBe(expected.kind)
      if (result!.kind === 'clarify') return
      const fx = result!.fx
      if (expected.targets) expect(fx.targets).toEqual(expected.targets)
      if ('preset' in expected) expect(fx.preset).toBe(expected.preset)
      if ('adjust' in expected) expect(fx.adjust).toBe(expected.adjust)
      if ('interval' in expected) expect(fx.interval).toEqual(expected.interval)
      if (expected.duck) expect(fx.duck_source).toBe(expected.duck)
      if (expected.lufs !== undefined) expect(fx.loudness_lufs).toBe(expected.lufs)
      if (expected.semitones !== undefined) expect(fx.chain?.[0]).toEqual({ type: 'pitch', params: { semitones: expected.semitones } })
    })
  }
  it('leaves non-effect asks to the other parsers', () => {
    for (const text of ['isolate the drums', 'remove the dog barking', 'mute the vocals from 1 to 2 seconds', 'split this song into tracks',
      'add a vine boom when he says what', 'bleep the swearing', 'make the dog bark sound like a cat', 'remove the reverb']) {
      expect(fxCommand(text, SONG), text).toBeNull()
    }
  })
  it('takes effect asks away from sound replacement', () => {
    // The panel routes effect-shaped asks to effects; replacement still owns "make X sound like Y" without an effect word.
    expect(requestsSoundReplacement('make the drums sound like a phone')).toBe(true)
    expect(fxCommand('make the drums sound like a phone', SONG)?.kind).toBe('fx')
    expect(fxCommand('make the dog bark sound like a cat', SONG)).toBeNull()
  })
})

describe('replacement and separation asks are never effects', () => {
  const REPLACEMENTS = [
    'replace the phone ringing with a doorbell', 'replace the dog bark with a monster roar', 'add a monster roar at 12',
    'turn the chorus into a verse', 'swap the radio chatter for birdsong', 'substitute the robot beeps with a cat meow',
    'replace the echo with silence', 'add a phone ringing at 3 seconds', 'turn the phone ringing into a doorbell',
    'make the phone ringing sound like a doorbell', 'change the giant roar into a whisper', 'replace the hall ambience with rain',
    'use a doorbell instead of the phone',
  ]
  for (const text of REPLACEMENTS) {
    it(text, () => {
      expect(fxCommand(text, SONG), text).toBeNull()
      expect(fxCommand(text, []), text).toBeNull()
    })
  }
  it('leaves isolate / separate / extract asks to separation', () => {
    for (const text of ['isolate the radio chatter', 'isolate the radio', 'separate the phone call audio', 'extract the robot voice',
      'remove the radio chatter', 'mute the phone ringing', 'remove the monster roar', 'isolate the chorus', 'solo the phone',
      'keep only the radio host', 'mute the radio']) {
      expect(fxCommand(text, VIDEO), text).toBeNull()
    }
    // "remove the radio chatter" never removes a Radio chip either; a baked chip is never removable.
    const chipped: FxStem[] = VIDEO.map(s => ({ ...s, fx: [{ id: 'r1', label: 'Radio', preset: 'radio', scope: 'layer', speed: null }] }))
    expect(fxCommand('remove the radio chatter', chipped)).toBeNull()
    const baked: FxStem[] = VIDEO.map(s => ({ ...s, fx: [{ id: 'r1', label: 'Radio', preset: 'radio', scope: 'layer', speed: null, baked: true }] }))
    expect(fxCommand('remove the radio effect', baked)).toBeNull()
    expect(fxCommand('remove the radio effect', chipped)?.kind).toBe('fx')
  })
  it('still reads effect-shaped asks with ambiguous preset words', () => {
    for (const [text, preset] of [['make the drums sound like a phone', 'telephone'], ['turn the vocals into a robot', 'robot'],
      ['make it sound like a monster', 'giant'], ['add chorus to the guitar', 'chorus'], ['radio effect on the voice', 'radio'],
      ['put the vocals in a concert hall', 'hall'], ['make the mix wide', 'wide'], ['echo the vocals', 'echo'], ['robot voice', 'robot'],
      ['add reverb to the vocals at 0:12', 'reverb']] as const) {
      const r = fxCommand(text, SONG)
      expect(r?.kind === 'fx' && r.fx.preset, text).toBe(preset)
    }
  })
})

describe('effect requests and chips', () => {
  const plan = { targets: ['v'], preset: 'church', chain: null, interval: null, adjust: null, speed: null, duck_source: null, duck_depth_db: null, loudness_lufs: null }
  it('turns plans into fx-preview requests', () => {
    expect(fxRequest('fx', plan)).toEqual({ op: 'chain', targets: ['v'], preset: 'church', chain: null, interval: null, speed: null })
    expect(fxRequest('fx', { ...plan, adjust: 'more' }, 'cand')).toEqual({ op: 'adjust', targets: ['v'], direction: 'more', preset: 'church', base_candidate_id: 'cand' })
    expect(fxRequest('fx', { ...plan, adjust: 'remove' })).toEqual({ op: 'remove', targets: ['v'], preset: 'church' })
    expect(fxRequest('duck', { ...plan, targets: ['m'], duck_source: 's' })).toEqual({ op: 'duck', targets: ['m'], duck: { source: 's', depth_db: null } })
    expect(fxRequest('loudness', { ...plan, targets: [], loudness_lufs: -14 })).toEqual({ op: 'loudness', targets: ['mix'], loudness_lufs: -14 })
  })
  it('maps chips from the backend and drops malformed ones', () => {
    expect(toFxChips({ items: [{ id: 'a', label: 'Church', preset: 'church', scope: 'layer' }, { label: 'no id' }, null] }))
      .toEqual([{ id: 'a', label: 'Church', preset: 'church', scope: 'layer', speed: null, kind: null }])
    expect(toFxChips(null)).toEqual([])
    const stem = toStem({ id: 's', name: 'Vocals', duration: 1, audio_url: '/x', fx: { items: [{ id: 'a', label: 'Echo', preset: 'echo', scope: 'mix', speed: null }] } }, 'blob:x')
    expect(stem.fx?.map(c => c.label)).toEqual(['Echo'])
    expect(toStem({ id: 's', name: 'Vocals', duration: 1, audio_url: '/x', fx: null }, 'blob:x').fx).toBeUndefined()
  })
})

describe('pitch defaults', () => {
  const stemsP = [{ id: 'v', label: 'Vocals' }, { id: 'b', label: 'Bass' }, { id: 'o', label: 'Other' }]
  it('higher/lower without an amount is ±3, "a lot" ±7, "a little" ±1; explicit amounts win', () => {
    expect(pitchShift('make the vocals higher pitch')).toBe(3)
    expect(pitchShift('lower the pitch of the bass')).toBe(-3)
    expect(pitchShift('make the vocals way higher pitched')).toBe(7)
    expect(pitchShift('lower the pitch a lot')).toBe(-7)
    expect(pitchShift('raise the pitch slightly')).toBe(1)
    expect(pitchShift('lower the pitch a little')).toBe(-1)
    expect(pitchShift('raise the pitch by 5 semitones')).toBe(5)
    expect(pitchShift('pitch the vocals down 2 semitones')).toBe(-2)
    expect(pitchShift('make the vocals louder')).toBeNull()
  })
  it('builds a pitch chain on the named layer', () => {
    const up = fxCommand('make the vocals higher pitch', stemsP)
    expect(up?.kind).toBe('fx')
    if (up?.kind === 'fx') { expect(up.fx.targets).toEqual(['v']); expect(up.fx.chain).toEqual([{ type: 'pitch', params: { semitones: 3 } }]) }
    const down = fxCommand('lower the pitch of the bass', stemsP)
    if (down?.kind !== 'fx') throw new Error('expected fx')
    expect(down.fx.targets).toEqual(['b']); expect(down.fx.chain).toEqual([{ type: 'pitch', params: { semitones: -3 } }])
  })
  it('pronouns never substring-match a label ("her" is not "Other")', () => {
    const r = fxCommand('add reverb to her', [{ id: 'v', label: 'Vocals' }, { id: 'o', label: 'Other' }])
    if (r?.kind !== 'fx') throw new Error('expected fx')
    expect(r.fx.targets).toEqual(['v'])
  })
})
