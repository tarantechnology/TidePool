import { describe, expect, it, vi } from 'vitest'
import type { AgentPlan } from '../api/backend'
import { isCompound, localMultiPlan, singleFlowClause, splitClauses } from './multiCommand'

const STEMS = [
  { id: 'v1', label: 'Vocals' }, { id: 'd1', label: 'Drums' }, { id: 'b1', label: 'Bass' }, { id: 'g1', label: 'Guitar' },
]

describe('splitClauses', () => {
  it.each<[string, string[]]>([
    ['add reverb to the vocals, make the drums sound like a phone, pitch the bass down 2 semitones, add a piano part and put a vine boom at 0:12',
      ['add reverb to the vocals', 'make the drums sound like a phone', 'pitch the bass down 2 semitones', 'add a piano part', 'put a vine boom at 0:12']],
    ['add reverb to the vocals and make the drums sound like a phone', ['add reverb to the vocals', 'make the drums sound like a phone']],
    ['make the vocals sound like a robot then add a laugh track at the end', ['make the vocals sound like a robot', 'add a laugh track at the end']],
    ['make the vocals sound like a robot, then add a laugh track at the end', ['make the vocals sound like a robot', 'add a laugh track at the end']],
    ['duck the music under the voice; normalize for youtube', ['duck the music under the voice', 'normalize for youtube']],
    ['add echo to the guitar and also mute the drums', ['add echo to the guitar', 'mute the drums']],
    ['put the vocals in a church. add a sad trombone at the end', ['put the vocals in a church', 'add a sad trombone at the end']],
    ['can you add reverb to the vocals and could you add a piano part', ['can you add reverb to the vocals', 'could you add a piano part']],
    ['lower the drums, raise the vocals and normalize for tiktok', ['lower the drums', 'raise the vocals', 'normalize for tiktok']],
    ['add a string pad under the chorus and then put an air horn at 0:30', ['add a string pad under the chorus', 'put an air horn at 0:30']],
    ['make the guitar distorted plus add reverb to the vocals', ['make the guitar distorted', 'add reverb to the vocals']],
    ['add some reverb to the vocals, and make the bass warmer', ['add some reverb to the vocals', 'make the bass warmer']],
    ["put a vine boom when he says 'what and add' then mute the drums", ["put a vine boom when he says 'what and add'", 'mute the drums']],
    ['add a vine boom when he says "turn and make" and add reverb', ['add a vine boom when he says "turn and make"', 'add reverb']],
    ['fade out the music after that add a record scratch at 0:05', ['fade out the music', 'add a record scratch at 0:05']],
    ['pan the guitar left, widen the vocals, boost the bass', ['pan the guitar left', 'widen the vocals', 'boost the bass']],
  ])('splits %j', (text, clauses) => {
    expect(splitClauses(text)).toEqual(clauses)
    expect(isCompound(text)).toBe(true)
  })

  it.each([
    'add reverb and echo to the vocals',
    'add reverb to the drums and bass',
    'slowed and reverb',
    'make it sound like a phone',
    'put a vine boom at 0:12 and a record scratch at 0:20',
    'duck the music under the voice',
    'rock and roll vibes on the guitar',
    'make the vocals warmer and brighter',
    // An elided verb is left to the AI planner, which splits it into two steps.
    'make the drums sound like a phone and the vocals sound like a robot',
  ])('keeps %j as one edit', (text) => {
    expect(splitClauses(text)).toHaveLength(1)
    expect(isCompound(text)).toBe(false)
  })
})

const insertPlan: AgentPlan = { kind: 'insert', description: null, remove: false, actions: [], message: '',
  inserts: [{ sfx_query: 'vine boom', sfx_id: 'vine_boom', anchor: { type: 'time', value: 12, relation: 'at', occurrence: null }, offset: 0, gain_db: 0, duck: false }] }
const generatePlan: AgentPlan = { kind: 'generate', description: null, remove: false, actions: [], message: '', generate: { instrument: 'piano', prompt: 'add a piano part' } }

describe('localMultiPlan', () => {
  it('plans effects locally and asks the server parsers for inserts and parts', async () => {
    const planClause = vi.fn(async (clause: string) => (/piano/.test(clause) ? generatePlan : /boom/.test(clause) ? insertPlan : null))
    const plan = await localMultiPlan('add reverb to the vocals, make the drums sound like a phone, add a piano part and put a vine boom at 0:12', STEMS, planClause)
    expect(plan.kind).toBe('multi')
    expect(plan.steps?.map((s) => s.kind)).toEqual(['fx', 'fx', 'generate', 'insert'])
    expect(plan.steps?.[0].fx?.targets).toEqual(['v1'])
    expect(plan.steps?.[1].fx?.preset).toBe('telephone')
    expect(planClause.mock.calls.map((c) => c[0])).toEqual(['add a piano part', 'put a vine boom at 0:12'])
  })

  it('asks instead of guessing when a clause has its own flow or cannot be planned', async () => {
    const none = async () => null
    expect((await localMultiPlan('bleep the swearing and add reverb to the vocals', STEMS, none)).message).toMatch(/Edit 1 .*on its own/)
    expect((await localMultiPlan('add reverb to the vocals and replace the dog bark with a meow', STEMS, none)).message).toMatch(/Edit 2/)
    expect((await localMultiPlan('add reverb to the vocals and make it nightcore', STEMS, none)).message).toMatch(/Edit 2/)
    const unknown = await localMultiPlan('add reverb to the vocals and make it pop', STEMS, none)
    expect(unknown.kind).toBe('clarify')
    expect(unknown.message).toMatch(/Edit 2 \("make it pop"\)/)
  })

  it('refuses outrageous requests', async () => {
    const many = Array.from({ length: 9 }, () => 'add reverb to the vocals').join(', ')
    expect((await localMultiPlan(many, STEMS, async () => null)).message).toMatch(/split the request/)
    const parts = ['violins', 'piano', 'a choir', 'a flute'].map((x) => `add ${x}`).join(' and ')
    expect((await localMultiPlan(parts, STEMS, async () => generatePlan)).message).toMatch(/up to 3 new parts/)
  })
})

describe('singleFlowClause: censorship and layer-scoped asks are never swallowed by a multi plan', () => {
  it('a compound prompt containing one is a clarify asking to do it separately', () => {
    for (const [ask, edit] of [
      ['bleep the swearing and add reverb to the vocals', 'Edit 1'],
      ['add reverb to the vocals, then censor the cuss words', 'Edit 2'],
      ['isolate the dog in the Music layer and add a piano part', 'Edit 1'],
      ['add a vine boom at 0:12 and remove the cough from the Vocals track', 'Edit 2'],
    ]) {
      const plan = singleFlowClause(ask, STEMS)
      expect(plan?.kind, ask).toBe('clarify'); expect(plan?.message, ask).toMatch(new RegExp(`${edit} .*separately`))
    }
  })
  it('leaves single asks and ordinary compound asks alone', () => {
    for (const ask of ['bleep the swearing', 'isolate the dog in the Music layer', 'add reverb on the Vocals track and put a vine boom at 0:12'])
      expect(singleFlowClause(ask, STEMS), ask).toBeNull()
  })
  it('localMultiPlan asks before planning any clause', async () => {
    const planClause = vi.fn(async () => null)
    expect((await localMultiPlan('isolate the dog in the Music layer and add a piano part', STEMS, planClause)).kind).toBe('clarify')
    expect(planClause).not.toHaveBeenCalled()
  })
})
