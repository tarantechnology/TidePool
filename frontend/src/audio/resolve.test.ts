import { describe, expect, it } from 'vitest'
import { envelopeAt } from '../edits'
import type { EditAction, Keyframe } from '../types'
import { resolveActions } from './resolve'
import type { SampleSource } from './resolve'

const SR = 1000

function ramp(): Float32Array {
  const out = new Float32Array(20 * SR)
  for (let i = 0; i < out.length; i++) out[i] = ((i % 2 === 0 ? 1 : -1) * i) / out.length
  return out
}

const source = (ids: string[]): SampleSource => ({
  getSamples: (id) => (ids.includes(id) ? { data: ramp(), sampleRate: SR } : null),
})

const follow: EditAction = { type: 'follow_intensity', stemId: 'siren', sourceStemIds: ['a', 'b'] }

const keyframesOf = (a: EditAction | undefined): Keyframe[] =>
  a?.type === 'set_automation' ? a.keyframes : []

describe('resolveActions', () => {
  it('turns follow_intensity into set_automation on the target stem', () => {
    const res = resolveActions([follow], source(['a', 'b']), 20)
    expect(res.warnings).toEqual([])
    expect(res.actions[0].type).toBe('set_automation')
    expect(res.actions[0].stemId).toBe('siren')
    const kfs = keyframesOf(res.actions[0])
    expect(kfs[0]).toEqual({ t: 0, value: 1 })
    expect(kfs[kfs.length - 1]).toEqual({ t: 20, value: 2 })
  })

  it('passes minValue and maxValue through', () => {
    const res = resolveActions([{ ...follow, minValue: 1, maxValue: 0.3 } as EditAction], source(['a']), 20)
    expect(keyframesOf(res.actions[0]).at(-1)?.value).toBe(0.3)
  })

  it('leaves other actions untouched and in order', () => {
    const gain: EditAction = { type: 'set_gain', stemId: 'siren', value: 1.5 }
    const res = resolveActions([gain, follow, gain], source(['a']), 20)
    expect(res.actions.map((a) => a.type)).toEqual(['set_gain', 'set_automation', 'set_gain'])
  })

  it('warns about a missing source but uses the rest', () => {
    const res = resolveActions([follow], source(['a']), 20)
    expect(res.actions).toHaveLength(1)
    expect(res.warnings).toEqual(['No audio for source stem b.'])
  })

  it('treats sources as silent after they end', () => {
    const short: SampleSource = {
      getSamples: () => ({ data: new Float32Array(5 * SR).fill(0.5), sampleRate: SR }),
    }
    const kfs = keyframesOf(resolveActions([follow], short, 20).actions[0])
    expect(envelopeAt(kfs, 2)).toBeCloseTo(2, 1)
    expect(envelopeAt(kfs, 15)).toBeCloseTo(1, 1)
    expect(kfs.at(-1)).toEqual({ t: 20, value: 1 })
  })

  it('drops the action with a warning when no source has audio', () => {
    const res = resolveActions([follow], source([]), 20)
    expect(res.actions).toEqual([])
    expect(res.warnings.at(-1)).toContain('no source audio')
  })

  it('lines the curve up with a source placed later on the timeline', () => {
    const loud = new Float32Array(SR * 2).fill(0.5)
    const placed = (startOffset: number): SampleSource => ({
      getSamples: () => ({ data: loud, sampleRate: SR, startOffset }),
    })
    const action: EditAction = { type: 'follow_intensity', stemId: 'siren', sourceStemIds: ['a'] }
    const curve = keyframesOf(resolveActions([action], placed(4), 8).actions[0])
    expect(envelopeAt(curve, 2)).toBeCloseTo(1, 1)
    expect(envelopeAt(curve, 5)).toBeCloseTo(2, 1)
    expect(envelopeAt(curve, 7.5)).toBeCloseTo(1, 1)
  })
})
