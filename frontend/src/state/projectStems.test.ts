import { describe, expect, it } from 'vitest'
import { PROJECT_STEMS_LABEL } from '../api/stemSources'
import { NO_PROJECT_STEMS, showsMockBadge, stemsReadySummary, stemsStateFromJob } from './projectStems'

describe('stemsStateFromJob', () => {
  it('keeps the mock flag and the warnings the job reported', () => {
    expect(
      stemsStateFromJob({ status: 'completed', inference: { model: { is_mock: true }, warnings: ['CPU fallback'] } }),
    ).toEqual({ mockStems: true, warnings: ['CPU fallback'] })
  })

  it('keeps a real model as a real model', () => {
    expect(stemsStateFromJob({ status: 'completed', inference: { model: { is_mock: false } } })).toEqual({
      mockStems: false,
      warnings: [],
    })
  })

  it('is unknown, not false, when the job said nothing about the model', () => {
    expect(stemsStateFromJob({ status: 'completed' })).toEqual({ mockStems: null, warnings: [] })
    expect(stemsStateFromJob({ status: 'completed', inference: null })).toEqual({ mockStems: null, warnings: [] })
    expect(stemsStateFromJob({ status: 'completed', inference: { model: {} } }).mockStems).toBeNull()
  })
})

describe('stemsReadySummary', () => {
  it('says which of the three it is', () => {
    expect(stemsReadySummary(true)).toBe('Ready — test stems; no AI separation was performed.')
    expect(stemsReadySummary(false)).toBe('Stems ready to edit.')
    expect(stemsReadySummary(null)).toBe('Stems ready; model identity unavailable.')
  })
})

describe('showsMockBadge', () => {
  const mock = { projectId: 'p1', mockStems: true, warnings: [] }

  it('badges mock output for as long as those stems are loaded', () => {
    expect(showsMockBadge(mock, PROJECT_STEMS_LABEL)).toBe(true)
  })

  it('stops once other stems replace them', () => {
    expect(showsMockBadge(mock, '2 audio files')).toBe(false)
    expect(showsMockBadge(mock, 'Original audio')).toBe(false)
  })

  it('says nothing for a real model, an unknown one, or no project', () => {
    expect(showsMockBadge({ ...mock, mockStems: false }, PROJECT_STEMS_LABEL)).toBe(false)
    expect(showsMockBadge({ ...mock, mockStems: null }, PROJECT_STEMS_LABEL)).toBe(false)
    expect(showsMockBadge(NO_PROJECT_STEMS, PROJECT_STEMS_LABEL)).toBe(false)
  })
})
