import { describe, expect, it, vi } from 'vitest'
import type { SavedStem } from './backend'
import { CONFLICT_MESSAGE, GONE_MESSAGE, saveStemSettings } from './stemSettings'
import type { StemSettings } from './stemSettings'

const stem = (revision: number, over: Partial<SavedStem> = {}): SavedStem => ({
  id: 's1',
  name: 'dog',
  duration: 4,
  audio_url: '/audio/s1',
  revision,
  volume: 1,
  muted: false,
  solo: false,
  automation: [],
  start_offset: 0,
  mute_ranges: [],
  ...over,
})

const settings: StemSettings = { name: 'dog barking', start_offset: 2 }

describe('saveStemSettings', () => {
  it('patches against the revision the project holds now, not the loaded snapshot', async () => {
    // "Load stem settings" saw revision 3; a remote command has since made it 7.
    const readStems = vi.fn(async () => [stem(7)])
    const patch = vi.fn(async () => {})

    const result = await saveStemSettings({ readStems, patch }, 's1', settings)

    expect(result).toEqual({ ok: true, saved: stem(7) })
    expect(readStems).toHaveBeenCalledOnce()
    expect(patch).toHaveBeenCalledWith(stem(7), settings)
  })

  it('reads before it writes', async () => {
    const order: string[] = []
    await saveStemSettings(
      {
        readStems: async () => {
          order.push('read')
          return [stem(1)]
        },
        patch: async () => {
          order.push('patch')
        },
      },
      's1',
      settings,
    )
    expect(order).toEqual(['read', 'patch'])
  })

  it('says so and refreshes the snapshot when the write is rejected', async () => {
    const readStems = vi
      .fn(async () => [stem(2)])
      .mockResolvedValueOnce([stem(2)])
      .mockResolvedValueOnce([stem(3, { name: 'renamed elsewhere', start_offset: 5 })])
    const patch = vi.fn(async () => {
      throw new Error('Backend returned 409: stem revision mismatch')
    })

    const result = await saveStemSettings({ readStems, patch }, 's1', settings)

    expect(result.ok).toBe(false)
    if (result.ok) throw new Error('expected a conflict')
    expect(result.message).toBe(CONFLICT_MESSAGE)
    expect(result.stems).toEqual([stem(3, { name: 'renamed elsewhere', start_offset: 5 })])
    expect(readStems).toHaveBeenCalledTimes(2)
  })

  it('reports a stem the project no longer has, without writing', async () => {
    const patch = vi.fn(async () => {})
    const result = await saveStemSettings({ readStems: async () => [stem(1, { id: 'other' })], patch }, 's1', settings)

    expect(result).toEqual({ ok: false, message: GONE_MESSAGE, stems: [stem(1, { id: 'other' })] })
    expect(patch).not.toHaveBeenCalled()
  })

  it('lets any other write failure through', async () => {
    const io = {
      readStems: async () => [stem(1)],
      patch: async () => {
        throw new Error('Backend returned 500')
      },
    }
    await expect(saveStemSettings(io, 's1', settings)).rejects.toThrow('Backend returned 500')
  })
})
