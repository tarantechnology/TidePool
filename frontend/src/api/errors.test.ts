import { describe, expect, it } from 'vitest'
import { backendDetail, backendStatus, isAbort, isConflict } from './errors'

describe('backend error reading', () => {
  it('reads the status and the detail request() put in the message', () => {
    const err = new Error('Backend returned 502: GPT is not configured on the server.')
    expect(backendStatus(err)).toBe(502)
    expect(backendDetail(err)).toBe('GPT is not configured on the server.')
  })

  it('reads a status with no detail', () => {
    expect(backendStatus(new Error('Backend returned 404'))).toBe(404)
    expect(backendDetail(new Error('Backend returned 404'))).toBeNull()
  })

  it('claims nothing about other failures', () => {
    for (const err of [new Error('Failed to fetch'), 'Backend returned 500', null, undefined]) {
      expect(backendStatus(err)).toBeNull()
      expect(backendDetail(err)).toBeNull()
    }
  })

  it('spots a rejected save, and only a rejected save', () => {
    expect(isConflict(new Error('Backend returned 409: revision mismatch'))).toBe(true)
    expect(isConflict(new Error('Backend returned 404'))).toBe(false)
    expect(isConflict(new Error('conflict'))).toBe(false)
  })
})

describe('isAbort', () => {
  it('recognises the rejection an aborted request or poll produces', async () => {
    const controller = new AbortController()
    controller.abort()
    const err = await Promise.resolve()
      .then(() => controller.signal.throwIfAborted())
      .catch((e: unknown) => e)
    expect(isAbort(err)).toBe(true)
  })

  it('does not treat a real failure as a cancellation', () => {
    expect(isAbort(new Error('Backend returned 500'))).toBe(false)
    expect(isAbort('aborted')).toBe(false)
  })
})
