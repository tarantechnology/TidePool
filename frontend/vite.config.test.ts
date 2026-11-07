import { afterEach, describe, expect, it, vi } from 'vitest'
import type { UserConfig } from 'vite'

async function proxyTarget(): Promise<unknown> {
  vi.resetModules()
  const config = (await import('./vite.config')).default as UserConfig
  const entry = config.server?.proxy?.['/api']
  return typeof entry === 'object' ? { ...entry } : entry
}

afterEach(() => vi.unstubAllEnvs())

describe('dev proxy', () => {
  it('defaults to the local backend', async () => {
    vi.stubEnv('BACKEND_URL', '')
    expect(await proxyTarget()).toMatchObject({ target: 'http://localhost:8000' })
  })

  it('honors BACKEND_URL', async () => {
    vi.stubEnv('BACKEND_URL', 'http://backend.test:9000')
    expect(await proxyTarget()).toMatchObject({ target: 'http://backend.test:9000' })
  })

  it('keeps the /api prefix, because the documented routes start with it', async () => {
    expect(await proxyTarget()).not.toHaveProperty('rewrite')
  })
})
