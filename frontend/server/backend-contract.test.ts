import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { getJob, patchStem } from '../src/api/backend'
import { pollJob, toJobStatus } from '../src/api/jobs'
import { HttpStemSource } from '../src/api/stemSources'

// Runs this client against a real backend, so the documented contract stays
// honest. Skipped unless BACKEND_URL is set, for example:
//   BACKEND_URL=http://localhost:8000 npx vitest run server/backend-contract
const baseUrl = process.env.BACKEND_URL
const POLL_MS = 100

describe.skipIf(!baseUrl)('backend contract', () => {
  const url = baseUrl ?? ''
  const dir = mkdtempSync(join(tmpdir(), 'soundcanvas-contract-'))
  afterAll(() => rmSync(dir, { recursive: true, force: true }))

  it('runs upload, process, stems, patch and export', async () => {
    const project = await (await fetch(`${url}/api/projects`, { method: 'POST' })).json()
    const config = { baseUrl: url, token: project.access_token as string }
    const auth = { Authorization: `Bearer ${config.token}` }

    const wav = join(dir, 'in.wav')
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', wav])
    const form = new FormData()
    form.append('file', new Blob([readFileSync(wav)], { type: 'audio/wav' }), 'in.wav')
    const upload = await fetch(`${url}/api/projects/${project.project_id}/upload`, {
      method: 'POST',
      headers: auth,
      body: form,
    })
    expect(upload.status).toBe(200)

    const started = await fetch(`${url}/api/projects/${project.project_id}/process`, {
      method: 'POST',
      headers: { ...auth, 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompts: ['dog', 'music'] }),
    })
    expect(started.status).toBe(202)
    const { job_id: jobId } = await started.json()

    const final = await pollJob(async () => toJobStatus(await getJob(config, jobId)), { intervalMs: POLL_MS })
    expect(final).toMatchObject({ state: 'done', progress: 100 })

    const source = new HttpStemSource(config, jobId)
    const stems = await source.load()
    expect(stems.map((s) => s.label).sort()).toEqual(['dog', 'music'])
    expect(stems[0].kind).toBe('sfx')
    expect(stems[0].duration).toBeCloseTo(2, 0)
    const audio = await (await fetch(stems[0].url)).arrayBuffer()
    expect(new TextDecoder().decode(audio.slice(0, 4))).toBe('RIFF')
    source.dispose()

    const stemId = stems[0].id
    await patchStem(config, project.project_id, {
      stemId,
      gain: 0.5,
      muted: false,
      solo: false,
      automation: [
        { t: 0, value: 1 },
        { t: 1, value: 1 },
        { t: 1, value: 0.25 },
        { t: 2, value: 0.25 },
      ],
    })
    const list = await (await fetch(`${url}/api/projects/${project.project_id}/stems`, { headers: auth })).json()
    const stored = list.find((s: { id: string }) => s.id === stemId)
    expect(stored.volume).toBe(0.5)
    expect(stored.automation).toEqual([
      { time: 0, value: 1 },
      { time: 1, value: 0.25 },
      { time: 2, value: 0.25 },
    ])

    const exported = await fetch(`${url}/api/projects/${project.project_id}/export`, {
      method: 'POST',
      headers: { ...auth, 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: 'wav', include_video: false }),
    })
    expect(exported.status).toBe(202)
    const { export_id: exportId } = await exported.json()
    const done = await pollJob(
      async () => toJobStatus(await (await fetch(`${url}/api/exports/${exportId}`, { headers: auth })).json()),
      { intervalMs: POLL_MS },
    )
    expect(done.state).toBe('done')
  }, 30_000)

  it('reports a missing job through the client error path', async () => {
    await expect(getJob({ baseUrl: url, token: 'wrong' }, 'missing')).rejects.toThrow(
      'Backend returned 404: Job not found',
    )
  })
})
