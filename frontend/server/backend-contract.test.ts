import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { getJob, patchStem, toTrackState } from '../src/api/backend'
import { pollJob, toJobStatus } from '../src/api/jobs'
import { HttpStemSource } from '../src/api/stemSources'
import { applyAction, normalizeKeyframes, parsePrompt } from '../src/edits'
import type { Stem, TrackState } from '../src/types'

// Runs this client against a real backend, so the documented contract stays
// honest. Skipped unless BACKEND_URL is set, for example:
//   BACKEND_URL=http://localhost:8000 npx vitest run server/backend-contract
// Also skipped when it is set but nothing answers. The reason is in the suite name.
const baseUrl = process.env.BACKEND_URL
const POLL_MS = 100
const SR = 48000

const reachable =
  baseUrl !== undefined &&
  (await fetch(`${baseUrl}/openapi.json`, { signal: AbortSignal.timeout(2000) }).then(
    (res) => res.ok,
    () => false,
  ))
const skipReason = reachable ? '' : baseUrl ? ' (skipped: BACKEND_URL not reachable)' : ' (skipped: BACKEND_URL not set)'

describe.skipIf(!reachable)(`backend contract${skipReason}`, () => {
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
      automation: normalizeKeyframes([
        { t: 0, value: 1 },
        { t: 1, value: 1 },
        { t: 1, value: 0.25 },
        { t: 2, value: 0.25 },
      ]),
    })
    const list = await (await fetch(`${url}/api/projects/${project.project_id}/stems`, { headers: auth })).json()
    const stored = list.find((s: { id: string }) => s.id === stemId)
    expect(stored.volume).toBe(0.5)
    expect(stored.automation).toEqual([
      { time: 0, value: 1 },
      { time: 1, value: 1 },
      { time: 1.001, value: 0.25 },
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

  /** A processed one-stem project over a 3s sine. */
  async function project3s() {
    const created = await (await fetch(`${url}/api/projects`, { method: 'POST' })).json()
    const id = created.project_id as string
    const config = { baseUrl: url, token: created.access_token as string }
    const auth = { Authorization: `Bearer ${config.token}` }
    const wav = join(dir, `${id}.wav`)
    const sine = 'sine=frequency=440:duration=3'
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', sine, '-ac', '2', '-ar', String(SR), wav])
    const form = new FormData()
    form.append('file', new Blob([readFileSync(wav)], { type: 'audio/wav' }), 'in.wav')
    await fetch(`${url}/api/projects/${id}/upload`, { method: 'POST', headers: auth, body: form })
    const started = await fetch(`${url}/api/projects/${id}/process`, {
      method: 'POST',
      headers: { ...auth, 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompts: ['tone'] }),
    })
    const { job_id: jobId } = await started.json()
    await pollJob(async () => toJobStatus(await getJob(config, jobId)), { intervalMs: POLL_MS })
    const [saved] = await (await fetch(`${url}/api/projects/${id}/stems`, { headers: auth })).json()
    const stem: Stem = { id: saved.id, label: 'tone', kind: 'sfx', url: '', duration: 3 }
    return { id, config, auth, stem, jobId }
  }

  async function exportWav(id: string, auth: Record<string, string>): Promise<Float32Array> {
    const res = await fetch(`${url}/api/projects/${id}/export`, {
      method: 'POST',
      headers: { ...auth, 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: 'wav', include_video: false }),
    })
    expect(res.status).toBe(202)
    const { export_id: exportId } = await res.json()
    const status = async () => (await fetch(`${url}/api/exports/${exportId}`, { headers: auth })).json()
    const done = await pollJob(async () => toJobStatus(await status()), { intervalMs: POLL_MS })
    expect(done.state).toBe('done')
    const { download_url: downloadUrl } = await status()
    const file = await fetch(`${url}${downloadUrl}`, { headers: auth })
    const path = join(dir, `${exportId}.wav`)
    writeFileSync(path, Buffer.from(await file.arrayBuffer()))
    const raw = join(dir, `${exportId}.f32`)
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', path, '-ac', '1', '-ar', String(SR), '-f', 'f32le', raw])
    const bytes = readFileSync(raw)
    return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.length / 4))
  }

  function rmsAt(data: Float32Array, t: number, win = 0.05): number {
    const start = Math.floor((t - win / 2) * SR)
    const end = Math.floor((t + win / 2) * SR)
    let sum = 0
    for (let i = start; i < end; i++) sum += data[i] * data[i]
    return Math.sqrt(sum / (end - start))
  }

  const BOUNDARY_CASES: Array<{ prompt: string; silent: number[]; loud: number[] }> = [
    { prompt: 'mute the tone from 0 to 1 seconds', silent: [0.15, 0.7], loud: [1.3, 2.5] },
    { prompt: 'mute the tone from 2 to 3 seconds', silent: [2.3, 2.85], loud: [0.3, 1.7] },
    { prompt: 'mute the tone at 1.5 seconds', silent: [1.8, 2.8], loud: [0.3, 1.2] },
    { prompt: 'mute the tone from 0 to 3 seconds', silent: [0.3, 1.5, 2.8], loud: [] },
  ]

  for (const { prompt, silent, loud } of BOUNDARY_CASES) {
    it(`stores and exports "${prompt}"`, async () => {
      const { id, config, auth, stem } = await project3s()
      const start: TrackState[] = [{ stemId: stem.id, gain: 1, muted: false, solo: false, automation: [] }]
      const [track] = parsePrompt(prompt, [stem], start).actions.reduce(applyAction, start)

      await patchStem(config, id, track)

      const [stored] = await (await fetch(`${url}/api/projects/${id}/stems`, { headers: auth })).json()
      expect(stored.automation).toEqual(track.automation.map((k) => ({ time: k.t, value: k.value })))

      const out = await exportWav(id, auth)
      for (const t of silent) expect(rmsAt(out, t), `silent at ${t}`).toBeLessThan(0.001)
      for (const t of loud) expect(rmsAt(out, t), `loud at ${t}`).toBeGreaterThan(0.01)
    }, 30_000)
  }

  it('restores saved state through the editor mapper', async () => {
    const { id, config, auth, stem, jobId } = await project3s()
    const track: TrackState = {
      stemId: stem.id,
      gain: 0.5,
      muted: true,
      solo: true,
      automation: [
        { t: 0.5, value: 1.5 },
        { t: 2, value: 0.25 },
      ],
    }
    await patchStem(config, id, track)
    const [row] = await (await fetch(`${url}/api/projects/${id}/stems`, { headers: auth })).json()
    expect(toTrackState(row)).toEqual(track)

    const source = new HttpStemSource(config, jobId)
    await source.load()
    source.dispose()
    expect(source.savedTracks).toEqual([track])
    expect(source.syncTarget).toEqual({ config, projectId: id, revisions: { [stem.id]: row.revision } })
  }, 30_000)

  it('previews an offset and a webhook mute range the way the export renders them', async () => {
    const { id, config, auth, stem, jobId } = await project3s()
    const patched = await fetch(`${url}/api/projects/${id}/stems/${stem.id}`, {
      method: 'PATCH',
      headers: { ...auth, 'Content-Type': 'application/json' },
      body: JSON.stringify({ start_offset: 1 }),
    })
    expect(patched.status).toBe(200)
    const hook = await fetch(`${url}/api/webhooks/local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: id, token: config.token, text: 'Mute the tone from 2 to 3 seconds' }),
    })
    expect((await hook.json()).status).toBe('applied')

    const source = new HttpStemSource(config, jobId)
    const [loaded] = await source.load()
    source.dispose()
    expect(loaded).toMatchObject({ startOffset: 1, muteRanges: [[2, 3]] })

    const out = await exportWav(id, auth)
    expect(rmsAt(out, 0.5)).toBeLessThan(0.001)
    expect(rmsAt(out, 1.5)).toBeGreaterThan(0.01)
    expect(rmsAt(out, 2.5)).toBeLessThan(0.001)
    expect(rmsAt(out, 3.5)).toBeGreaterThan(0.01)
  }, 30_000)

  it('reports a missing job through the client error path', async () => {
    await expect(getJob({ baseUrl: url, token: 'wrong' }, 'missing')).rejects.toThrow(
      'Backend returned 404: Job not found',
    )
  })
})
