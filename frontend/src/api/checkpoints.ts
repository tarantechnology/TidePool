import { request } from './backend'
import type { BackendConfig } from './backend'

/** Snapshot the project's layers (audio, faders, curves) so a step can be undone or redone later. */
export async function createCheckpoint(config: BackendConfig, projectId: string, label: string): Promise<{ id: string; stems: number }> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/checkpoints`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label: label.slice(0, 200) }),
  })
  return (await res.json()) as { id: string; stems: number }
}

/** Put the layers back as a checkpoint saw them. Returns the job to load them from. */
export async function restoreCheckpoint(config: BackendConfig, projectId: string, id: string): Promise<{ job_id: string; stems: number }> {
  const res = await request(config, `/api/projects/${encodeURIComponent(projectId)}/checkpoints/${encodeURIComponent(id)}/restore`, { method: 'POST' })
  return (await res.json()) as { job_id: string; stems: number }
}
