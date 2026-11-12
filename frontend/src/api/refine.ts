import type { Refinement } from '../types'
import { getJob, startRefine } from './backend'
import type { BackendConfig, SavedStem } from './backend'
import { backendDetail, backendStatus } from './errors'
import { pollJob, toJobStatus } from './jobs'
import type { JobStatus } from './jobs'

// Multi-pass separation: the thoroughness control on the separation forms, the
// Extract more action on a target stem, and the badge for a refined stem. Kept
// out of the components so it can be tested without a DOM.

/** The most passes the backend contract allows. */
export const MAX_PASSES = 3

export const PASSES_HELP = 'Extra passes rerun the prompt on what is left to catch sound the first pass only quieted.'

/**
 * The pass counts to offer: 1 up to the backend's `max_passes`, or up to
 * MAX_PASSES when it does not say. Never more than the contract allows.
 */
export function passChoices(maxPasses?: unknown): number[] {
  const max = Number.isInteger(maxPasses) && (maxPasses as number) >= 1 ? Math.min(maxPasses as number, MAX_PASSES) : MAX_PASSES
  return Array.from({ length: max }, (_, i) => i + 1)
}

export function passLabel(passes: number): string {
  if (passes === 1) return '1 (fast)'
  if (passes === MAX_PASSES) return `${passes} (thorough)`
  return String(passes)
}

/** A chosen pass count the backend may since have lowered its limit below. */
export function clampPasses(passes: number, choices: readonly number[]): number {
  return Math.max(1, Math.min(passes, choices.length))
}

/** Badge text and tooltip for a stem, or null when there is nothing to say. */
export function refinementBadge(refinement: Refinement | undefined): { text: string; title: string } | null {
  if (!refinement) return null
  const { passesRun, passesRequested, perPassEnergyDb, stoppedEarly } = refinement
  if (passesRun <= 1 && !stoppedEarly) return null
  const text = `${passesRun} pass${passesRun === 1 ? '' : 'es'}${stoppedEarly ? ' · stopped early' : ''}`
  const lines = [`Ran ${passesRun} of ${passesRequested} pass${passesRequested === 1 ? '' : 'es'}.`]
  if (perPassEnergyDb.length > 0) {
    // `|| 0` turns -0 into 0, so a rounded -0.4 dB does not read "-0 dB".
    lines.push(`Energy per pass: ${perPassEnergyDb.map((db) => `${Math.round(db) || 0} dB`).join(', ')}`)
  }
  return { text, title: lines.join(' ') }
}

const isResidual = (stem: SavedStem): boolean => stem.role === 'residual'

/**
 * Why Extract more is unavailable for `stem`, or null when it can run. The
 * backend checks the same things and more (offset stems), and answers 409.
 */
export function refineBlockReason(
  stem: SavedStem,
  stems: readonly SavedStem[],
  { busy, supported }: { busy: boolean; supported?: boolean },
): string | null {
  if (supported === false) return 'This backend cannot extract more.'
  if (stem.role !== 'target') return 'Only an extracted sound can be refined, not the residual or an unlabelled layer.'
  const residuals = stems.filter(isResidual).length
  if (residuals === 0) return 'There is no residual layer to pull more sound from.'
  if (residuals > 1) return 'There is more than one residual layer, so there is no single one to pull from.'
  if (busy) return 'Wait for the current job to finish.'
  return null
}

/** The server's reason when it refused the refine (409), word for word. Null for any other failure. */
export function refineConflictMessage(err: unknown): string | null {
  if (backendStatus(err) !== 409) return null
  return backendDetail(err) ?? 'The backend refused to extract more from this stem right now.'
}

export interface RefineOptions {
  /** Left out of the request when undefined, so the backend picks. */
  passes?: number
  signal?: AbortSignal
  onUpdate?: (status: JobStatus) => void
  intervalMs?: number
}

/**
 * Start Extract more and poll it to the end. Resolves with the job id once the
 * target and residual hold new audio; the caller reloads them. Rejects with
 * the job's error when it fails, and with the signal's reason when cancelled.
 */
export async function runRefine(config: BackendConfig, projectId: string, stemId: string, opts: RefineOptions = {}): Promise<string> {
  const { passes, signal, onUpdate, intervalMs } = opts
  const jobId = await startRefine(config, projectId, stemId, passes, signal)
  const result = await pollJob(async (s) => toJobStatus(await getJob(config, jobId, s)), { signal, onUpdate, intervalMs })
  if (result.state === 'failed') throw new Error(result.error || 'Extract more failed')
  return jobId
}
