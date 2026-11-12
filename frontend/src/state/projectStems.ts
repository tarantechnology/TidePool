import type { BackendJob } from '../api/backend'
import { PROJECT_STEMS_LABEL } from '../api/stemSources'

// What the editor shell knows about the project the panel holds. Mock output is
// a property of the stems on the timeline, not a one-off toast: it stays
// visible for as long as those stems are loaded.

export interface ProjectStemsState {
  /** Set while a project is loaded, so AI commands have somewhere to run. */
  projectId: string | null
  /** True while the loaded stems are mock output. Null when the job did not say. */
  mockStems: boolean | null
  /** Warnings the inference backend reported for the loaded stems. */
  warnings: string[]
}

export const NO_PROJECT_STEMS: ProjectStemsState = { projectId: null, mockStems: null, warnings: [] }

/** What a finished job says about the stems it produced. */
export function stemsStateFromJob(job: BackendJob): Omit<ProjectStemsState, 'projectId'> {
  const isMock = job.inference?.model?.is_mock
  return {
    mockStems: typeof isMock === 'boolean' ? isMock : null,
    warnings: job.inference?.warnings ?? [],
  }
}

/** The line the panel shows once stems are loaded. */
export function stemsReadySummary(mockStems: boolean | null): string {
  if (mockStems === null) return 'Stems ready; model identity unavailable.'
  return mockStems ? 'Ready — test stems; no AI separation was performed.' : 'Stems ready to edit.'
}

/**
 * Whether to badge the stems as mock output. Importing audio or taking a
 * video's own audio replaces the project's stems, and those are not mock output
 * whatever the last job said.
 */
export function showsMockBadge(state: ProjectStemsState, stemsLabel: string): boolean {
  return state.mockStems === true && stemsLabel === PROJECT_STEMS_LABEL
}
