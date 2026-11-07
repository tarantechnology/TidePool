import { createMockStems } from '../audio/mockStems'
import type { Stem } from '../types'
import { fetchStemBlobUrl, getJob, toStem } from './backend'
import type { BackendConfig, SyncTarget } from './backend'

/**
 * Where stems come from. dispose() is called once the engine has decoded them
 * (or failed to), so sources can release their blob URLs.
 */
export interface StemSource {
  load(): Promise<Stem[]>
  dispose?(): void
  /** Set after load() when edits to these stems belong on a backend. */
  syncTarget?: SyncTarget
  /** Set after load() when they should but cannot get there. */
  syncIssue?: string
}

const AUDIO_EXT = /\.(wav|mp3|m4a|aac|flac|ogg|opus)$/i

export function isAudioFile(file: { name: string; type: string }): boolean {
  return file.type.startsWith('audio/') || AUDIO_EXT.test(file.name)
}

export function isVideoFile(file: { name: string; type: string }): boolean {
  return file.type.startsWith('video/') || /\.(mp4|mov|webm|m4v)$/i.test(file.name)
}

/** "dog_bark-01.wav" becomes "dog bark 01". */
export function labelFromFilename(name: string): string {
  const base = name.replace(/\.[^.]+$/, '')
  const label = base.replace(/[_-]+/g, ' ').trim()
  return label || name
}

/** Unique across a batch, including files with the same name. */
export function stemIdFromFilename(name: string, index: number): string {
  const slug = labelFromFilename(name).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
  return `file-${index}-${slug || 'audio'}`
}

export class MockStemSource implements StemSource {
  private revoke?: () => void

  load(): Promise<Stem[]> {
    const mock = createMockStems()
    this.revoke = mock.revoke
    return Promise.resolve(mock.stems)
  }

  dispose(): void {
    this.revoke?.()
  }
}

export class FileStemSource implements StemSource {
  private files: File[]
  private label?: string
  private urls: string[] = []

  /** label overrides the name taken from the file, for a single-file source. */
  constructor(files: File[], label?: string) {
    this.files = files
    this.label = label
  }

  load(): Promise<Stem[]> {
    return Promise.resolve(
      this.files.map((file, i) => {
        const url = URL.createObjectURL(file)
        this.urls.push(url)
        return {
          id: stemIdFromFilename(file.name, i),
          label: this.label ?? labelFromFilename(file.name),
          // Kinds are not guessed from filenames. The engine fills in duration.
          kind: 'sfx',
          url,
          duration: 0,
        }
      }),
    )
  }

  dispose(): void {
    this.urls.forEach((u) => URL.revokeObjectURL(u))
    this.urls = []
  }
}

/**
 * Stems of a completed processing job. Uses the documented GET /api/jobs/{id},
 * not the /jobs/{id}/stems path assumed in the sprint brief (see docs/CONTRACTS.md).
 */
export class HttpStemSource implements StemSource {
  private config: BackendConfig
  private jobId: string
  private urls: string[] = []
  private disposed = false
  syncTarget?: SyncTarget
  syncIssue?: string

  constructor(config: BackendConfig, jobId: string) {
    this.config = config
    this.jobId = jobId
  }

  async load(): Promise<Stem[]> {
    const job = await getJob(this.config, this.jobId)
    const stems = job.stems ?? []
    if (stems.length === 0) throw new Error('Job has no stems yet')
    if (job.project_id) this.syncTarget = { config: this.config, projectId: job.project_id }
    else this.syncIssue = 'Job has no project_id, so edits stay in this browser.'
    return Promise.all(
      stems.map(async (stem) => {
        const url = await fetchStemBlobUrl(this.config, stem.audio_url)
        // Another stem may have failed already, and dispose() runs at that point.
        if (this.disposed) URL.revokeObjectURL(url)
        else this.urls.push(url)
        return toStem(stem, url)
      }),
    )
  }

  dispose(): void {
    this.disposed = true
    this.urls.forEach((u) => URL.revokeObjectURL(u))
    this.urls = []
  }
}
