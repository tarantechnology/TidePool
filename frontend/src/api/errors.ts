// Reading backend failures back out of an Error. request() in backend.ts throws
// `Backend returned <status>` with `: <detail>` appended when FastAPI sent a
// string detail, and that is the only shape these helpers claim to understand.

const BACKEND_ERROR = /^Backend returned (\d{3})(?::\s?([\s\S]+))?$/

function parse(err: unknown): { status: number; detail: string | null } | null {
  if (!(err instanceof Error)) return null
  const match = BACKEND_ERROR.exec(err.message)
  if (!match) return null
  return { status: Number(match[1]), detail: match[2] ?? null }
}

/** The HTTP status of a backend error, or null for anything else (a timeout, a bug). */
export function backendStatus(err: unknown): number | null {
  return parse(err)?.status ?? null
}

/** The server's own words for the failure, when it sent any. */
export function backendDetail(err: unknown): string | null {
  return parse(err)?.detail ?? null
}

/** A rejected save: the stem changed since the revision we sent. */
export function isConflict(err: unknown): boolean {
  return backendStatus(err) === 409
}

/** What cancelling is reported as, since it is not a failure. */
export const CANCELLED_MESSAGE = 'Cancelled. Anything already started on the server keeps running.'

/**
 * The rejection an aborted fetch or an aborted poll produces. Cancelling is not
 * a failure, so it must not reach the user as one.
 */
export function isAbort(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError'
}
