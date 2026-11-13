/** The entrance ripple's origin, handed from the landing page's click to the editor. */
/** Where the visitor clicked to enter, so the pool opens from that point. */
export const ENTRY_KEY = 'tidepool.entry'

export function rememberEntry(x: number, y: number) {
  try { sessionStorage.setItem(ENTRY_KEY, JSON.stringify({ x: x / window.innerWidth, y: y / window.innerHeight })) } catch { /* storage blocked: open from the center */ }
}

export function readEntry(): { x: number; y: number } {
  try {
    const raw = sessionStorage.getItem(ENTRY_KEY)
    const v = raw ? JSON.parse(raw) as { x?: unknown; y?: unknown } : null
    if (v && typeof v.x === 'number' && typeof v.y === 'number') return { x: v.x, y: v.y }
  } catch { /* fall through */ }
  return { x: 0.5, y: 0.5 }
}


/** Forget the click once the entrance has used it, so a reload opens from the center. */
export function clearEntry() {
  try { sessionStorage.removeItem(ENTRY_KEY) } catch { /* nothing stored */ }
}
