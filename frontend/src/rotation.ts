/**
 * Turning the picture in quarter steps. The video file and everything the vision
 * models return (boxes, masks) stay in the clip's own orientation; the editor
 * turns them for display, and Outflow turns the exported MP4 to match.
 */
export type Rotation = 0 | 90 | 180 | 270

/** One quarter turn clockwise (dir 1) or counter-clockwise (dir -1). */
export function turn(r: Rotation, dir: 1 | -1 = 1): Rotation {
  return (((r + dir * 90) % 360) + 360) % 360 as Rotation
}

export const isSideways = (r: Rotation) => r % 180 !== 0

/** Width over height of the picture as shown, given the clip's own aspect. */
export function displayAspect(aspect: number, r: Rotation): number {
  return isSideways(r) ? 1 / aspect : aspect
}

/**
 * A box in the clip's normalized coordinates ([x1, y1, x2, y2], 0 to 1) moved to
 * where it lands once the picture is turned clockwise by r.
 */
export function rotateBox([x1, y1, x2, y2]: readonly number[], r: Rotation): [number, number, number, number] {
  switch (r) {
    case 90: return [1 - y2, x1, 1 - y1, x2]
    case 180: return [1 - x2, 1 - y2, 1 - x1, 1 - y1]
    case 270: return [y1, 1 - x2, y2, 1 - x1]
    default: return [x1, y1, x2, y2]
  }
}

export interface RotatedFit {
  /** The turned picture as seen, inside the container. Overlays in display coordinates use this. */
  left: number
  top: number
  width: number
  height: number
  /** Where to place the unturned element (video, mask) so that rotating it about its centre fills the box above. */
  element: { left: number; top: number; width: number; height: number; transform: string }
}

/** Fit a source of sw x sh, turned by r, inside a w x h container (like object-fit: contain). */
export function fitRotated(w: number, h: number, sw: number, sh: number, r: Rotation): RotatedFit {
  const [dw, dh] = isSideways(r) ? [sh, sw] : [sw, sh]
  const scale = Math.min(w / dw, h / dh)
  const width = dw * scale, height = dh * scale
  const ew = sw * scale, eh = sh * scale
  return {
    left: (w - width) / 2,
    top: (h - height) / 2,
    width,
    height,
    element: { left: (w - ew) / 2, top: (h - eh) / 2, width: ew, height: eh, transform: r ? `rotate(${r}deg)` : 'none' },
  }
}

// Per project, so a reload or a later Outflow export keeps the turn the user chose.
const KEY = 'tidepool.rotation.'
const memory = new Map<string, Rotation>()
const valid = (v: unknown): v is Rotation => v === 0 || v === 90 || v === 180 || v === 270

/** The saved turn for a project, or null when it has none yet. */
export function readRotation(projectId: string | null | undefined): Rotation | null {
  if (!projectId) return null
  if (memory.has(projectId)) return memory.get(projectId)!
  try {
    const v = Number(localStorage.getItem(KEY + projectId) ?? NaN)
    if (valid(v)) { memory.set(projectId, v); return v }
  } catch { /* storage unavailable: memory only */ }
  return null
}

export function writeRotation(projectId: string | null | undefined, r: Rotation): void {
  if (!projectId) return
  memory.set(projectId, r)
  try {
    if (r) localStorage.setItem(KEY + projectId, String(r))
    else localStorage.removeItem(KEY + projectId)
  } catch { /* storage unavailable: memory only */ }
}
