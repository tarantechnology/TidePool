/**
 * Loading, as water: a drop falls, meets the surface, and rings spread out.
 * The rings are flattened ellipses so the surface reads as seen from above at an angle.
 */
export function DropLoader({ size = 64, label }: { size?: number; label?: string }) {
  return (
    <span className="inline-flex flex-col items-center gap-2" role="status" aria-live="polite">
      <span className="tp-drop relative block" style={{ width: size, height: size }} aria-hidden="true">
        <span className="tp-drop-pool" />
        {[0, 1, 2].map((i) => <span key={i} className="tp-drop-ring" style={{ ['--i' as string]: i }} />)}
        <span className="tp-drop-splash" />
        <svg className="tp-drop-fall" viewBox="0 0 12 16" width={size * 0.16} height={size * 0.21}>
          <path d="M6 0C6 0 0 7.2 0 10.2A6 6 0 0012 10.2C12 7.2 6 0 6 0z" fill="currentColor" />
        </svg>
      </span>
      {label && <span className="text-xs text-accent">{label}</span>}
    </span>
  )
}
