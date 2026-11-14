import type { ReactNode } from 'react'

/**
 * An instant hover and focus tooltip for small controls whose label is a letter or an icon.
 * `label` is also the accessible name, so screen readers hear the same words.
 */
export function Tip({ label, detail, side = 'top', children }: { label: string; detail?: string; side?: 'top' | 'bottom'; children: ReactNode }) {
  return (
    <span className="group/tip relative inline-flex">
      {children}
      <span role="tooltip"
        className={`pointer-events-none absolute left-1/2 z-50 w-max max-w-56 -translate-x-1/2 rounded-lg bg-fg px-2.5 py-1.5 text-left text-[11px] leading-snug text-white opacity-0 shadow-lg transition-opacity duration-150 group-hover/tip:opacity-100 group-focus-within/tip:opacity-100 ${side === 'top' ? 'bottom-full mb-1.5' : 'top-full mt-1.5'}`}>
        <span className="block font-medium">{label}</span>
        {detail && <span className="block text-white/75">{detail}</span>}
      </span>
    </span>
  )
}
