/** Tidepool: a drop and the rings it sends out. */
export function Logo({ className = '' }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="10.25" stroke="#2270c8" strokeOpacity="0.28" strokeWidth="1.5" />
        <circle cx="12" cy="12" r="6.25" stroke="#2270c8" strokeOpacity="0.6" strokeWidth="1.5" />
        <circle cx="12" cy="12" r="2.6" fill="#2270c8" />
      </svg>
      <span className="font-display text-[17px] font-bold tracking-[-0.02em]">Tidepool</span>
    </span>
  )
}
