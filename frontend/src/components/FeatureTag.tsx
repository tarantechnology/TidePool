export function FeatureTag({ name, className = '' }: { name: string; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full bg-accent-soft px-1.5 py-px text-[10px] font-medium text-accent ${className}`}>
      <svg width="9" height="9" viewBox="0 0 12 12" fill="none" aria-hidden="true">
        <circle cx="6" cy="6" r="5" stroke="currentColor" strokeOpacity="0.45" />
        <circle cx="6" cy="6" r="2" fill="currentColor" />
      </svg>
      {name}
    </span>
  )
}
