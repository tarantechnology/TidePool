export function Logo({ className = '' }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <rect x="1.5" y="1.5" width="21" height="21" rx="6" stroke="currentColor" strokeOpacity="0.28" />
        <path
          d="M6 13v-2M9 16V8M12 18V6M15 15V9M18 13v-2"
          stroke="#2b7bd6"
          strokeWidth="1.8"
          strokeLinecap="round"
        />
      </svg>
      <span className="text-[15px] font-semibold tracking-tight">SoundCanvas</span>
    </span>
  )
}
