export function formatDuration(s: number): string {
  const m = Math.floor(s / 60), r = Math.round(s % 60)
  return m ? `${m}:${String(r).padStart(2, '0')}` : `${s.toFixed(s < 10 ? 1 : 0)}s`
}

export function relative(t: number, now = Date.now()): string {
  const m = Math.round((now - t) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.round(h / 24)
  return d === 1 ? 'yesterday' : d < 7 ? `${d} days ago` : new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
