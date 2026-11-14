import { PASSES_HELP, passLabel } from '../api/refine'

interface Props {
  value: number
  choices: readonly number[]
  onChange: (passes: number) => void
  disabled?: boolean
  className: string
}

/** Separation thoroughness. Hidden when the backend allows only one pass. */
export function PassesSelect({ value, choices, onChange, disabled, className }: Props) {
  if (choices.length < 2) return null
  return (
    <label className="block text-xs">
      Passes
      <select aria-label="Passes" className={className} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))}>
        {choices.map((n) => (
          <option key={n} value={n}>
            {passLabel(n)}
          </option>
        ))}
      </select>
      <span className="text-muted">{PASSES_HELP}</span>
    </label>
  )
}
