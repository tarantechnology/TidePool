import type { KeyboardEvent, PointerEvent } from 'react'

interface Props {
  /** 'row' drags up and down between stacked areas; 'column' drags left and right. */
  direction: 'row' | 'column'
  label: string
  /** Pointer position along the drag axis, in client pixels. */
  onDrag: (clientPos: number) => void
  /** Arrow keys move by this many pixels; Shift moves five times as far. */
  onStep: (delta: number) => void
  onReset: () => void
}

/** A thin handle between two panes. Drag it, use the arrow keys on it, or double-click to reset. */
export function Splitter({ direction, label, onDrag, onStep, onReset }: Props) {
  const row = direction === 'row'
  const move = (e: PointerEvent<HTMLDivElement>) => { if (e.buttons & 1) onDrag(row ? e.clientY : e.clientX) }
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const back = row ? 'ArrowUp' : 'ArrowLeft', fwd = row ? 'ArrowDown' : 'ArrowRight'
    if (e.key !== back && e.key !== fwd) return
    e.preventDefault()
    onStep((e.key === fwd ? 1 : -1) * (e.shiftKey ? 80 : 16))
  }
  return (
    <div
      role="separator"
      aria-orientation={row ? 'horizontal' : 'vertical'}
      aria-label={label}
      title={`${label}. Drag, or double-click to reset`}
      tabIndex={0}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); document.body.style.cursor = row ? 'row-resize' : 'col-resize' }}
      onPointerMove={move}
      onPointerUp={() => { document.body.style.cursor = '' }}
      onDoubleClick={onReset}
      onKeyDown={key}
      className={`group relative z-10 flex shrink-0 touch-none items-center justify-center outline-none ${row ? 'h-3 cursor-row-resize' : 'w-3 cursor-col-resize'}`}
    >
      <span className={`rounded-full bg-line-strong transition-all duration-200 group-hover:bg-accent/60 group-focus-visible:bg-accent group-active:bg-accent ${row ? 'h-1 w-10 group-hover:w-16' : 'h-10 w-1 group-hover:h-16'}`} />
    </div>
  )
}
