import { useState } from 'react'
import { FEATURES } from '../features'

const KEY = 'tidepool.layerTips.hidden'

function readHidden(): boolean {
  try { return localStorage.getItem(KEY) === '1' } catch { return false }
}

/** What you can do once a clip has layers. Shown under the steps, where the panel is otherwise empty. */
export function LayerTips() {
  const [hidden, setHidden] = useState(readHidden)
  const hide = (value: boolean) => {
    setHidden(value)
    try { if (value) localStorage.setItem(KEY, '1'); else localStorage.removeItem(KEY) } catch { /* only this view forgets */ }
  }
  if (hidden) {
    return <button type="button" className="mx-2 mt-2 text-[11px] text-subtle hover:text-accent" onClick={() => hide(false)}>Show layer tips</button>
  }
  const tips: Array<[string, string]> = [
    ['Rename', 'Double-click a layer name. Layers start out named after the sound they were pulled out with.'],
    ['Mix', `Drag a layer's slider, or switch to the Mixer for tall sliders. M mutes a layer, S plays it alone.`],
    ['Shape over time', `On the ${FEATURES.timeline}, double-click a lane to add a volume point, then drag it.`],
    ['Repeat an effect', `${FEATURES.symbolism}, top right, makes a layer react every time something happens.`],
    ['Compare', 'Listen to: Original plays the untouched clip. Switch back to Edited to hear your changes.'],
    ['Change your mind', 'Undo and redo (⌘Z, ⇧⌘Z) cover everything, including Explode and deletes.'],
  ]
  return (
    <section className="mx-2 mt-3 rounded-xl border border-line p-3 text-xs" aria-label="Working with your layers">
      <div className="mb-1.5 flex items-center justify-between">
        <p className="font-medium text-fg">Working with your layers</p>
        <button type="button" className="text-[11px] text-subtle hover:text-fg" onClick={() => hide(true)}>Hide</button>
      </div>
      <dl className="space-y-1.5">
        {tips.map(([what, how]) => (
          <div key={what}>
            <dt className="text-[11px] font-medium text-fg-2">{what}</dt>
            <dd className="text-[11px] leading-snug text-muted">{how}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
