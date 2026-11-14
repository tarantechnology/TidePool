import { useState } from 'react'
import type { ReactNode } from 'react'
import { FeaturesDialog } from '../components/FeaturesDialog'
import { HeroPreview } from '../components/HeroPreview'
import { Logo } from '../components/Logo'
import { RippleField } from '../components/RippleField'

function OpenButton({ children, large = false }: { children: ReactNode; large?: boolean }) {
  return (
    <a
      href="#/app"
      className={`glow-accent inline-flex items-center gap-2 rounded-full bg-accent font-medium text-white transition-all hover:bg-[#1b62b3] active:scale-[0.98] ${
        large ? 'h-12 px-7 text-[15px]' : 'h-9 px-4 text-sm'
      }`}
    >
      {children}
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </a>
  )
}

export default function Landing() {
  const [showFeatures, setShowFeatures] = useState(false)
  return (
    <div className="relative min-h-screen overflow-x-clip bg-canvas text-fg">
      {/* Nav */}
      <nav className="sticky top-0 z-40 border-b border-line bg-canvas/70 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
          <a href="#/" className="text-fg">
            <Logo />
          </a>
          <button type="button" onClick={() => setShowFeatures(true)}
            className="inline-flex h-9 items-center rounded-full border border-line-strong bg-surface/70 px-4 text-sm font-medium text-fg transition-colors hover:border-accent/40 hover:bg-accent-soft hover:text-accent">
            Features
          </button>
        </div>
      </nav>

      {/* Hero */}
      <header className="relative">
        <RippleField className="[mask-image:linear-gradient(to_bottom,#000_55%,transparent)]" />
        <div className="pointer-events-none absolute -top-32 left-1/2 h-[26rem] w-[50rem] -translate-x-1/2 rounded-full bg-sky/25 blur-[140px]" />

        <div className="relative mx-auto max-w-6xl px-5 pt-20 pb-16 text-center sm:pt-28">
          <div className="mx-auto mb-7 inline-flex items-center gap-2 rounded-full border border-line-strong bg-tint/[0.03] px-3 py-1.5 font-mono text-[11px] text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_8px_rgb(43_123_214/0.9)]" />
            Prompt-driven audio editing, early prototype
          </div>
          <h1 className="mx-auto max-w-3xl text-balance font-display text-5xl leading-[1.02] font-bold tracking-[-0.02em] text-fg sm:text-7xl">
            Edit sound with a sentence.
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            Split a clip into layers, then mute a voice, bring up a siren, or automate the mix over time. Say what you
            want and hear it change.
          </p>
          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <OpenButton large>Open the editor</OpenButton>
          </div>
        </div>

        <div className="relative mx-auto max-w-5xl px-5 pb-24">
          <HeroPreview />
        </div>
      </header>

      <FeaturesDialog open={showFeatures} onClose={() => setShowFeatures(false)} />

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-6 font-mono text-[11px] text-faint">
          <Logo className="text-subtle" />
          <span>Built at HackPrinceton</span>
        </div>
      </footer>
    </div>
  )
}
