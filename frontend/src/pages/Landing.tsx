import type { ReactNode } from 'react'
import { HeroPreview } from '../components/HeroPreview'
import { Logo } from '../components/Logo'
import { rememberEntry } from '../entry'
import { RippleField } from '../components/RippleField'

function OpenButton({ children, large = false }: { children: ReactNode; large?: boolean }) {
  return (
    <a
      href="#/app"
      onClick={(e) => rememberEntry(e.clientX, e.clientY)}
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

const STEPS = [
  {
    n: '01',
    title: 'Split',
    body: 'Bring in a clip. Every sound lands on its own layer: each voice, the siren, the room.',
  },
  {
    n: '02',
    title: 'Describe',
    body: 'Say what you want changed in plain words. Louder siren, quieter crowd, one voice gone.',
  },
  {
    n: '03',
    title: 'Automate',
    body: 'Tie a change to a moment. Lift the siren as the argument heats up, then let it fall away.',
  },
]

type Status = 'Live' | 'Beta' | 'Soon'

const FEATURES: Array<{ title: string; body: string; status: Status; icon: ReactNode }> = [
  {
    title: 'Layered timeline',
    body: 'Every sound on its own track, with mute, solo and volume that respond instantly.',
    status: 'Live',
    icon: <path d="M4 7h16M4 12h16M4 17h16" />,
  },
  {
    title: 'Time-based automation',
    body: 'Gain keyframes play back on the audio clock and draw right on the waveform.',
    status: 'Live',
    icon: <path d="M3 17l5-8 4 5 4-9 5 12" />,
  },
  {
    title: 'Prompt to edit',
    body: 'Type an instruction and watch the mix change. Mute, solo, louder, quieter, over a time range.',
    status: 'Beta',
    icon: <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" />,
  },
  {
    title: 'Text-prompted separation',
    body: 'Pull a siren, a dog or a single voice out of any clip using SAM Audio.',
    status: 'Soon',
    icon: <path d="M5 12h3l2-6 4 12 2-6h3" />,
  },
  {
    title: 'Click to isolate',
    body: 'Click a person in the video and hear only their sound.',
    status: 'Soon',
    icon: <path d="M6 4l12 7-5 1.5L11 18 6 4z" />,
  },
  {
    title: 'Edit from your phone',
    body: 'Send a message with an instruction and get the finished video back.',
    status: 'Soon',
    icon: <path d="M8 3h8a1 1 0 011 1v16a1 1 0 01-1 1H8a1 1 0 01-1-1V4a1 1 0 011-1zM11 18h2" />,
  },
]

function StatusChip({ status }: { status: Status }) {
  const cls =
    status === 'Soon'
      ? 'border-line text-subtle'
      : 'border-accent/30 bg-accent/10 text-accent'
  return (
    <span className={`rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${cls}`}>
      {status}
    </span>
  )
}

export default function Landing() {
  return (
    <div className="relative min-h-screen overflow-x-clip bg-canvas text-fg">
      {/* Nav */}
      <nav className="sticky top-0 z-40 border-b border-line bg-canvas/70 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
          <a href="#/" className="text-fg">
            <Logo />
          </a>
          <div className="hidden items-center gap-8 text-sm text-muted sm:flex">
            <a href="#how" className="transition-colors hover:text-fg">
              How it works
            </a>
            <a href="#features" className="transition-colors hover:text-fg">
              Features
            </a>
          </div>
          <OpenButton>Open editor</OpenButton>
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
          <h1 className="mx-auto max-w-3xl text-balance text-fg text-5xl leading-[1.04] font-semibold tracking-tight sm:text-7xl">
            Edit sound with a sentence.
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            Split a clip into layers, then mute a voice, bring up a siren, or automate the mix over time. Say what you
            want and hear it change.
          </p>
          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <OpenButton large>Open the editor</OpenButton>
            <a
              href="#how"
              className="inline-flex h-12 items-center rounded-full border border-line-strong px-6 text-[15px] text-fg-2 transition-colors hover:bg-tint/[0.04] hover:text-fg"
            >
              How it works
            </a>
          </div>
        </div>

        <div className="relative mx-auto max-w-5xl px-5 pb-24">
          <HeroPreview />
        </div>
      </header>

      {/* How it works */}
      <section id="how" className="relative mx-auto max-w-6xl scroll-mt-20 px-5 py-20">
        <div className="mb-12 max-w-xl">
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-accent">How it works</div>
          <h2 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">From one messy clip to a mix you control.</h2>
        </div>
        <div className="grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-3">
          {STEPS.map((s) => (
            <div key={s.n} className="group bg-surface p-7 transition-colors hover:bg-surface-2">
              <div className="font-mono text-xs text-accent">{s.n}</div>
              <div className="mt-8 text-xl font-medium tracking-tight">{s.title}</div>
              <p className="mt-2 text-sm leading-relaxed text-muted">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative mx-auto max-w-6xl scroll-mt-20 px-5 py-20">
        <div className="mb-12 max-w-xl">
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-accent">Features</div>
          <h2 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Built around the edit, not the file.</h2>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="group relative overflow-hidden rounded-2xl border border-line bg-surface p-6 transition-colors hover:border-line-strong"
            >
              <div className="pointer-events-none absolute -top-16 -right-16 h-40 w-40 rounded-full bg-accent/0 blur-3xl transition-all group-hover:bg-accent/10" />
              <div className="flex items-center justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-tint/[0.03] text-accent">
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    {f.icon}
                  </svg>
                </span>
                <StatusChip status={f.status} />
              </div>
              <div className="mt-6 text-[17px] font-medium tracking-tight">{f.title}</div>
              <p className="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative mx-auto max-w-6xl px-5 pt-12 pb-28">
        <div className="relative overflow-hidden rounded-3xl border border-line-strong bg-surface px-6 py-16 text-center">
          <div className="bg-grid fade-edges pointer-events-none absolute inset-0 opacity-60" />
          <div className="pointer-events-none absolute -bottom-24 left-1/2 h-56 w-[34rem] -translate-x-1/2 rounded-full bg-accent/20 blur-[100px]" />
          <h2 className="relative text-balance text-3xl text-fg font-semibold tracking-tight sm:text-5xl">
            Hear it before you read about it.
          </h2>
          <p className="relative mx-auto mt-4 max-w-md text-muted">
            The editor runs in your browser on a mock scene: two voices, a siren, a room.
          </p>
          <div className="relative mt-8">
            <OpenButton large>Open the editor</OpenButton>
          </div>
        </div>
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-6 font-mono text-[11px] text-faint">
          <Logo className="text-subtle" />
          <span>Hackathon prototype</span>
        </div>
      </footer>
    </div>
  )
}
