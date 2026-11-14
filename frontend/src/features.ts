/**
 * Tidepool's feature names, each something you'd find in or near a tidepool:
 * Whirlpool separates, Ripple selects by sight, Current takes commands,
 * Reflection compares, Deep dive retries for quality, Undertow pulls out more,
 * Tide chart is the timeline, Outflow exports, Moon pull keeps a sound's meaning for the whole clip.
 */
export const FEATURES = {
  separate: 'Whirlpool',
  visual: 'Ripple',
  command: 'Current',
  compare: 'Reflection',
  quality: 'Deep dive',
  refine: 'Undertow',
  timeline: 'Tide chart',
  export: 'Outflow',
  symbolism: 'Moon pull',
} as const

/** What each feature does, in the order a first-time user meets them. Shown in the landing page's Features dialog. */
export const FEATURE_GUIDE: Array<{ name: string; what: string }> = [
  { name: FEATURES.command, what: 'Type what you want in plain words, like "isolate the dog barking" or "make the siren quieter after 10 seconds".' },
  { name: FEATURES.visual, what: 'Click an object in the video to hear just the sound it makes.' },
  { name: FEATURES.separate, what: 'Pulls one sound out of the mix onto its own layer, leaving everything else on another.' },
  { name: FEATURES.compare, what: 'Switch between the original, the edited mix and the isolated sound at the same moment before you commit.' },
  { name: FEATURES.quality, what: 'A slower, more careful separation pass for when the first result is not clean enough.' },
  { name: FEATURES.refine, what: 'Pulls out more of a sound that the first pass only made quieter.' },
  { name: FEATURES.timeline, what: 'Every layer on one timeline: mute, solo, set volume and draw volume changes over time.' },
  { name: FEATURES.symbolism, what: 'Give a sound a meaning and it keeps it for the whole clip, like thunder that answers every lie. Rules come from lines of dialogue, moments you mark, or when a layer gets loud.' },
  { name: FEATURES.export, what: 'Download the result as an MP4 with your new audio, or as WAV or MP3.' },
]
