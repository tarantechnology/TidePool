# Plan: chat-driven layer effects and creator tools

Status: proposed 2026-11-14, branch `feature/chat-fx` (TidePool). Builds on the music-stems work
(docs/plans/music-stems.md) and the existing preview → compare → apply/undo candidate flow.

## Goal

After a song or video is split into layers, the user types what they want into the **existing
command bar**, and TidePool renders it as a previewable, undoable edit on the right layer(s):

- "add some reverb to the vocals"
- "make the drums sound like they're coming through a phone"
- "slowed + reverb the whole song"
- "bleep the swearing"
- "add a vine boom when he says 'what'"

There is no new chat UI. The OpenAI planner (`app/agent.py`, strict Structured Outputs) gains new
plan kinds. Everything runs through the same candidate flow, so A/B compare, apply and undo come for free.

## Principles

1. **The LLM plans and the engine executes.** The model returns a bounded, validated, declarative plan
   (an effect chain with clamped parameters). It never writes filter strings or code. Unknown or
   out-of-range input becomes `clarify`.
2. **Non-destructive by default.** Every edit is a candidate preview. Apply renders into the layer,
   and undo restores it. The timeline (sample rate, channels, length) never changes unless the edit
   is explicitly a tempo edit, which is validated separately.
3. **Presets first, knobs second.** Creators say "make it sound like a phone", not "bandpass
   300–3400 Hz". Named presets map to chains; "more" / "less" / "a bit wetter" adjust the last chain.
4. **Honest about what it can hear.** Anything that depends on content (words, beats, laughs) comes
   from a model with timestamps. The planner never invents times.

## Feature set

### A. Layer effects from chat (core demo, P0)

| Ask | What renders |
|---|---|
| reverb on vocals / "in a church", "stadium", "bathroom", "small room" | convolution reverb with synthesized impulse responses (size, decay, pre-delay, wet) |
| echo / delay ("echo on the last word") | delay with feedback and ping-pong; optional time range |
| "telephone", "radio", "walkie-talkie", "through a wall", "underwater", "megaphone" | EQ/bandpass + saturation + light reverb presets |
| "lo-fi", "vinyl", "8-bit", "bitcrush" | lowpass, wow/flutter, crackle noise layer, bitcrusher |
| chorus / flanger / phaser / tremolo / auto-pan / wider / mono | ffmpeg `chorus`, `flanger`, `aphaser`, `tremolo`, `apulsator`, `stereotools` |
| distortion / overdrive on guitar | soft-clip waveshaper + tone EQ |
| pitch up/down (keep speed), "chipmunk", "deep voice" | rubberband pitch shift (formant-preserving option) |
| "slowed + reverb", "nightcore", "sped up" (whole song) | joint tempo+pitch resample + reverb; the timeline length changes, flagged as such |
| EQ words ("brighter", "warmer", "muddy", "less harsh") | the existing bass/treble/highpass plus parametric bands |
| fade in/out, "cut the music at 0:30", "build-up" | gain automation (already supported) |
| "more", "less", "undo that", "a bit wetter" | adjust the last chain's parameters by a step |

### B. Video-editor and creator utilities (P0/P1)

| Ask | How |
|---|---|
| "duck the music under the speech" (auto-ducking) | sidechain envelope from the dialogue layer's loudness → gain automation on music (P0, no ML) |
| "normalize for YouTube / TikTok" | EBU R128 loudness (`loudnorm`, −14 LUFS / −1 dBTP) on the mix or a layer (P0) |
| "clean up the voice", "remove background noise", "remove echo from the room" | speech enhancement model on the dialogue layer (P1) |
| "bleep the swear words" / "mute the curse words" | word-level ASR → profanity list → 1 kHz beep or a mute range, previewable per word (P1) |
| "cut the ums and uhs" / "remove the long pauses" | word timestamps → mute ranges (audio-only; video cutting is out of scope) (P1) |
| "make the bass hit on the beat drop" / "find the drop" | beat and downbeat tracking → timestamps the planner may use (P1) |

### C. Comedy and meme tools (P1, the fun demo)

| Ask | How |
|---|---|
| "add a vine boom when he says 'what'", "record scratch at 0:12", "sad trombone at the end", "add a laugh track after the joke" | SFX insert: a new layer with a licensed one-shot from a CC0 library, placed at a word timestamp, a beat or an explicit time; retrieved by text with CLAP |
| "dramatic zoom sound", "suspense sting before the reveal" | same insert flow; "before X" uses the word or beat timeline |
| "make him sound like a chipmunk / robot / announcer / giant" | voice presets on the dialogue layer (pitch, formant, ring-mod robot, megaphone, big hall); no voice cloning of real people |
| "awkward silence: crickets" | mute every layer except dialogue for a range + crickets SFX |
| "earrape" / "deep fried" (meme distortion) | heavy saturation + bitcrush + limiter preset, with a loudness guard so it can't clip the export or hurt ears |
| "make it sound like an old movie" | bandpass + crackle + mono + slight wow |

### D. Stretch (P2)

- Per-stem key change and tempo ("change the key to match", "double-time drums"), via rubberband per stem.
- Generative SFX when the library has no match (text-to-audio). It sits behind a licence decision; see Models.
- "Make the music match the mood of the video": combine the existing vision detections with director presets.

## Architecture

```
command bar → POST /api/projects/{pid}/agent (OpenAI planner, strict schema)
   kind=fx      {targets:[stem_id|"mix"], chain:[{type, params}], preset?, interval?}
   kind=insert  {sfx_query, anchor:{word|beat|time}, offset, gain}
   kind=speech  {op: bleep|cut_fillers|enhance, stem_id, words?}
   kind=duck / loudness …
      ↓ validated + clamped (app/fx_schema.py)
   POST /api/projects/{pid}/fx-preview → candidate job (existing flow)
      ↓ app/fx.py renders the chain (ffmpeg filter graph + numpy DSP); audio-service for ML ops
   compare (current / candidate / mix) → apply (renders into the layer) → undo
```

- **Effect DSL** (`app/fx_schema.py`): about 20 effect types, each with typed params and hard ranges
  (reverb: size 0–1, decay 0.1–8 s, pre-delay 0–200 ms, wet 0–1; delay: time 20–2000 ms, feedback
  0–0.9; pitch: −12…+12 semitones; …). There are about 30 named presets. Effect type and preset
  names are strict enums in the OpenAI schema.
- **Renderer** (`app/fx.py`): ffmpeg filter graph for standard effects (already a dependency). Numpy/scipy
  covers what ffmpeg lacks: convolution reverb with procedurally synthesized impulse responses
  (exponentially decaying filtered noise + early reflections), so there are no IR licensing issues;
  pyroomacoustics (MIT) is an option for room-modelled IRs. Rubberband (ffmpeg `rubberband` filter)
  handles pitch and tempo. Rendering is CPU-only and fast (a 4-min stem in about 1–3 s), so it runs
  in the app backend. Guards: true-peak limiter on output, length preserved, NaN/clip checks.
- **ML ops in audio-service (A40)**: `/v1/words` (ASR with word timestamps), `/v1/enhance` (speech
  enhancement), `/v1/beats` (beats/downbeats), `/v1/sfx/search` (CLAP text → library clip). These return
  timestamps or audio, and the app turns them into candidates.
- **Editor**: the existing compare panel; an "FX" chip row on each track showing its applied chain
  (click to remove or adjust); inserted SFX layers render as normal tracks, tagged `sfx`.

## Models and technologies

| Need | Pick | Licence / notes |
|---|---|---|
| Planning from chat | existing OpenAI planner (`OPENAI_AGENT_MODEL`), strict JSON schema | already integrated |
| Effects | ffmpeg filters + numpy/scipy DSP, rubberband | ffmpeg already used. Spotify **pedalboard** sounds better but is **GPLv3**: only adopt after a licence decision |
| Reverb IRs | procedurally synthesized; optionally pyroomacoustics | MIT; avoids the CC-BY/NC terms on IR libraries |
| Word timestamps (bleep, ums, "when he says X") | **faster-whisper** large-v3 (CTranslate2) + WhisperX-style alignment | MIT/BSD; about 2–5 s for 4 min on an A40 |
| Speech enhancement | **DeepFilterNet3** (denoise); Resemble Enhance as an alternative | MIT/Apache; verify weights |
| Beats / downbeats | **beat_this** (CPJKU, ISMIR 2024) or librosa as fallback | MIT / ISC |
| SFX library + search | CC0 one-shots (curated Freesound CC0 subset, Sonniss GDC bundles per their licence) + **LAION-CLAP** text→audio retrieval | CC0 assets; CLAP CC0/Apache. Ship a curated pack of about 100 meme/comedy sounds |
| Laugh/applause/music detection (for "after the joke") | existing PANNs tagger | already deployed |
| Generative SFX (P2) | Stable Audio Open 1.0 | Stability AI Community licence (revenue cap). Decide before using |
| Voice presets | DSP only (pitch/formant/ring-mod) | no voice cloning; avoids impersonation risk |

## Safety and limits

- Loudness guard on every render: true-peak ≤ −1 dBTP, plus a warning when an edit raises integrated
  loudness by more than 6 dB, so "earrape" can't damage hearing.
- Profanity list is configurable; bleeps are previewed word by word before apply.
- No voice cloning of real people. Inserted SFX come only from the licensed library.
- Edits that change length (tempo) are explicit and can't be mixed into a timeline-preserving chain.

## Phases and owners

| Phase | Work | Owner |
|---|---|---|
| A1 | Effect DSL + presets + renderer + `fx-preview` candidate + planner `kind=fx` (+ rule-based fallback) + tests | agent: fx-engine |
| A2 | Editor: FX chips per track, "more/less", compare panel wiring, insert-layer rendering | agent: fx-editor |
| B | Audio-service: `/v1/words` (faster-whisper), `/v1/enhance` (DeepFilterNet), `/v1/beats` (beat_this) on the A40, with benchmarks | agent: speech-ml |
| C | SFX library (CC0 pack + manifest + licences), CLAP search, insert layers, word/beat anchors, comedy presets | agent: sfx |
| Demo | A 60–90 s creator video: split → vocals reverb → duck music → bleep → vine boom on punchline → TikTok loudness → export | lead |

## Acceptance

- Each P0 ask above works from the command bar, with a preview that can be compared and undone. Renders preserve length and are peak-safe.
- The planner either maps each of a test set of 60 natural-language asks (fixtures) to the right
  kind/preset/targets, or asks a sensible clarifying question. This is tested with a mocked LLM for the schema and live once for quality.
- Bleep: ≥ 95% of profanity in a scripted test clip is caught, with timing within 80 ms.
- SFX insert lands within 50 ms of the anchor word.
