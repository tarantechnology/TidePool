# Plan: song upload → per-instrument tracks, with model routing

Status: in progress on branch `audio/music-stems`. Owner: Person 1 (audio).

## Goal

A user uploads a whole song (audio file, or a music video) and gets one editable track per
instrument (vocals, drums, bass, guitar, piano, other) in the existing editor, with the
existing volume / mute / solo / automation tools. A model router picks the right separator:
dedicated music-source-separation (MSS) models for songs, the prompted text model
(AudioSep / SAM Audio) for dialogue, video soundtracks and arbitrary sounds.

Why not AudioSep for songs: text-queried models are trained on AudioSet/VGGSound captions
and are weak on music stems. Dedicated MSS models (Demucs v4, BS-/Mel-RoFormer) are trained
on MUSDB-style multitracks and are much better on them. Phase 0 measures this on our A40
and does not assume it.

## Architecture

```
app /process {mode} ──► audio-service /separate {options.mode}
                              │
                         routing.route()            ◄── content classifier (AudioSet tagger, lazy)
                        /                \
          mode=music_stems              mode=prompted
          StemSplitter.split()          Engine (existing): Backend.separate() per prompt
          (Demucs / RoFormer chain)     (AudioSep or SAM Audio)
```

### 1. `StemSplitter` (audio-service, new; separate from the prompted `Backend`)

The prompted `Backend` shape is one mono signal and one prompt in, one target out. A splitter
is one stereo mixture in, N named stems out. Forcing it into `Backend.separate` would mean
running the model N times or caching outputs. So it gets its own small interface in
`sam_audio_service/stems/`:

- `StemSplitter.load()`, `info()`, `stems: tuple[str, ...]`, `sample_rate`,
  `split(mixture: float32[channels, n]) -> dict[str, float32[channels, n]]`.
- Implementations: `mock` (deterministic band-split filters for CI; not separation),
  `demucs` (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`), and `chain` (a RoFormer vocal model,
  then Demucs on the instrumental) if Phase 0 shows it is worth the cost.
- Native stereo. Resample 48 kHz ↔ model rate inside the splitter. Long songs use the
  model's own overlap-add chunking.
- **Mixture consistency:** after splitting, `error = mix - sum(stems)` is spread over the
  stems in proportion to their energy, so unmuting every track reproduces the upload exactly.
  The effect on SDR is reported.
- Lazy, loaded once, alongside the prompted backend on the same GPU (Demucs needs about 1–3 GB).

### 2. Router (`sam_audio_service/routing.py`)

`route(audio, prompts, options, hints) -> RouteDecision{mode, engine, model_id, stems, reason, scores}`

1. `options.mode` is `prompted` or `music_stems`: obey it.
2. `auto` (default):
   - No prompts → `music_stems` (all stems). Warn if the content does not look like music.
   - Every prompt maps onto the stem vocabulary (aliases: vocals/voice/singing/singer/lead
     vocal → vocals; drums/percussion/beat → drums; bass/bassline → bass; guitar; piano/keys;
     instrumental/accompaniment/backing track/karaoke → instrumental = sum of non-vocal stems)
     **and** the content classifier says music → `music_stems` returning the requested stems
     plus one `residual` (the sum of the rest). Stem names keep the caller's wording.
   - Otherwise → `prompted` (unchanged behaviour). Mixed prompts ("vocals" + "dog barking")
     → prompted, with the reason recorded.
3. Content classifier: an AudioSet tagger (AST or PANNs, whichever is lighter and accurate
   enough in Phase 0) on a few windows sampled across the file, giving `music` and `speech`
   scores. If it is unavailable (mock/CI, or not installed), the decision uses the prompt
   rules alone and records `scores: null`.
4. `hints.source_kind` (`audio` | `video`) from the app is a tie-breaker only.
5. **Intent from natural language (OpenAI planner, app side).** `app/agent.py` already turns
   editor instructions into bounded operations with OpenAI Structured Outputs (Responses
   API, strict JSON schema). That schema gains `separation_mode` (`auto|prompted|music_stems`)
   and `stems`, so "split this song into tracks" or "make a karaoke version" becomes an
   explicit mode. The app forwards it as `options.mode`. The LLM decides *intent*; it cannot
   hear the audio, so the *content* signal stays with the GPU classifier. The service router
   is the authority and works with no OpenAI key (CI, offline). The planner is a better front
   end for it, not a dependency.

Every response carries `routing` so decisions can be explained and debugged. There is also
a dry-run endpoint, `POST /v1/route` (multipart, same as `/separate`), which returns only the
decision. The editor uses it to suggest "This sounds like a song: split into tracks?"

### 3. Contract additions (all optional; existing callers unaffected)

Request `metadata.options`:
- `mode: "auto" | "prompted" | "music_stems"`. Default is `AUDIO_ROUTING_DEFAULT`, which is
  `auto`. In `auto`, non-music prompts route exactly as today.
- `stems: list[str] | null`: subset of the splitter vocabulary (music_stems only).
- `stem_model: str | null`: override `AUDIO_STEM_MODEL`.

Response:
- Top-level `routing: {mode, engine, model_id, reason, scores, stems}`.
- Music stems have `role: "target"`, `label` = canonical stem (`vocals`), and `name` = the
  caller's wording or the title-cased canonical name (`Vocals`). A **full split returns no
  residual** (the stems already sum to the mix). A subset split returns exactly one `residual`.
- `model.backend` reports what actually ran (`demucs`, `chain`, `audiosep`, `sam`, `mock`).

Capabilities: `music_stems: {available, model_id, stems, max_duration_seconds}` and
`routing: {default_mode, classifier}`.

Errors: requesting `music_stems` with no splitter configured gives 400 `UNSUPPORTED_OPERATION`.
Unknown stem names give 400 `INVALID_REQUEST` listing the valid ones.

Config: `AUDIO_STEM_SPLITTER` (`off|mock|demucs|chain`), `AUDIO_STEM_MODEL`,
`AUDIO_ROUTING_DEFAULT`, `AUDIO_ROUTER_CLASSIFIER` (`off|ast|panns`),
`AUDIO_ROUTER_MUSIC_THRESHOLD`, `AUDIO_STEM_SHIFTS`, `AUDIO_STEM_OVERLAP`.

### 4. App backend (`app/`)

- `Process.mode` (`auto|prompted|music_stems`) and optional `stems`, forwarded in
  `options`. Empty prompts are allowed when `mode` is `music_stems` (or `auto` on the http
  adapter, in which case the service decides).
- `source_kind` hint from the project's upload (audio-only vs video).
- The `fake` adapter emits named instrument stems for `music_stems` (copies, clearly marked as mock).
- Planner (`app/agent.py`): add `separation_mode` + `stems` to the structured-output schema
  and validate them; the rule-based frontend parser gets the same intents ("split",
  "stems", "karaoke", "instrumental") so it works without a key.
- Persist `routing` in the job output; `GET /api/projects/{pid}/route` proxies `/v1/route`.
- Projects without a video stream already work (audio-only path). Verify, and fix if not.

### 5. Editor (`frontend/`)

- Upload accepts audio files (`audio/*`) as well as video. With no video, the video panel
  shows a waveform/placeholder instead.
- A "Split into tracks" primary action (mode `music_stems`) next to the prompt bar, plus
  an auto-suggestion when `/route` says the upload is music.
- Stems render as named tracks (Vocals, Drums, Bass, …). Existing edit tools apply unchanged.
  "Remove vocals" on a split project mutes the vocals track and does not need a residual.
- Show the routing reason/model in the job status card.

## Phases and owners

| Phase | Work | Where |
|---|---|---|
| 0 | Benchmark: MSS candidates on 50 MUSDB clips (SDR per stem, speed, VRAM), AudioSep text-prompt baseline on the same clips, classifier accuracy (music vs speech/SFX) | A40 only |
| 1 | audio-service: splitter, router, contract, mock, tests | worktree + A40 validation |
| 2 | app + frontend: mode, upload audio, split action, tracks | worktree |
| 3 | End-to-end on the A40: real song → tracks in the editor; docs | A40 + local editor via tunnel |

## Acceptance

- Full split of a 4-minute song on the A40 finishes in under 60 s with the chosen model.
- Mean MUSDB SDR for the chosen model beats the AudioSep text-prompt baseline on every stem,
  with numbers in `audio-service/benchmarks/results/2026-11-14-music-stems/`.
- Router: music vs non-music accuracy ≥ 95% on the labelled clip set. Every existing
  prompted test passes unchanged.
- Sum of the returned stems equals the decoded mix (max abs error < 1e-4).
- Mock path is fully covered by CPU tests. No model weights are downloaded on CI or on dev Macs.

## Phase 4: "Separate the rest" (nested split of the residual instrument stem)

**Status: dropped (2026-11-14).** It was built and benchmarked on the A40, then removed before
merge because it was not useful enough to ship. Of the licence-clean models, only Demucs
`htdemucs_6s` separated anything from the `other` stem, and only guitar and piano/keys, at
roughly +2 dB SDR over the input; organ, synth, strings, brass and woodwinds were not
separable. Banquet (the strongest query-based option) has CC BY-NC-SA weights, AudioSep
returned phantom instruments, and SAM Audio was never evaluated. The per-stem instrument
tags that came out of this work were kept (see the evaluation below). The plan is kept for
reference.

The main split stays as is (vocals, drums, bass, guitar…, plus a named "other"). The user can
ask to split *that one track* further. A nested model separates the instruments inside it.

**Contract (audio-service).**

Request: `POST /separate` with the parent stem's audio and
`options.mode = "music_substems"`, plus:
- `options.parent_label` (`"other"`), used for logging and naming.
- `options.stems`: a list of instrument names, or null. Null means detect them with the
  instrument tagger and take the present ones, up to `AUDIO_SUBSTEM_MAX` (default 4).

Response:
- One target stem per instrument found: `label` = canonical sub-instrument (`piano`,
  `keys`, `synth`, `strings`, `brass`, `woodwinds`, `organ`, `guitar`, …), a human `name`, and `tags`.
- Exactly one `residual`, named `Rest`. With the targets it sums to the input (mixture consistency).
- Instruments that come back near-silent are folded into `Rest`, with a warning.

`routing`: `{mode: "music_substems", engine: <nested model>, …}`.

Config: `AUDIO_SUBSTEM_MODEL` (chosen by the Phase 4 benchmark), `AUDIO_SUBSTEM_MAX`.

**Contract (app).**

Endpoint: `POST /api/projects/{pid}/stems/{sid}/split` (202, `{job_id}`), body `{stems?: string[]}`.
- Allowed on any music target stem; the editor offers it on the `other` stem.
- On success, the parent row is replaced by its children. Each child has `source_stem_id` = the parent's
  id and inherits the parent's start_offset, mute and solo, so the mix is unchanged.

Editor: a "Separate the rest" action on the track; the children render grouped under the
parent's name; undo works as for other replacements.

## Stem labels / instrument ID evaluation (2026-11-14)

Data: 20 BabySlakh songs (CC-BY 4.0, 16 kHz, per-instrument stems + MIDI programs) mixed and
separated by our chain (BS-RoFormer vocals -> htdemucs); 144 MUSDB18 previews (separated the
same way); 240 synthetic "other" stems (1-4 BabySlakh instruments + drums/bass, through the
chain) for presence calibration. Scripts were run ad hoc on the pod (`/workspace/instid`).

Why tags were poor: (1) one free-for-all whitelist, so artefacts became "Gong"/"Didgeridoo" on
bass stems and isolated vocals (which AudioSet reads as "A capella"/"Mantra"/"Chant"/"Speech",
generic "Singing" ~0.07) came out as "Chant"/"Humming"/"Yodeling" or nothing; (2) the trivial
filter removed the only confident class on many stems ("Drum" on drums: 38/140 MUSDB drum stems
had no tags); (3) one global 0.1 threshold, while family scales differ by 10x (Guitar 0.3-0.9,
Brass/Mallets 0.02-0.2 when audible).

| PANNs, old -> new | old | new |
|---|---|---|
| MUSDB vocals: top tag is a singing/rap label (not none/Chant/Humming/Yodeling) | 101/140 | 140/140 (31 plain "Singing", 8 "Spoken word") |
| MUSDB drums / bass stems with a tag | 106 / 114 of 144 | 144 / 141 |
| BabySlakh other: name's family is in the song | 17/20 | 17/20 (AST: 16 -> 19/20) |
| BabySlakh other presence (P / R, 9 families) | 0.81 / 0.39 | 0.82 / 0.40 (AST 0.82 / 0.46) |
| BabySlakh bass subtype | 18/19 | 18/19 |

Taggers compared (clean BabySlakh sources, top-1 family of 9): PANNs 0.74, AST 0.78,
CLAP music+speech 0.80, CLAP music-only 0.11 (collapses to "woodwinds"). Synthetic presence F1
(per-family thresholds, 2-fold CV): PANNs 0.59, AST 0.63, CLAP 0.57, ensembles <= 0.63. CLAP is
poor on separated vocals ("rapping"/"synth vocals" bias) and bass (9/19). Gender on MUSDB vocals
has no reliable ground truth (F0 does not separate high male rock singers from women); PANNs
Male-vs-Female with a 0.05 margin looked right on ~22/27 songs whose singer we could name.
Default stays PANNs (already loaded, ~0.8 s warm for 4 stems of a 4-min song on a busy A40;
AST ~2.6 s). Essentia's instrument models were not used: CC BY-NC-ND.
