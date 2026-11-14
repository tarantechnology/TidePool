# Editor fixture (for Person 3)

**Status: proposed, awaiting team agreement.** Everything here is Person 1's proposal for
the stem payload the editor consumes; field names can still change.

The editor is not in this repository yet, so cross-team Test 1 (editor -> backend -> audio)
cannot be executed. This directory is the stand-in: a committed example of what the audio
service returns, plus a generator for playable files you can drop straight into a dev
server.

```
fixtures/editor/
  example_response.json   committed. Exactly what POST /separate returns (Contract C),
                          audio_base64 TRUNCATED to a placeholder. Illustrative numbers.
  README.md               this file
fixtures/generated/editor/   written by the generator, git-ignored
  source.wav              the 48 kHz stereo mixture (what Person 4 uploads to us)
  speech.wav siren.wav residual.wav
  stems.json              Contract C, with audio_url instead of audio_base64
```

## Generate the playable fixture

```bash
cd audio-service
.venv/bin/python scripts/make_editor_fixture.py            # -> fixtures/generated/editor/
.venv/bin/python scripts/make_editor_fixture.py --out ../web/public/fixtures/editor
```

Needs FFmpeg on `PATH`; no GPU, no model checkpoint, no network. Re-runnable: it rewrites
the directory. `--seconds` and `--prompts` change the mixture and the stem list.

## The mock output is NOT real separation

The generator runs the service's `AUDIO_BACKEND=mock` backend, a band-pass filter chosen
from the prompt word. The stems are band-limited copies of the mixture: `siren.wav` is the
500-2200 Hz band of the mix, not an isolated siren. Treat the audio as test tone material
and the JSON as the contract.

Every mock response carries `model.is_mock: true` and a warning in `warnings[]`; the
manifest also sets `fixture.is_real_separation: false`. **Surface `model.is_mock` in the UI
(a "demo separation" badge) whenever it is true** so a mock run is never mistaken for a
model result. Real separation requires a CUDA GPU and approved access to the gated
`facebook/sam-audio-*` checkpoints; neither is available on this machine yet, so no real
SAM Audio output exists to ship here.

## Loading the stems

```js
const manifest = await fetch('/fixtures/editor/stems.json').then(r => r.json());
const tracks = manifest.stems.map(s => ({
  id: s.stem_id,                 // stable within one separation response
  name: s.name,                  // user-visible label ("siren", "residual")
  role: s.role,                  // "target" | "residual"
  url: new URL(s.audio_url, '/fixtures/editor/').href,
  startTime: s.start_time,       // always 0.0
  duration: s.duration,
}));
```

`audio_url` exists **only in this fixture** so a browser can load the files directly. Over
HTTP the service returns `audio_base64` (a whole WAV, base64) or, with
`options.return_mode: "file"`, an `audio_ref` `file://` URL. In the real product Person 3
never calls this service: Person 4 stores the decoded stems and exposes
`GET /api/projects/{id}/stems/{stem_id}/audio` (bearer token required, fetch as a blob) and
their own stem objects (`id`, `name`, `duration`, `start_offset`, `volume`, `muted`, `solo`,
`automation`, `audio_url`). The fields below are what we guarantee about the audio behind
whichever of those URLs you use.

## Staying in sync

Every stem in one response describes the **same window of the source timeline**:

| Guarantee | Value |
| --- | --- |
| `start_time` | always `0.0` -- stems are full-timeline, never clipped to the event |
| `duration`, `sample_count` | identical for every stem **and** equal to `source.*` |
| `sample_rate` | `48000` for every stem and the source |
| `channels` | matches the source (1 or 2); identical across stems |
| `format` | `wav/pcm_s16le` (RIFF WAV, signed 16-bit little-endian PCM) |
| timeline origin | second `0.0` = first sample of the uploaded media |

So: load every stem at offset 0, start them together, and no per-stem offset, trim, padding
or resampling is needed. `sample_count` is the authoritative length -- compare it rather
than `duration` (a float) when checking alignment. Sync to one clock (one `AudioContext`,
or the `<video>` element's `currentTime`); do not let each `<audio>` element run free.

If a future response ever carries a non-zero `start_time`, that is a contract change we
have to agree first (Open Question 5 in the handoff doc) -- assert on it rather than
silently trusting it.

`stems[].spans` (when present) is `[[start, end], ...]` in source-timeline seconds where
the model believes the target is active. It is a **hint for the UI** (where to draw blocks
or where to auto-place keyframes), not a trim instruction: the audio outside a span is
present in the file and is usually near-silent.

## Mute / solo / volume

The stems sum back to the mixture sample-exactly (float), and within a few 16-bit LSBs once
each stem is quantised to PCM16:

```
sum(stems) == source        # speech + siren + residual == source.wav
```

`residual` (`role: "residual"`, label "everything else") is literally
`mixture - sum(targets)`. That is what makes the editor's semantics exact:

- **mute one stem** = original minus that stem. Muting `siren` leaves speech + residual.
- **solo one stem** = only that stem; the mix is the stem alone.
- **unity mix** = all stems at `volume: 1`, unmuted, summed -> bit-identical to the original
  (to within PCM16 rounding). Do **not** apply any normalisation or make-up gain in the
  default state, or the "nothing changed yet" mix will not match the original.
- **volume / automation** multiply the stem before the sum. Person 4's range is `0-2`
  (`1` = unity, `2` = +6 dB) with linear interpolation between automation keyframes, in
  project timeline seconds.
- **clipping**: summing N stems each boosted above unity can exceed full scale. The targets
  overlap in content when prompts overlap (the mock especially), so a limiter on the master
  bus is reasonable, but it must be bypassed at the unity mix for the point above to hold.
- Edits are **non-destructive** everywhere: Person 4 stores them as stem rows and applies
  them at export; the WAV files never change. The editor should keep the same model.

What we deliver is therefore: one WAV per prompt plus one residual, all the same length as
the source, summing to the source. What we *assume* the editor needs beyond that
(waveform peaks, a loudness figure per stem, a `color`/ordering hint, chunked or streaming
delivery for long files) is **not implemented** -- Open Questions 6 and 7 in
`docs/audio_integration_handoff.md`. Tell us which of those you actually want.
