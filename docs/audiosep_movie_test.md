# Real movie clip: mute the dog, keep the actor

The user supplied `https://www.youtube.com/watch?v=WddwSrJNQgg` as the next
AudioSep test after reporting that the synthetic-demo edit changed the voice.
YouTube identifies it as a roughly 21-second *Dumb and Dumber To* dog scene.
The downloaded source is retained separately. No synthetic voices are added.

The original recording has video duration 20.559044 s (609 VFR frames) and audio
duration 20.642540 s, stereo AAC at 44.1 kHz, both starting at zero. The test input
is normalized to 620 CFR frames at 30 FPS and 992,000 stereo samples at 48 kHz
(20.666666667 s). A brief held last frame and silent audio tail retain the complete
source rather than truncating the scene. The normalized original is stored as
H.264/ALAC and the lossless inference input is `mixture.wav`.

## Reproduce

Use the existing isolated AudioSep checkout, official weights and environment
from [the initial GPU experiment](audiosep_dog_demo.md). No SAM inference is used.

```sh
cd /workspace/SoundCanvas-dog-comparison-8a5cb33
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/cache/torch
/workspace/venvs/dog-audiosep/bin/python scripts/run_audiosep_dog_demo.py \
  --input-dir /workspace/experiments/dumb-and-dumber-dog-20261114 \
  --output /workspace/experiments/dumb-and-dumber-dog-20261114/audiosep \
  --audiosep-root /workspace/vendor/AudioSep --chunk-seconds 10
```

One real AudioSep `dog barking` run, float32, seed 42, one candidate/pass, 10-second
windows with existing 2-second overlaps and batch 1. The engine creates the exact
mixture-derived residual and existing editor export. No reference stems exist for
this movie; objective barking attenuation and actor preservation cannot be scored.

After copying the result back to the Mac, run:

```sh
python scripts/compare_audiosep_suppression.py \
  --directory /path/to/SoundCanvas-Dumb-and-Dumber-Dog-Test
```

## Voice preservation candidates

Full subtraction also removes speech detail that leaks into the barking estimate.
The previous demo's near-unity voice projection gains missed the user's audible
voice-timbre complaint. New candidates are proposed for listening, not automatically
promoted as perceptually better.

The optional `sam_audio_service.suppression.suppress_target` helper reuses existing
`mask_refine`: retain original mixture phase, suppress only the high target-ratio
bins (soft knee .85), and smoothly reduce processing when the target's own activity
is weak (-36 to -24 dB relative to its peak). These ratios are heuristics, not
calibrated confidence or a speech detector. It consumes only the mixture and model
estimate; no clean reference or hand-selected bark timing conditions processing.

- `full-subtraction`: original full-removal method, HQ re-encode for comparison.
- `voice-first`: the optional mask/activity method at .9 removal strength; fully
  target-dominated regions retain .1 amplitude, trading some barking for voice detail.
- `dog-muted`: the mask/activity method at full removal strength.

The alternatives run locally on CPU, reuse the saved real estimate, retain stereo
and exact sample counts, and export float32 WAVs plus copied video/HQ AAC 192k.
Each removed stem allows listening for accidentally removed actor vocalizations.
AudioSep can confuse actor imitations or other effects with barking. Speech and
barking that share frequencies remain difficult. A person must review the clip
before calling any candidate successful.

## Verification

55 focused audio tests passed, including all original AudioSep tests, STFT/mask
regressions and seven new suppression cases. The new cases cover no-op identity,
quiet passages with model speech leakage, real target reduction, invalid controls,
and timeline mismatch. Existing service defaults and API contracts are unchanged.
The movie runner now accepts arbitrary positive-duration 48-kHz inputs and
optional source references, preserving the original 12-second fixture behavior.

## Recorded execution and exports

Real L4 CUDA inference succeeded, taking 0.504 s after a 66.736 s first load,
with 2.217 GiB peak allocated VRAM. The model used 1,079,728 stereo samples at
48 kHz (22.494333333 s): preparation had appended excess **zero-only** padding
because sample trimming ran before resampling. No scene audio was stretched or
lost. The full execution files and original report are retained under
`audiosep/full-run/`; the input is `inference-mixture.wav` there.

Exports and source audio were cropped to the intended 992,000-sample movie
interval without a second model call. The cropped report records both inference
and export lengths/hashes explicitly. For future preparation, put
`aresample=48000` before `apad,atrim=end_sample=992000` in the audio filter chain.
The aligned source SHA256 is
`7533510549ac6932f2437290e65925601fb9b9deb65b756708a502a1a57616f1`.

All three candidate MP4s were verified on the Mac: 620 H.264 frames, stereo AAC,
20.666666667 s audio and video, origins both 0. All six candidate WAVs are
992,000 samples per channel at 48 kHz. Removed + retained audio reconstructs the
input with maximum error at most 1.49e-8. The transfer archive and source result
manifest were checksum verified. The GPU returned to 0 MiB after the one real
run. Alternative processing and HQ exports were generated entirely on the Mac.

The artifacts are under `~/Downloads/SoundCanvas-Dumb-and-Dumber-Dog-Test`, with
`voice-preserving/comparison.html` for time-matched listening and per-candidate
removed WAVs. The source movie has no isolated reference stems. These checks
establish execution, synchronization and reconstruction; they do not establish
that only the dog was removed. Listening to the actor and removed stems remains
required, especially for human vocalizations that resemble barking.
