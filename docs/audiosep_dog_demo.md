# AudioSep: keep the conversation, quiet the dog

The experiment was narrowed to **AudioSep only** at the user's request. SAM Audio
and SAM 3.1 inference are excluded. AudioSep receives the text `dog barking` and
mixture audio only. No masks, frames, anchors, or reference stems condition it.

## Reproduce

Start from TidePool commit `8a5cb33`. These scripts run from a feature branch
and write all media outside Git. Render the authored dialogue on CPU with the Apache-2.0 Kokoro model using
`scripts/synthesize_demo_dialogue.py` (isolated `kokoro-onnx` environment; explicit
ONNX model and voices paths). Then prepare the controlled input:

```sh
python scripts/prepare_dog_demo.py --speech-dir /path/to/kokoro-speech \
  --output /path/to/fresh-demo-directory
```

Preparation needs FFmpeg, curl, NumPy and the generated Kokoro speech WAVs.
Voices are `af_heart` and `bm_george`, synthesized locally with CPUExecutionProvider.
The model card and complete model/runtime provenance are retained. Apple System
Voices are excluded from the public demo because their license restricts sharing.
The 12-second, 48-kHz mono mixture combines two authored synthetic voice turns
and CC0 recorded barking at +2 dB RMS relative to combined speech. One common gain
limits its peak to 0.85. Clean references, source hashes, and provenance are saved.
Gustavo Fring's Pexels footage shows two people and a visible dog, but its audio is
replaced; it is **not a natural synchronized recording**. Original video audio
uses ALAC; `mixture.wav` is the authoritative inference input.

Connect to the authorized Pod:

```sh
ssh y8ervobcjwwbyc-64410f3b@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The existing `/workspace/SoundCanvas` contained uncommitted work. A separate
tracked-source snapshot is used at `/workspace/SoundCanvas-dog-comparison-8a5cb33`,
preserving that checkout, previous experiments, and authentication. Model downloads
and environments remain under `/workspace`; credentials never enter artifacts.

Official AudioSep checkout: `https://github.com/Audio-AGI/AudioSep`, revision
`944583f18b84589dc965de3ad77525c945334252`, with the repository's existing
`audio-service/patches/audiosep-checkpoint-compat.patch`. Official checkpoints come
from `Audio-AGI/AudioSep` (Hugging Face **Space**, not a model repository), revision
`0807a713283035d8b0d0628f1b8e3d6580459f7b`:

| File | Bytes | SHA256 |
|---|---:|---|
| audiosep_base_4M_steps.ckpt | 1264844076 | f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c |
| music_speech_audioset_epoch_15_esc_89.98.pt | 2352471003 | 51c68f12f9d7ea25fdaaccf741ec7f81e93ee594455410f3bca4f47f88d8e006 |

Both local file hashes were verified against official Hub ETags. The original
SAM checkpoints are retained and are not used for this experiment.

The separate runtime is `/workspace/venvs/dog-audiosep`, created with
`python -m venv --system-site-packages` to reuse template Torch 2.8.0+cu129,
torchaudio 2.8.0+cu129, and torchvision 0.23.0+cu129. An explicit constraint file
pins all three. Use pip in this environment: uv attempted to resolve the custom
cu129 build from PyPI instead of reusing the template package.

Runtime dependencies include NumPy 1.26.4, transformers 4.37.2, Hub 0.36.2,
lightning/pytorch-lightning 2.5.6, torchmetrics 0.11.4, torchlibrosa 0.1.0,
librosa 0.11.0, timm 0.9.16, scipy, soundfile, einops, ftfy, regex, pyyaml,
webdataset, braceexpand, wget, h5py 3.16.0, and pandas 2.3.3. Pin NumPy and pandas together so
pandas does not upgrade NumPy to 2.x. Install `./audio-service` editable
into this environment. The upstream conda file is not used unchanged: it targets
Python 3.10 and older CUDA/Torch. Exact final package versions are captured in
experiment evidence.

After transferring the controlled inputs to a fresh experiment directory:

```sh
cd /workspace/SoundCanvas-dog-comparison-8a5cb33
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/cache/torch
/workspace/venvs/dog-audiosep/bin/python scripts/run_audiosep_dog_demo.py \
  --input-dir /workspace/experiments/dog-audiosep-kokoro-20261114 \
  --output /workspace/experiments/dog-audiosep-kokoro-20261114/audiosep \
  --audiosep-root /workspace/vendor/AudioSep
python scripts/report_audiosep_dog_demo.py \
  --directory /workspace/experiments/dog-audiosep-kokoro-20261114
```

The runner requires real CUDA, acquires the existing exclusive experiment GPU
lease, loads the real adapter, and calls TidePool's `Engine.separate` with one
pass, no chunking, no refinement, and float32. The existing `app.media.export`
mutes the target and keeps the mixture-derived residual. There is no oracle
processing, reference-based cleanup, or model-specific level normalization.

## Evaluation

The runner records model load/inference time, peak allocated/reserved VRAM,
full-length float32 WAVs, exact reconstruction error, source hash, and MP4 probes.
Reference stems are read only after inference for evaluation. Joint waveform
projection estimates gain retained from each voice and barking. Aggregate speech
SDR and SI-SDR improvement also penalize distortion or residual interference.
Projection can miss distorted/delayed bark remnants and is not a perceptual score.
Listening by a person is required before judging demo quality or making model
superiority claims. The listening page supports matched-volume before/after
playback and isolated-target audition.

The source audio is authored synthetic dialogue plus one repeated real barking
recording, limiting generalization. The depicted people did not speak these
lines. A natural recording with two human speakers should be the next validation.
No speech-recognition intelligibility score or subjective listening score is
invented.

## Regression checks

Before experiment completion: backend 116 passed, 2 skipped; audio service
345 passed (including the three original AudioSep tests); vision 42 passed,
5 deselected. Three additional metric tests check source-gain attribution,
identity baseline, and rejection of invalid references/timeline drift.

## Real L4 result (2026-11-14)

One real CUDA inference succeeded on the Kokoro fixture. The exact input WAV
SHA256 is `42ca152f95971b0730bd37f3d13e1f6884020637f91fbb65ef38da17ab41cf46`.

| Measurement | Result |
|---|---:|
| Model load including first imports/read | 84.111 s |
| Inference on 12 seconds (CUDA synchronized) | 0.384 s |
| Peak allocated / reserved during inference | 2.461 / 3.543 GiB |
| Speech SDR, original → edited | -2.000 → 18.780 dB |
| Speech SI-SDR improvement | +20.701 dB |
| Voice A / B retained waveform gain | -0.241 / -0.028 dB |
| Bark waveform-projection reduction proxy | 60.589 dB |
| Isolated barking SI-SDR improvement | +18.759 dB |
| Target + residual max reconstruction error | 1.49e-8 |

The bark projection figure is **not** a perceptual attenuation measurement:
distorted remnants evade waveform projection. The speech SDR is the more
conservative aggregate reconstruction measure. Listening remains pending and no
sound-quality preference is inferred. No further quality sweep was run.

The edited MP4 has 180 H.264 frames, 12.000-second audio and video, both origins
0.000 seconds, AAC mono at 48 kHz. Both WAV stems have exactly 576,000 samples.
The runner used the existing adapter, engine and `app.media.export`; target muted,
residual retained. No source references were used to alter the output. The model
process exited and the GPU returned to 0 MiB allocated by processes.

The earlier initialization attempt stopped at an optional OpenCV import before
model load or inference. The runner now loads the existing GPU lease helper
without importing the vision package. The intermediate Apple-voice fixture was
excluded from the public demo and replaced before any successful inference.
No SAM Audio or SAM 3.1 inference was run.
