# Recorded benchmark runs

Each directory is one `run_benchmark.py` run that is worth keeping (real model, named hardware).
Mock runs are never recorded here.

| Run | Model | Hardware | Headline |
| --- | --- | --- | --- |
| `2026-11-07-real-model-backend-e2e/` | `facebook/sam-audio-small`, same settings, through Person 4's real backend | Apple M1 Max 64 GB, MPS | Backend e2e PASS (upload → stems → edit → MP4); float32 stems reconstruct to 1.5e-8; 1.1 s/s whole-file, 1.3 s/s with 10 s chunks; timeout/retry verified; prompt sweep: speech prompts 11–13 dB, synthetic sine siren NOT isolated by any siren prompt |
| `2026-11-07-sam-small-mps/` | `facebook/sam-audio-small`, span predictor and re-rankers off, 1 candidate, float32 | Apple M1 Max 64 GB, MPS, torch 2.15 | SI-SDRi: siren-from-speech 11.6 dB, one-of-two-speakers 10.2 dB, guitar-from-mix 12.9 dB; 4–7.5 s processing per second of audio (6 s clips), 1.5 s/s on a 15 s clip after warm-up |

Fixtures are synthetic but use real spoken speech (macOS `say`), so the two-speaker and
speech-plus-siren numbers are meaningful; the music fixture is tones and noise bursts. No
GPU numbers exist yet; the first CUDA run should be recorded here the same way.
