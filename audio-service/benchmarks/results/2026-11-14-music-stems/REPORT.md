# Phase 0: music stems — which separator, which router classifier (A40, 2026-11-14)

Harness: `audio-service/benchmarks/music_stems.py` (runs on the pod). Venv: `audio-service/scripts/setup_music_stems_venv.sh`.
Raw results in `raw/`:
- `all_summary.json`: aggregates.
- `all_per_clip.csv`: one row per clip, stem and variant.
- `sep_*.json`: timing and VRAM per candidate.
- `fullsong_*.json`, `classify_*.json` (+ `_per_file`).

Host: RunPod NVIDIA A40 46 GB, driver CUDA 13.2, torch 2.8.0+cu128, Python 3.12.3.

Contention:
- Every MUSDB separator run and both classifier runs had the GPU to themselves.
- The htdemucs, htdemucs_ft and htdemucs_6s full-song timings probably did too.
- **The chain full-song timing overlapped other GPU work and is unreliable.**

## Recommendation

| Decision | Choice | Why |
|---|---|---|
| Default splitter | **htdemucs** (fast) or **htdemucs_ft** (+0.1–0.5 dB), shifts=0, overlap=0.25, projection on | MIT. 4-min song in about 10 s (htdemucs) or 32–37 s (ft). Beats AudioSep by 7–10 dB on every stem. |
| Guitar/piano tracks | htdemucs_6s as an option | About as fast as htdemucs; bass −1.3 dB and drums −0.3 dB vs ft. |
| Chain (ep_317 vocals → htdemucs_ft) | HQ mode, not the default | +3.1 dB vocals, +1.3 dB other. Too slow through audio-separator (stage 1 alone about 200 s per song). ep_317 licence unverified. |
| Quality-upgrade candidate | ZFTurbo 4-stem BS-RoFormer fp16 (MIT) | +2.1 dB vocals, +1.2 drums, +1.3 other, −1.3 bass. fp16 matches fp32. |
| shifts | 0 | shifts=1 changes the result by ±0.06 dB. |
| Mixture consistency | Always on | Raw max error up to 0.43; after projection 1.8e-7. Costs ≤ 0.2 dB. |
| Router classifier | **PANNs Cnn14**, mean P(Music) over 4×10 s, threshold ≥ 0.70 | 100% on 51 music / 87 speech / 100 sound-effect files. No-error threshold range 0.59–0.84. 8 s load, 0.12 s per file, 1.1 GB. |
| Speech over music | P(Speech) ≥ 0.5 → prompted model | P(Music) alone calls 20/20 of these clips music. The gate catches 20/20 and wrongly flags 1/51 songs. |

**Licences.**
- BS-Roformer-SW: **cannot ship**. It is jarredou's BS-ROFO-SW; the Hugging Face licence is "unknown", and one integrator lists CC-BY-NC-SA-4.0. Included only as a reference ceiling.
- Demucs: MIT.
- ZFTurbo 4-stem: MIT (a release asset of the Music-Source-Separation-Training repo).
- viperx ep_317 and Kim Mel-RoFormer: not verified.

## 1. Separators on MUSDB18 (50 test previews of 6.8 s, 48 kHz stereo)

Pipeline as in the service: 48 kHz → julius → 44.1 kHz → model → 48 kHz.

Metrics:
- **uSDR**: SDR over the whole clip, both channels, no BSSEval filter (MDX'21 SDR).
- **cSDR**: the same on 1 s chunks; median over non-silent chunks.

Cells are mean uSDR / median cSDR in dB. Time is per clip, warm (mostly fixed overhead on 7 s clips).

| Candidate | vocals | drums | bass | other | s/clip | VRAM MB |
|---|---|---|---|---|---|---|
| htdemucs | 8.47 / 8.17 | 9.81 / 9.96 | 8.51 / 9.77 | 5.01 / 5.54 | 0.79 | 778 |
| htdemucs shifts=1 | 8.48 / 8.14 | 9.76 / 9.84 | 8.56 / 9.72 | 5.03 / 5.39 | 0.77 | 854 |
| StemForge default (htdemucs s=1, no norm, clip) | 8.48 / 8.12 | 9.79 / 9.86 | 8.57 / 9.76 | 5.01 / 5.51 | 0.92 | 854 |
| htdemucs_ft | 8.61 / 8.37 | 9.80 / 10.06 | 8.98 / 9.27 | 5.09 / 5.36 | 2.08 | 1330 |
| htdemucs_ft shifts=1 | 8.61 / 8.43 | 9.86 / 10.02 | 8.96 / 9.11 | 5.11 / 5.41 | 1.97 | 1290 |
| htdemucs_6s (gtr+pno+other→other) | 8.61 / 8.82 | 9.47 / 9.51 | 7.69 / 8.04 | 4.62 / 5.00 | 0.91 | 830 |
| ZFTurbo 4-stem fp32 (StemForge) | 10.67 / 11.12 | 10.98 / 11.27 | 7.68 / 7.57 | 6.36 / 6.38 | 1.71 | 2364 |
| ZFTurbo 4-stem fp16 | 10.67 / 11.12 | 10.98 / 11.27 | 7.68 / 7.57 | 6.37 / 6.38 | 0.95 | 1978 |
| chain 317 → htdemucs_ft | 11.72 / 12.24 | 10.15 / 10.55 | 9.10 / 9.44 | 6.42 / 7.16 | 5.02 | 2286 |
| chain 317 → htdemucs_6s | 11.72 / 12.24 | 9.98 / 9.91 | 7.84 / 8.21 | 5.56 / 6.31 | 3.87 | 1840 |
| BS-Roformer-SW (no licence) | 11.30 / 11.68 | 11.55 / 11.43 | 9.24 / 9.11 | 7.17 / 8.01 | 1.91 | 1648 |
| ep_317 vocals overlap 4 | 11.72 / 12.24 | – | – | – | 2.72 | 1806 |
| ep_317 vocals overlap 8 | 11.74 / 12.25 | – | – | – | 4.69 | 1806 |
| Mel-RoF Kim vocals | 11.34 / 11.64 | – | – | – | 0.95 | 2292 |

**StemForge.**
- Its normalisation and clipping differences change the result by ≤ 0.05 dB.
- ZFTurbo checkpoint: loads with 0 missing keys only with bs-roformer 0.3.10, after rebuilding the mask estimators with `mlp_expansion_factor=2`.
- bs-roformer 1.x gives 444 missing keys, i.e. a mostly random model.

**Projection.** Changes the result by ±0.2 dB at most. Max |mix − Σ stems| goes from 0.09–0.43 raw to ≤ 2.4e-7.

## 2. AudioSep baseline (mono-scored)

| | vocals | drums | bass |
|---|---|---|---|
| htdemucs_ft | 8.84 / 8.67 | 10.21 / 10.28 | 9.20 / 9.35 |
| chain 317 → ft | 11.95 / 12.52 | 10.59 / 11.13 | 9.33 / 9.53 |
| ZFTurbo fp16 | 10.95 / 11.59 | 11.37 / 11.64 | 7.80 / 7.63 |
| AudioSep "vocals" / "singing" | 1.38 / 2.15 ; −1.10 / −0.70 | | |
| AudioSep "drums" | | −0.45 / 0.42 | |
| AudioSep "bass" / "bass guitar" | | | −0.49 / 0.04 ; −0.80 / 0.07 |

AudioSep is the wrong tool for songs, so the router sends music stems to the dedicated splitter.

## 3. Router classifier

**Labelled set.**
- Music: 50 MUSDB mixes plus the concatenated song.
- Speech: 14 `say` clips and 73 LibriSpeech clips.
- Sound effects: 100 ESC-50 clips.
- Hard case: 20 speech-over-music clips (synthetic, speech at −10 / −20 dB over MUSDB instrumentals).

| Tagger | Rule | Threshold (no-error range) | Accuracy | Speech over music → music | Cold load | Warm per file | VRAM |
|---|---|---|---|---|---|---|---|
| **PANNs Cnn14** | P(Music) | 0.72 (0.59–0.84) | 100% | 20/20 | 8.1 s | 0.12 s | 1.08 GB |
| PANNs | Music − Speech | 0.60 | 99.2% | 0/20 | | | |
| AST 10-10-0.4593 | P(Music) | 0.40 (0.38–0.42) | 100% (97.1% at 0.5) | 20/20 | 21.9 s | 0.17 s | 0.62 GB |
| AST | Music − Speech | 0.21 | 99.6% | 0/20 | | | |

Caveat: the set is small and the thresholds were tuned on it. No solo, classical or ambient music was tested.

## 4. Full song (230.7 s; 34 joined MUSDB previews, since no free full-length track was available)

| Model | Load | Run 1 / run 2 | RTF | VRAM |
|---|---|---|---|---|
| htdemucs | 3.5 s | 11.0 / 10.1 s | 0.044 | 1.87 GB |
| htdemucs_ft | 5.7 s | 37.5 / 32.2 s | 0.140 | 2.32 GB |
| htdemucs_6s | 2.1 s | 10.5 / 9.3 s | 0.040 | 2.38 GB |
| chain 317 → ft (GPU shared) | 12.5 s | 257 / 394 s | 1.7 | 3.1–3.8 GB |

**Live service (end to end over HTTP, GPU shared):**
- htdemucs: 21 s wall, 14 s model.
- chain 317 (overlap 4) → htdemucs: 208 s.

## 5. Dependencies

torch==2.8.0+cu128, torchaudio==2.8.0+cu128, demucs==4.0.1, julius==0.2.8, audio-separator[gpu]==0.48.0, onnxruntime-gpu==1.32.0, audioread==3.1.0, bs-roformer==0.3.10, beartype==0.18.5, panns-inference==0.1.1, transformers==5.23.0, lightning==2.6.6, wget==3.2, numpy==2.5.4.

`audioread` (audio-separator) and `wget` (AudioSep) are undeclared dependencies of those packages and must be installed explicitly.

## Not measured

- Full-song times for ZFTurbo, SW, chain → 6s, and the fast chain.
- Loading everything in one process.
- museval with its BSSEval filter.
- Licences of the ep_317 and Kim weights.
