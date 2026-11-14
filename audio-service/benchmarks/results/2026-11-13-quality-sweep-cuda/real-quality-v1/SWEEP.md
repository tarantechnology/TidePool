# Separation settings sweep: `quality-v1`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-large` on `cuda`
- **Generated**: 2026-11-14T00:19:46+00:00; baseline (Δ reference) is the first row, `off`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint). **Resid-leak: lower is better** (dB of the target sound still inside the residual stem; the 'muted sound still faintly there' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `speech_plus_siren` resid-leak | `two_speakers` SI-SDRi | `two_speakers` leak | `two_speakers` resid-leak | `music_mix` SI-SDRi | `music_mix` leak | `music_mix` resid-leak | `competing_voices` SI-SDRi | `competing_voices` leak | `competing_voices` resid-leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | Mean resid-leak | Δ | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `off` | 12.72 | -13.05 | -13.95 | -56.56 | 12.07 | 9.45 | 13.85 | -30.71 | -13.94 | -8.78 | 3.12 | 9.66 | -9.69 | -- | -7.14 | -- | -2.19 | -- | 0.479 | 29772 MB |
| `mask` | 17.17 (+4.45) | -15.41 (-2.36) | -18.31 (-4.36) | -26.01 (+30.54) | 13.35 (+1.29) | 11.89 (+2.44) | 19.60 (+5.74) | -35.49 (-4.79) | -19.65 (-5.71) | -8.18 (+0.60) | 3.12 (-0.00) | 12.79 (+3.13) | 0.64 | +10.33 | -8.61 | -1.47 | -3.32 | -1.13 | 0.490 | 29772 MB |
| `mask-k2-auto` | 17.49 (+4.78) | -16.12 (-3.07) | -18.60 (-4.65) | -5.58 (+50.98) | 4.48 (-7.58) | 9.35 (-0.10) | 9.47 (-4.38) | -35.01 (-4.30) | -9.44 (+4.50) | -27.37 (-18.59) | 14.12 (+11.00) | 13.36 (+3.70) | -1.49 | +8.20 | -8.13 | -0.99 | -1.33 | +0.86 | 0.864 | 29772 MB |
| `mask-k3-auto` | 16.56 (+3.84) | -16.93 (-3.88) | -14.27 (-0.32) | -0.17 (+56.38) | -0.48 (-12.54) | 0.22 (-9.24) | 15.56 (+1.70) | -35.49 (-4.78) | -14.97 (-1.03) | 2.11 (+10.89) | -2.88 (-6.00) | -10.97 (-20.63) | 8.51 | +18.20 | -13.94 | -6.80 | -10.00 | -7.81 | 1.285 | 29772 MB |
| `quality-profile` | 11.88 (-0.84) | -17.85 (-4.80) | -10.13 (+3.82) | -0.46 (+56.10) | -0.50 (-12.57) | 0.48 (-8.97) | 10.18 (-3.67) | -35.85 (-5.15) | -10.04 (+3.89) | 1.84 (+10.61) | -2.86 (-5.98) | -9.03 (-18.69) | 5.86 | +15.55 | -14.27 | -7.13 | -7.18 | -4.99 | 1.263 | 29772 MB |
| `mask-whole-file` | 17.17 (+4.45) | -15.41 (-2.36) | -18.31 (-4.36) | -26.01 (+30.54) | 13.35 (+1.29) | 11.89 (+2.44) | 19.60 (+5.74) | -35.49 (-4.79) | -19.65 (-5.71) | -0.24 (+8.54) | -0.45 (-3.58) | 1.11 (-8.55) | 2.63 | +12.32 | -9.50 | -2.36 | -6.24 | -4.05 | 0.396 | 29772 MB |
| `mask-chunk20` | 17.17 (+4.45) | -15.41 (-2.36) | -18.31 (-4.36) | -26.01 (+30.54) | 13.35 (+1.29) | 11.89 (+2.44) | 19.60 (+5.74) | -35.49 (-4.79) | -19.65 (-5.71) | -27.40 (-18.62) | 14.20 (+11.08) | 13.69 (+4.03) | -4.16 | +5.53 | -5.84 | +1.30 | -3.09 | -0.90 | 0.400 | 29772 MB |

## Configs

- **`off`** -- passes=1, seed=1234, window_candidates=1, refine=off; request options `{"seed": 1234, "candidates": 1, "refine": "off"}`; full report `off/REPORT.md`
- **`mask`** -- passes=1, seed=1234, window_candidates=1, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "candidates": 1, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask/REPORT.md`
- **`mask-k2-auto`** -- passes=1, seed=1234, window_candidates=2, select=auto, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "candidates": 2, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask-k2-auto/REPORT.md`
- **`mask-k3-auto`** -- passes=1, seed=1234, window_candidates=3, select=auto, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "candidates": 3, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask-k3-auto/REPORT.md`
- **`quality-profile`** -- passes=1, seed=1234, window_candidates=3, select=auto, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5; request options `{"seed": 1234, "candidates": 3, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5}`; full report `quality-profile/REPORT.md`
- **`mask-whole-file`** -- passes=1, seed=1234, window_candidates=1, refine=mask, refine_p=2, refine_nfft=2048, chunk_seconds=0.0; request options `{"seed": 1234, "candidates": 1, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "chunk_seconds": 0.0}`; full report `mask-whole-file/REPORT.md`
- **`mask-chunk20`** -- passes=1, seed=1234, window_candidates=1, refine=mask, refine_p=2, refine_nfft=2048, chunk_seconds=20.0; request options `{"seed": 1234, "candidates": 1, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "chunk_seconds": 20.0}`; full report `mask-chunk20/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep quality-v1 --out /workspace/bench/real-quality-v1
```

