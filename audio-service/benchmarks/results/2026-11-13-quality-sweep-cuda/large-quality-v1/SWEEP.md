# Separation settings sweep: `quality-v1`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-large` on `cuda`
- **Generated**: 2026-11-13T22:49:06+00:00; baseline (Δ reference) is the first row, `off`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `off` | -32.82 | 17.86 | 11.98 | -16.05 | 13.85 | -30.71 | -2.82 | -15.79 | -2.45 | -- | -11.17 | -- | 0.482 | 29781 MB |
| `mask` | -22.38 (+10.44) | 19.29 (+1.43) | 15.30 (+3.32) | -17.92 (-1.87) | 19.60 (+5.74) | -35.49 (-4.79) | -2.66 (+0.16) | -17.83 (-2.04) | 2.47 | +4.92 | -12.99 | -1.82 | 0.493 | 29781 MB |
| `mask-k2-auto` | -0.48 (+32.34) | 6.65 (-11.21) | 15.38 (+3.40) | -18.02 (-1.97) | 9.47 (-4.38) | -35.01 (-4.30) | -55.36 (-52.54) | 0.43 (+16.22) | -7.75 | -5.29 | -11.49 | -0.32 | 0.862 | 29781 MB |
| `mask-k3-auto` | -10.49 (+22.33) | 15.33 (-2.52) | 14.29 (+2.30) | -19.30 (-3.25) | 15.56 (+1.70) | -35.49 (-4.78) | -55.36 (-52.54) | 0.43 (+16.22) | -9.00 | -6.55 | -9.76 | +1.41 | 1.267 | 29781 MB |
| `quality-profile` | -4.29 (+28.52) | 4.70 (-13.16) | 13.18 (+1.20) | -20.28 (-4.23) | 2.84 (-11.01) | -35.49 (-4.79) | -66.96 (-64.14) | 0.98 (+16.77) | -13.81 | -11.36 | -12.52 | -1.35 | 0.866 | 29781 MB |

## Configs

- **`off`** -- passes=1, seed=1234, window_candidates=1, refine=off; request options `{"seed": 1234, "candidates": 1, "refine": "off"}`; full report `off/REPORT.md`
- **`mask`** -- passes=1, seed=1234, window_candidates=1, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "candidates": 1, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask/REPORT.md`
- **`mask-k2-auto`** -- passes=1, seed=1234, window_candidates=2, select=auto, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "candidates": 2, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask-k2-auto/REPORT.md`
- **`mask-k3-auto`** -- passes=1, seed=1234, window_candidates=3, select=auto, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "candidates": 3, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask-k3-auto/REPORT.md`
- **`quality-profile`** -- passes=1, seed=1234, window_candidates=2, select=auto, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5; request options `{"seed": 1234, "candidates": 2, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5}`; full report `quality-profile/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode synthetic --seed 1234 --sweep quality-v1 --out /workspace/bench/large-quality-v1
```

