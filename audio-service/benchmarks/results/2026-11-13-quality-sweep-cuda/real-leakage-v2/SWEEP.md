# Separation settings sweep: `leakage-v2`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-large` on `cuda`
- **Generated**: 2026-11-14T00:30:49+00:00; baseline (Δ reference) is the first row, `mask`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint). **Resid-leak: lower is better** (dB of the target sound still inside the residual stem; the 'muted sound still faintly there' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `speech_plus_siren` resid-leak | `two_speakers` SI-SDRi | `two_speakers` leak | `two_speakers` resid-leak | `music_mix` SI-SDRi | `music_mix` leak | `music_mix` resid-leak | `competing_voices` SI-SDRi | `competing_voices` leak | `competing_voices` resid-leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | Mean resid-leak | Δ | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mask` | 17.17 | -15.41 | -18.31 | -26.01 | 13.35 | 11.89 | 19.60 | -35.49 | -19.65 | -8.18 | 3.12 | 12.79 | 0.64 | -- | -8.61 | -- | -3.32 | -- | 0.487 | 29763 MB |
| `thr0.5` | 14.98 (-2.19) | -17.15 (-1.74) | -15.42 (+2.89) | -35.03 (-9.02) | 15.01 (+1.65) | 10.06 (-1.83) | 16.71 (-2.89) | -36.40 (-0.90) | -16.73 (+2.92) | -8.80 (-0.62) | 3.25 (+0.13) | 10.77 (-2.02) | -3.04 | -3.68 | -8.82 | -0.22 | -2.83 | +0.49 | 0.490 | 29763 MB |
| `rthr0.5` | 17.87 (+0.70) | -13.71 (+1.70) | -21.13 (-2.82) | -22.08 (+3.93) | 11.62 (-1.73) | 13.59 (+1.70) | 22.59 (+3.00) | -34.16 (+1.34) | -22.91 (-3.26) | -7.70 (+0.48) | 2.89 (-0.23) | 14.83 (+2.04) | 2.67 | +2.03 | -8.34 | +0.27 | -3.90 | -0.59 | 0.493 | 29763 MB |
| `both0.5` | 16.05 (-1.11) | -15.94 (-0.53) | -17.09 (+1.22) | -31.74 (-5.73) | 13.91 (+0.56) | 10.98 (-0.91) | 17.65 (-1.95) | -35.64 (-0.15) | -17.75 (+1.90) | -8.52 (-0.34) | 3.07 (-0.05) | 11.67 (-1.13) | -1.64 | -2.28 | -8.65 | -0.05 | -3.05 | +0.27 | 0.494 | 29763 MB |
| `both0.7` | 14.86 (-2.31) | -16.99 (-1.58) | -15.64 (+2.66) | -35.73 (-9.72) | 14.62 (+1.26) | 10.16 (-1.73) | 16.75 (-2.85) | -35.88 (-0.39) | -16.85 (+2.79) | -8.70 (-0.53) | 3.15 (+0.03) | 10.94 (-1.86) | -3.21 | -3.85 | -8.78 | -0.17 | -2.85 | +0.47 | 0.495 | 29763 MB |
| `p4-both0.5` | 16.29 (-0.88) | -15.41 (+0.00) | -17.58 (+0.73) | -30.16 (-4.15) | 13.47 (+0.12) | 11.29 (-0.60) | 17.99 (-1.61) | -35.46 (+0.04) | -18.10 (+1.55) | -8.47 (-0.29) | 3.05 (-0.07) | 11.96 (-0.84) | -1.09 | -1.73 | -8.59 | +0.02 | -3.11 | +0.21 | 0.512 | 29763 MB |
| `passes2` | 17.39 (+0.23) | -14.89 (+0.52) | -19.13 (-0.83) | -26.01 (+0.00) | 13.35 (+0.00) | 11.89 (+0.00) | 19.60 (+0.00) | -35.49 (+0.00) | -19.65 (+0.00) | -0.21 (+7.96) | -0.46 (-3.58) | 1.04 (-11.76) | 2.69 | +2.05 | -9.37 | -0.77 | -6.46 | -3.15 | 0.987 | 29763 MB |
| `passes2-both0.5` | 16.42 (-0.75) | -15.00 (+0.42) | -17.85 (+0.45) | -31.74 (-5.73) | 13.91 (+0.56) | 10.98 (-0.91) | 17.65 (-1.95) | -35.64 (-0.15) | -17.75 (+1.90) | -0.26 (+7.91) | -0.48 (-3.60) | 0.77 (-12.02) | 0.51 | -0.13 | -9.30 | -0.69 | -5.96 | -2.65 | 1.013 | 29763 MB |
| `k3auto-both0.5` | 12.56 (-4.61) | -18.64 (-3.23) | -12.96 (+5.35) | -0.47 (+25.54) | -0.38 (-13.73) | 0.84 (-11.05) | 16.58 (-3.01) | -35.87 (-0.38) | -16.66 (+2.99) | 2.05 (+10.22) | -2.94 (-6.06) | -9.98 (-22.77) | 7.68 | +7.03 | -14.46 | -5.85 | -9.69 | -6.37 | 1.275 | 29763 MB |
| `k3auto-passes2-both0.5` | 16.18 (-0.99) | -17.14 (-1.72) | -16.92 (+1.38) | -0.18 (+25.83) | -0.48 (-13.84) | -0.62 (-12.51) | 16.58 (-3.01) | -35.87 (-0.38) | -16.66 (+2.99) | -0.27 (+7.90) | -0.47 (-3.59) | 0.90 (-11.89) | 8.08 | +7.43 | -13.49 | -4.88 | -8.33 | -5.01 | 2.562 | 29763 MB |

## Configs

- **`mask`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask/REPORT.md`
- **`thr0.5`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5}`; full report `thr0.5/REPORT.md`
- **`rthr0.5`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_residual_threshold=0.5; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_residual_threshold": 0.5}`; full report `rthr0.5/REPORT.md`
- **`both0.5`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_residual_threshold=0.5; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_residual_threshold": 0.5}`; full report `both0.5/REPORT.md`
- **`both0.7`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.7, refine_residual_threshold=0.7; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.7, "refine_residual_threshold": 0.7}`; full report `both0.7/REPORT.md`
- **`p4-both0.5`** -- passes=1, seed=1234, refine=mask, refine_p=4, refine_nfft=2048, refine_threshold=0.5, refine_residual_threshold=0.5; request options `{"seed": 1234, "refine": "mask", "refine_p": 4, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_residual_threshold": 0.5}`; full report `p4-both0.5/REPORT.md`
- **`passes2`** -- passes=2, seed=1234, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `passes2/REPORT.md`
- **`passes2-both0.5`** -- passes=2, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_residual_threshold=0.5; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_residual_threshold": 0.5}`; full report `passes2-both0.5/REPORT.md`
- **`k3auto-both0.5`** -- passes=1, seed=1234, window_candidates=3, select=auto, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_residual_threshold=0.5; request options `{"seed": 1234, "candidates": 3, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_residual_threshold": 0.5}`; full report `k3auto-both0.5/REPORT.md`
- **`k3auto-passes2-both0.5`** -- passes=2, seed=1234, window_candidates=3, select=auto, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_residual_threshold=0.5; request options `{"seed": 1234, "candidates": 3, "candidate_select": "auto", "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_residual_threshold": 0.5}`; full report `k3auto-passes2-both0.5/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep leakage-v2 --out /workspace/bench/real-leakage-v2
```

