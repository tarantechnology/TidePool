# Separation settings sweep: `leakage-v2`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-large` on `cuda`
- **Generated**: 2026-11-13T23:00:48+00:00; baseline (Δ reference) is the first row, `mask`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint). **Resid-leak: lower is better** (dB of the target sound still inside the residual stem; the 'muted sound still faintly there' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `speech_plus_siren` resid-leak | `two_speakers` SI-SDRi | `two_speakers` leak | `two_speakers` resid-leak | `music_mix` SI-SDRi | `music_mix` leak | `music_mix` resid-leak | `competing_voices` SI-SDRi | `competing_voices` leak | `competing_voices` resid-leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | Mean resid-leak | Δ | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mask` | -22.38 | 19.29 | 8.49 | 15.30 | -17.92 | -14.75 | 19.60 | -35.49 | -19.65 | -2.66 | -17.83 | -1.44 | 2.47 | -- | -12.99 | -- | -6.84 | -- | 0.488 | 29780 MB |
| `thr0.5` | -30.87 (-8.50) | 21.76 (+2.47) | 6.12 (-2.37) | 13.00 (-2.30) | -20.24 (-2.32) | -12.22 (+2.53) | 16.71 (-2.89) | -36.40 (-0.90) | -16.73 (+2.92) | -3.36 (-0.70) | -19.74 (-1.91) | -1.22 (+0.21) | -1.13 | -3.60 | -13.66 | -0.67 | -6.01 | +0.82 | 0.489 | 29780 MB |
| `rthr0.5` | -18.36 (+4.02) | 16.97 (-2.32) | 11.00 (+2.51) | 15.56 (+0.26) | -15.62 (+2.30) | -17.32 (-2.57) | 22.59 (+3.00) | -34.16 (+1.34) | -22.91 (-3.26) | -2.42 (+0.24) | -16.09 (+1.74) | -1.53 (-0.09) | 4.34 | +1.88 | -12.23 | +0.76 | -7.69 | -0.85 | 0.492 | 29780 MB |
| `both0.5` | -27.21 (-4.83) | 19.52 (+0.23) | 7.38 (-1.12) | 14.36 (-0.94) | -19.02 (-1.10) | -13.45 (+1.30) | 17.65 (-1.95) | -35.64 (-0.15) | -17.75 (+1.90) | -3.14 (-0.47) | -18.53 (-0.70) | -1.30 (+0.13) | 0.42 | -2.05 | -13.42 | -0.43 | -6.28 | +0.55 | 0.494 | 29780 MB |
| `both0.7` | -29.53 (-7.15) | 20.33 (+1.05) | 6.40 (-2.09) | 13.37 (-1.93) | -19.60 (-1.68) | -12.57 (+2.18) | 16.75 (-2.85) | -35.88 (-0.39) | -16.85 (+2.79) | -3.48 (-0.82) | -19.17 (-1.33) | -1.19 (+0.24) | -0.72 | -3.19 | -13.58 | -0.59 | -6.05 | +0.78 | 0.494 | 29780 MB |
| `p4-both0.5` | -26.55 (-4.17) | 19.24 (-0.04) | 7.70 (-0.80) | 14.63 (-0.67) | -18.58 (-0.66) | -13.82 (+0.93) | 17.99 (-1.61) | -35.46 (+0.04) | -18.10 (+1.55) | -2.98 (-0.32) | -18.22 (-0.39) | -1.36 (+0.08) | 0.77 | -1.69 | -13.25 | -0.27 | -6.40 | +0.44 | 0.495 | 29780 MB |
| `passes2` | -0.24 (+22.14) | 6.66 (-12.63) | -6.30 (-14.79) | -0.17 (-15.48) | -0.44 (+17.48) | 1.38 (+16.13) | 0.86 (-18.74) | -1.97 (+33.53) | -11.60 (+8.05) | -2.66 (+0.00) | -17.83 (+0.00) | -1.44 (+0.00) | -0.55 | -3.02 | -3.39 | +9.59 | -4.49 | +2.35 | 0.983 | 29780 MB |
| `passes2-both0.5` | -0.16 (+22.21) | 6.54 (-12.74) | -6.52 (-15.01) | -0.24 (-15.54) | -0.44 (+17.48) | 1.44 (+16.19) | 0.52 (-19.08) | -1.67 (+33.82) | -9.62 (+10.03) | -3.14 (-0.47) | -18.53 (-0.70) | -1.30 (+0.13) | -0.75 | -3.22 | -3.52 | +9.47 | -4.00 | +2.84 | 1.003 | 29780 MB |
| `k3auto-both0.5` | -32.22 (-9.84) | 21.40 (+2.11) | 4.84 (-3.65) | 11.16 (-4.15) | -21.44 (-3.52) | -10.59 (+4.16) | 16.58 (-3.01) | -35.87 (-0.38) | -16.66 (+2.99) | -66.06 (-63.40) | 1.01 (+18.84) | 0.51 (+1.95) | -17.63 | -20.10 | -8.73 | +4.26 | -5.47 | +1.36 | 1.277 | 29780 MB |
| `k3auto-passes2-both0.5` | -0.03 (+22.34) | 6.40 (-12.89) | -8.71 (-17.20) | 11.92 (-3.38) | -19.52 (-1.61) | -11.33 (+3.42) | 16.58 (-3.02) | -35.61 (-0.12) | -16.67 (+2.98) | -62.30 (-59.63) | 0.87 (+18.70) | 0.51 (+1.95) | -8.46 | -10.92 | -11.97 | +1.02 | -9.05 | -2.21 | 2.527 | 29780 MB |

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
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode synthetic --seed 1234 --sweep leakage-v2 --out /workspace/bench/large-leakage-v2
```

