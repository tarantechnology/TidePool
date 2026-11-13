# Separation settings sweep: `refine`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-small` on `mps`
- **Generated**: 2026-11-13T18:55:57+00:00; baseline (Δ reference) is the first row, `refine-off`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refine-off` | 12.16 | -12.79 | 10.23 | -12.21 | 12.90 | -28.81 | 10.11 | -13.60 | 11.35 | -- | -16.85 | -- | 1.371 | 10781 MB |
| `mask-p1-n1024` | 15.78 (+3.62) | -14.40 (-1.61) | 7.93 (-2.29) | -10.00 (+2.22) | 19.18 (+6.28) | -33.74 (-4.93) | 9.12 (-0.99) | -12.29 (+1.31) | 13.00 | +1.65 | -17.61 | -0.75 | 1.365 | 10781 MB |
| `mask-p1-n2048` | 16.28 (+4.11) | -15.34 (-2.55) | 11.41 (+1.18) | -12.81 (-0.60) | 19.20 (+6.30) | -34.44 (-5.63) | 11.58 (+1.47) | -14.52 (-0.93) | 14.62 | +3.27 | -19.28 | -2.43 | 1.369 | 10781 MB |
| `mask-p1-n4096` | 16.21 (+4.04) | -15.37 (-2.58) | 12.31 (+2.09) | -12.80 (-0.59) | 19.06 (+6.16) | -34.13 (-5.32) | 12.45 (+2.34) | -14.60 (-1.00) | 15.01 | +3.66 | -19.22 | -2.37 | 1.526 | 10781 MB |
| `mask-p2-n1024` | 15.37 (+3.20) | -15.78 (-2.99) | 8.86 (-1.37) | -11.98 (+0.24) | 18.83 (+5.93) | -34.33 (-5.52) | 9.66 (-0.45) | -14.53 (-0.93) | 13.18 | +1.83 | -19.15 | -2.30 | 1.389 | 10781 MB |
| `mask-p2-n2048` | 15.83 (+3.66) | -16.46 (-3.67) | 11.99 (+1.76) | -13.94 (-1.73) | 18.76 (+5.86) | -35.26 (-6.45) | 11.86 (+1.75) | -16.10 (-2.51) | 14.61 | +3.26 | -20.44 | -3.59 | 1.458 | 10781 MB |
| `mask-p2-n4096` | 15.64 (+3.47) | -16.43 (-3.64) | 12.58 (+2.35) | -13.33 (-1.11) | 18.55 (+5.65) | -35.29 (-6.48) | 12.54 (+2.43) | -15.50 (-1.91) | 14.83 | +3.48 | -20.14 | -3.28 | 1.429 | 10781 MB |

## Configs

- **`refine-off`** -- passes=1, seed=1234, refine=off; request options `{"seed": 1234, "refine": "off"}`; full report `refine-off/REPORT.md`
- **`mask-p1-n1024`** -- passes=1, seed=1234, refine=mask, refine_p=1, refine_nfft=1024; request options `{"seed": 1234, "refine": "mask", "refine_p": 1, "refine_n_fft": 1024}`; full report `mask-p1-n1024/REPORT.md`
- **`mask-p1-n2048`** -- passes=1, seed=1234, refine=mask, refine_p=1, refine_nfft=2048; request options `{"seed": 1234, "refine": "mask", "refine_p": 1, "refine_n_fft": 2048}`; full report `mask-p1-n2048/REPORT.md`
- **`mask-p1-n4096`** -- passes=1, seed=1234, refine=mask, refine_p=1, refine_nfft=4096; request options `{"seed": 1234, "refine": "mask", "refine_p": 1, "refine_n_fft": 4096}`; full report `mask-p1-n4096/REPORT.md`
- **`mask-p2-n1024`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=1024; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 1024}`; full report `mask-p2-n1024/REPORT.md`
- **`mask-p2-n2048`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048}`; full report `mask-p2-n2048/REPORT.md`
- **`mask-p2-n4096`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=4096; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 4096}`; full report `mask-p2-n4096/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep refine --out benchmarks/out/sweep-refine
```

