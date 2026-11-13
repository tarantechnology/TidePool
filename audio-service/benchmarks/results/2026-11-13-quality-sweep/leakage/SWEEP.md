# Separation settings sweep: `leakage`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-small` on `mps`
- **Generated**: 2026-11-13T19:06:15+00:00; baseline (Δ reference) is the first row, `mask-thr0-gate0`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mask-thr0-gate0` | 15.83 | -16.46 | 11.99 | -13.94 | 18.76 | -35.26 | 11.86 | -16.10 | 14.61 | -- | -20.44 | -- | 1.564 | 10760 MB |
| `mask-thr0-gate30` | 15.82 (-0.00) | -16.46 (-0.00) | 11.99 (-0.00) | -13.94 (-0.00) | 18.76 (+0.00) | -35.26 (+0.00) | 11.86 (-0.00) | -16.10 (-0.00) | 14.61 | -0.00 | -20.44 | -0.00 | 1.595 | 10760 MB |
| `mask-thr0.3-gate0` | 14.49 (-1.34) | -17.55 (-1.09) | 11.02 (-0.97) | -15.15 (-1.21) | 16.77 (-1.99) | -36.02 (-0.76) | 10.74 (-1.12) | -17.87 (-1.76) | 13.25 | -1.36 | -21.65 | -1.21 | 1.718 | 10760 MB |
| `mask-thr0.3-gate30` | 14.48 (-1.34) | -17.55 (-1.09) | 11.02 (-0.97) | -15.15 (-1.21) | 16.77 (-1.99) | -36.02 (-0.76) | 10.74 (-1.12) | -17.87 (-1.77) | 13.25 | -1.36 | -21.65 | -1.21 | 1.885 | 10760 MB |
| `mask-thr0.5-gate0` | 13.40 (-2.42) | -18.20 (-1.74) | 10.12 (-1.87) | -15.81 (-1.87) | 15.82 (-2.95) | -36.40 (-1.13) | 9.86 (-2.00) | -18.75 (-2.65) | 12.30 | -2.31 | -22.29 | -1.85 | 1.665 | 10760 MB |
| `mask-thr0.5-gate30` | 13.40 (-2.43) | -18.20 (-1.74) | 10.12 (-1.87) | -15.81 (-1.87) | 15.82 (-2.95) | -36.40 (-1.13) | 9.86 (-2.00) | -18.76 (-2.65) | 12.30 | -2.31 | -22.29 | -1.85 | 1.532 | 10760 MB |
| `mask-p4-thr0.3` | 14.81 (-1.02) | -17.10 (-0.65) | 11.31 (-0.68) | -14.69 (-0.75) | 17.11 (-1.66) | -35.59 (-0.32) | 11.05 (-0.81) | -17.25 (-1.15) | 13.57 | -1.04 | -21.16 | -0.72 | 1.549 | 10760 MB |

## Configs

- **`mask-thr0-gate0`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.0, refine_gate_db=0.0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.0, "refine_gate_db": 0.0}`; full report `mask-thr0-gate0/REPORT.md`
- **`mask-thr0-gate30`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.0, refine_gate_db=30.0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.0, "refine_gate_db": 30.0}`; full report `mask-thr0-gate30/REPORT.md`
- **`mask-thr0.3-gate0`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.3, refine_gate_db=0.0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.3, "refine_gate_db": 0.0}`; full report `mask-thr0.3-gate0/REPORT.md`
- **`mask-thr0.3-gate30`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.3, refine_gate_db=30.0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.3, "refine_gate_db": 30.0}`; full report `mask-thr0.3-gate30/REPORT.md`
- **`mask-thr0.5-gate0`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_gate_db=0.0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_gate_db": 0.0}`; full report `mask-thr0.5-gate0/REPORT.md`
- **`mask-thr0.5-gate30`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_gate_db=30.0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_gate_db": 30.0}`; full report `mask-thr0.5-gate30/REPORT.md`
- **`mask-p4-thr0.3`** -- passes=1, seed=1234, refine=mask, refine_p=4, refine_nfft=2048, refine_threshold=0.3; request options `{"seed": 1234, "refine": "mask", "refine_p": 4, "refine_n_fft": 2048, "refine_threshold": 0.3}`; full report `mask-p4-thr0.3/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep leakage --out benchmarks/out/sweep-leakage
```

