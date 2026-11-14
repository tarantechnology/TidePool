# Separation settings sweep: `gate`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-large` on `cuda`
- **Generated**: 2026-11-13T22:43:32+00:00; baseline (Δ reference) is the first row, `thr0.5-gate0`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `thr0.5-gate0` | -30.87 | 21.76 | 13.00 | -20.24 | 16.71 | -36.40 | -3.36 | -19.74 | -1.13 | -- | -13.66 | -- | 0.487 | 29796 MB |
| `thr0.5-gate10` | -30.87 (-0.00) | 21.76 (-0.00) | 12.92 (-0.08) | -20.27 (-0.03) | 15.40 (-1.31) | -35.78 (+0.62) | -3.51 (-0.15) | -19.73 (+0.01) | -1.52 | -0.39 | -13.51 | +0.15 | 0.491 | 29796 MB |
| `thr0.5-gate15` | -30.87 (-0.00) | 21.76 (-0.00) | 12.99 (-0.01) | -20.24 (-0.00) | 16.67 (-0.04) | -36.44 (-0.04) | -3.40 (-0.04) | -19.74 (+0.00) | -1.15 | -0.02 | -13.67 | -0.01 | 0.494 | 29796 MB |
| `thr0.5-gate20` | -30.87 (-0.00) | 21.76 (-0.00) | 13.00 (-0.00) | -20.24 (-0.00) | 16.71 (+0.00) | -36.40 (+0.00) | -3.37 (-0.01) | -19.74 (+0.00) | -1.13 | -0.00 | -13.66 | +0.00 | 0.496 | 29796 MB |

## Configs

- **`thr0.5-gate0`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_gate_db=0; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_gate_db": 0}`; full report `thr0.5-gate0/REPORT.md`
- **`thr0.5-gate10`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_gate_db=10; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_gate_db": 10}`; full report `thr0.5-gate10/REPORT.md`
- **`thr0.5-gate15`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_gate_db=15; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_gate_db": 15}`; full report `thr0.5-gate15/REPORT.md`
- **`thr0.5-gate20`** -- passes=1, seed=1234, refine=mask, refine_p=2, refine_nfft=2048, refine_threshold=0.5, refine_gate_db=20; request options `{"seed": 1234, "refine": "mask", "refine_p": 2, "refine_n_fft": 2048, "refine_threshold": 0.5, "refine_gate_db": 20}`; full report `thr0.5-gate20/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode synthetic --seed 1234 --sweep gate --out /workspace/bench/large-gate
```

