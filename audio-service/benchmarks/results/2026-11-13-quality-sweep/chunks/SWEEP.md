# Separation settings sweep: `chunks`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-small` on `mps`
- **Generated**: 2026-11-13T18:46:17+00:00; baseline (Δ reference) is the first row, `chunk10`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `chunk10` | 12.16 | -12.79 | 10.23 | -12.21 | 12.90 | -28.81 | 10.11 | -13.60 | 11.35 | -- | -16.85 | -- | 1.370 | 10763 MB |
| `chunk15` | 12.16 (+0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (+0.00) | 12.90 (+0.00) | -28.81 (+0.00) | 9.59 (-0.52) | -13.07 (+0.52) | 11.22 | -0.13 | -16.72 | +0.13 | 1.042 | 10763 MB |
| `chunk20` | 12.16 (+0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (+0.00) | 12.90 (+0.00) | -28.81 (+0.00) | 0.07 (-10.04) | -4.29 (+9.30) | 8.84 | -2.51 | -14.53 | +2.33 | 1.083 | 10763 MB |
| `whole-file` | 12.16 (+0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (+0.00) | 12.90 (+0.00) | -28.81 (+0.00) | 10.38 (+0.27) | -13.71 (-0.12) | 11.42 | +0.07 | -16.88 | -0.03 | 1.046 | 10763 MB |

## Configs

- **`chunk10`** -- passes=1, seed=1234, window_candidates=1, chunk_seconds=10.0; request options `{"seed": 1234, "candidates": 1, "chunk_seconds": 10.0}`; full report `chunk10/REPORT.md`
- **`chunk15`** -- passes=1, seed=1234, window_candidates=1, chunk_seconds=15.0; request options `{"seed": 1234, "candidates": 1, "chunk_seconds": 15.0}`; full report `chunk15/REPORT.md`
- **`chunk20`** -- passes=1, seed=1234, window_candidates=1, chunk_seconds=20.0; request options `{"seed": 1234, "candidates": 1, "chunk_seconds": 20.0}`; full report `chunk20/REPORT.md`
- **`whole-file`** -- passes=1, seed=1234, window_candidates=1, chunk_seconds=0.0; request options `{"seed": 1234, "candidates": 1, "chunk_seconds": 0.0}`; full report `whole-file/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep chunks --out benchmarks/out/sweep-chunks
```

