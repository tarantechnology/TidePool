# Separation settings sweep: `candidates`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-small` on `mps`
- **Generated**: 2026-11-13T21:05:06+00:00; baseline (Δ reference) is the first row, `baseline-k1-chunk10`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-k1-chunk10` | 12.16 | -12.79 | 10.23 | -12.21 | 12.90 | -28.81 | 10.11 | -13.60 | 11.35 | -- | -16.85 | -- | 3.263 | 10775 MB |
| `k2-overlap-chunk10` | 12.16 (-0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (-0.00) | 12.90 (+0.00) | -28.81 (-0.00) | 10.11 (+0.00) | -13.60 (-0.00) | 11.35 | +0.00 | -16.85 | +0.00 | 5.599 | 10775 MB |
| `k2-mean-chunk10` | 13.20 (+1.03) | -13.94 (-1.15) | 11.38 (+1.16) | -12.47 (-0.26) | 13.90 (+1.00) | -30.35 (-1.54) | 8.87 (-1.24) | -10.59 (+3.00) | 11.84 | +0.49 | -16.84 | +0.01 | 2.548 | 10775 MB |
| `k3-overlap-chunk10` | 12.16 (-0.00) | -12.79 (+0.00) | 10.74 (+0.51) | -12.14 (+0.07) | 12.90 (+0.00) | -28.81 (-0.00) | 10.58 (+0.47) | -13.82 (-0.22) | 11.59 | +0.25 | -16.89 | -0.04 | 3.481 | 10775 MB |
| `k3-mean-chunk10` | 13.83 (+1.66) | -14.59 (-1.80) | 12.07 (+1.85) | -12.49 (-0.28) | 11.58 (-1.32) | -30.91 (-2.10) | 8.77 (-1.34) | -10.26 (+3.33) | 11.56 | +0.21 | -17.06 | -0.21 | 3.755 | 10775 MB |
| `k2-overlap-chunk15` | 12.16 (-0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (-0.00) | 12.90 (+0.00) | -28.81 (-0.00) | 10.33 (+0.22) | -13.60 (-0.01) | 11.41 | +0.06 | -16.86 | -0.00 | 2.199 | 10775 MB |
| `k1-chunk20` | 12.16 (+0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (+0.00) | 12.90 (+0.00) | -28.81 (+0.00) | 0.07 (-10.04) | -4.29 (+9.30) | 8.84 | -2.51 | -14.53 | +2.33 | 1.167 | 10775 MB |
| `k2-overlap-chunk20` | 12.16 (-0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (-0.00) | 12.90 (+0.00) | -28.81 (-0.00) | 9.94 (-0.17) | -13.64 (-0.04) | 11.31 | -0.04 | -16.86 | -0.01 | 2.157 | 10775 MB |
| `k3-overlap-chunk20` | 12.16 (-0.00) | -12.79 (+0.00) | 10.74 (+0.51) | -12.14 (+0.07) | 12.90 (+0.00) | -28.81 (-0.00) | 9.85 (-0.26) | -13.58 (+0.01) | 11.41 | +0.06 | -16.83 | +0.02 | 3.175 | 10775 MB |
| `k2-overlap-whole` | 12.16 (-0.00) | -12.79 (+0.00) | 10.23 (+0.00) | -12.21 (-0.00) | 12.90 (+0.00) | -28.81 (-0.00) | 10.38 (+0.27) | -13.71 (-0.12) | 11.42 | +0.07 | -16.88 | -0.03 | 2.056 | 10775 MB |

## Configs

- **`baseline-k1-chunk10`** -- passes=1, seed=1234, window_candidates=1, chunk_seconds=10.0; request options `{"seed": 1234, "candidates": 1, "chunk_seconds": 10.0}`; full report `baseline-k1-chunk10/REPORT.md`
- **`k2-overlap-chunk10`** -- passes=1, seed=1234, window_candidates=2, select=overlap, chunk_seconds=10.0; request options `{"seed": 1234, "candidates": 2, "candidate_select": "overlap", "chunk_seconds": 10.0}`; full report `k2-overlap-chunk10/REPORT.md`
- **`k2-mean-chunk10`** -- passes=1, seed=1234, window_candidates=2, select=mean, chunk_seconds=10.0; request options `{"seed": 1234, "candidates": 2, "candidate_select": "mean", "chunk_seconds": 10.0}`; full report `k2-mean-chunk10/REPORT.md`
- **`k3-overlap-chunk10`** -- passes=1, seed=1234, window_candidates=3, select=overlap, chunk_seconds=10.0; request options `{"seed": 1234, "candidates": 3, "candidate_select": "overlap", "chunk_seconds": 10.0}`; full report `k3-overlap-chunk10/REPORT.md`
- **`k3-mean-chunk10`** -- passes=1, seed=1234, window_candidates=3, select=mean, chunk_seconds=10.0; request options `{"seed": 1234, "candidates": 3, "candidate_select": "mean", "chunk_seconds": 10.0}`; full report `k3-mean-chunk10/REPORT.md`
- **`k2-overlap-chunk15`** -- passes=1, seed=1234, window_candidates=2, select=overlap, chunk_seconds=15.0; request options `{"seed": 1234, "candidates": 2, "candidate_select": "overlap", "chunk_seconds": 15.0}`; full report `k2-overlap-chunk15/REPORT.md`
- **`k1-chunk20`** -- passes=1, seed=1234, window_candidates=1, chunk_seconds=20.0; request options `{"seed": 1234, "candidates": 1, "chunk_seconds": 20.0}`; full report `k1-chunk20/REPORT.md`
- **`k2-overlap-chunk20`** -- passes=1, seed=1234, window_candidates=2, select=overlap, chunk_seconds=20.0; request options `{"seed": 1234, "candidates": 2, "candidate_select": "overlap", "chunk_seconds": 20.0}`; full report `k2-overlap-chunk20/REPORT.md`
- **`k3-overlap-chunk20`** -- passes=1, seed=1234, window_candidates=3, select=overlap, chunk_seconds=20.0; request options `{"seed": 1234, "candidates": 3, "candidate_select": "overlap", "chunk_seconds": 20.0}`; full report `k3-overlap-chunk20/REPORT.md`
- **`k2-overlap-whole`** -- passes=1, seed=1234, window_candidates=2, select=overlap, chunk_seconds=0.0; request options `{"seed": 1234, "candidates": 2, "candidate_select": "overlap", "chunk_seconds": 0.0}`; full report `k2-overlap-whole/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep candidates --out benchmarks/out/sweep-candidates
```

