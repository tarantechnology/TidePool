# Separation settings sweep: `ode`

- **Backend**: `sam` (engine mode), model `facebook/sam-audio-small` on `mps`
- **Generated**: 2026-11-13T18:41:06+00:00; baseline (Δ reference) is the first row, `step-0.0625`
- Cells read `value (Δ vs baseline)`. **SI-SDRi: higher is better. Leakage: lower is better** (dB of non-target sound inside the stem; the 'faint background still audible' complaint).
- Peak RSS is the process high-water mark up to that config (it never goes down).

| Config | `speech_plus_siren` SI-SDRi | `speech_plus_siren` leak | `two_speakers` SI-SDRi | `two_speakers` leak | `music_mix` SI-SDRi | `music_mix` leak | `competing_voices` SI-SDRi | `competing_voices` leak | Mean SI-SDRi | Δ mean SI-SDRi | Mean leak | Δ mean leak | RTF | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `step-0.0625` | 12.16 | -12.79 | 10.23 | -12.21 | 12.90 | -28.81 | 10.11 | -13.60 | 11.35 | -- | -16.85 | -- | 1.510 | 10752 MB |
| `step-0.03125` | 12.15 (-0.02) | -12.77 (+0.02) | 10.21 (-0.01) | -12.21 (+0.00) | 12.87 (-0.03) | -28.77 (+0.04) | 10.10 (-0.01) | -13.59 (+0.00) | 11.33 | -0.02 | -16.84 | +0.02 | 1.946 | 10752 MB |
| `step-0.015625` | 12.14 (-0.02) | -12.77 (+0.02) | 10.21 (-0.01) | -12.21 (+0.00) | 12.86 (-0.04) | -28.77 (+0.04) | 10.09 (-0.02) | -13.59 (+0.01) | 11.33 | -0.02 | -16.83 | +0.02 | 2.981 | 10752 MB |

## Configs

- **`step-0.0625`** -- passes=1, seed=1234, ode_step=0.0625; request options `{"seed": 1234}`; full report `step-0.0625/REPORT.md`
- **`step-0.03125`** -- passes=1, seed=1234, ode_step=0.03125; request options `{"seed": 1234}`; full report `step-0.03125/REPORT.md`
- **`step-0.015625`** -- passes=1, seed=1234, ode_step=0.015625; request options `{"seed": 1234}`; full report `step-0.015625/REPORT.md`

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep ode --out benchmarks/out/sweep-ode
```

