# SoundCanvas audio separation benchmark

- **Backend**: `sam` (engine mode)
- **Mock backend**: **no**
- **Model**: `facebook/sam-audio-small` on `mps` (float32)
- **Generated**: 2026-11-07T05:01:44+00:00 on macOS-26.6.2-arm64-arm-64bit, python 3.12.6, numpy 1.26.4
- **Fixtures**: synthetic, deterministic, 6s quality clips + 15s latency clip, speech source `say`

| Fixture | Prompt | Target | Dur (s) | SI-SDR (dB) | input SI-SDR (dB) | SI-SDRi (dB) | SDR (dB) | Leakage (dB) | Recon err | Latency (s) | RTF (proc/audio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `speech_plus_siren` | `siren` | `siren` | 6.00 | 5.01 | -6.59 | 11.60 | 6.20 | -12.73 | 8.57e-08 | 39.129 | 6.522 |
| `two_speakers` | `man speaking` | `speaker_low` | 6.00 | 10.67 | 0.49 | 10.19 | 11.00 | -12.34 | 4.47e-08 | 24.364 | 4.061 |
| `music_mix` | `guitar` | `guitar` | 6.00 | 13.94 | 1.00 | 12.94 | 13.65 | -28.69 | 4.47e-08 | 45.030 | 7.505 |
| `long_mixture_15s` | `siren` | `siren` | 15.00 | 1.95 | -5.01 | 6.96 | 4.06 | -15.96 | 2.98e-08 | 22.296 | 1.486 |

**Latency reference** (`long_mixture_15s`, 15.0s mono, 1 run(s)): 22.296 s = **1.486 s of processing per second of audio** (0.7x faster than real time).

## What the fixtures are

- **`speech_plus_siren`** -- prompt `siren`, ground-truth target `siren`, 2ch, speech source `say:Daniel`. Cinematic demo case: siren must come out as its own stem, 6 dB under the voice, panned right.
- **`two_speakers`** -- prompt `man speaking`, ground-truth target `speaker_low`, 2ch, speech source `low=say:Daniel high=say:Samantha`. Two talkers at equal level, ~100 Hz vs ~205 Hz f0, panned apart. Hard case: prompts cannot name a person.
- **`music_mix`** -- prompt `guitar`, ground-truth target `guitar`, 2ch, speech source `n/a`. Percussive kit vs sustained harmonic tone; overlapping low band makes band-pass mocks fail here.
- **`long_mixture_15s`** -- prompt `siren`, ground-truth target `siren`, 1ch, speech source `say:Daniel`. Mono 15s mixture for latency and real-time factor.

## Metrics

- **SI-SDR** -- scale-invariant signal-to-distortion ratio of the target stem against ground truth. Higher is better; above ~10 dB the wanted sound clearly dominates.
- **input SI-SDR** -- the same measure on the untouched mixture: the score for doing nothing.
- **SI-SDRi** -- improvement over that baseline. **This is the headline number.** <= 0 dB means the stem is no better than the mixture.
- **SDR** -- scale-dependent variant; punishes level errors, which matter when the stem is dropped into the editor's mix.
- **Leakage** -- energy of non-target sound in the target stem relative to the wanted sound, in dB. Lower is better; 0 dB means half the stem is the wrong sound.
- **Recon err** -- max absolute sample error of `target + residual - mixture`. ~1e-7 is float round-off, ~1e-4 is 16-bit WAV quantisation over HTTP; larger means a pipeline bug.
- **RTF** -- seconds of processing per second of audio (lower is faster; < 1 is faster than real time).

## Reproduce

```sh
cd audio-service && python -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --speech-mode say --out benchmarks/out/sam-small-mps
```

See `benchmarks/README.md` for how to run this against a real SAM Audio service.
