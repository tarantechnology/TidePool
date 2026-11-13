# Audio separation benchmarks (Person 1)

Quality and latency benchmarks for the SoundCanvas audio inference service. Everything is
generated at runtime from fixed seeds, so there is no audio in git and every machine
measures the same signals.

```
benchmarks/
  fixtures.py        synthetic mixtures + ground-truth sources (deterministic)
  run_benchmark.py   runs the fixtures, measures, writes the report (or a settings sweep)
  sweeps.py          named settings sweeps for --sweep
  out/               report.json + REPORT.md         (gitignored)
../fixtures/generated/  WAVs + manifest.json for other teams (gitignored)
```

Metrics live in `sam_audio_service/metrics.py` (pure numpy: no torch, scipy or librosa).

## Run it

```sh
cd audio-service
.venv/bin/python -m benchmarks.run_benchmark                      # mock backend, defaults
.venv/bin/python -m benchmarks.run_benchmark --seconds 10 --repeat 3
.venv/bin/python -m benchmarks.run_benchmark --fixtures music_mix --latency-seconds 0
```

The default run takes well under a minute with the mock backend and writes
`benchmarks/out/report.json` and `benchmarks/out/REPORT.md`.

| Option | Meaning |
|---|---|
| `--backend mock\|sam` | run in-process through `Engine` (default `mock`) |
| `--url URL` | instead, POST to a running service's `/separate` |
| `--seconds N` | length of each quality fixture (default 6) |
| `--latency-seconds N` | length of the latency fixture, `0` to skip (default 30) |
| `--repeat N` | separations per fixture; reported latency is the median (default 1) |
| `--long-seconds N` | length of the long fixtures (`competing_voices`), `0` to skip (default 24) |
| `--fixtures a,b` | subset of `speech_plus_siren,two_speakers,music_mix,competing_voices` |
| `--speech-mode auto\|say\|synthetic` | speech source (see below) |
| `--passes N` | `options.passes` (multi-pass; default 1) |
| `--write-fixtures [DIR]` | also write the WAVs + `manifest.json` for Person 3 / Person 4 |
| `--out DIR` | report directory (default `benchmarks/out`) |
| `--timeout N` | HTTP timeout in seconds (default 600) |
| `--sweep NAME\|FILE.json` | run every config of a sweep in one process (see "Settings sweeps") |

### Separation settings

Each of these is sent as a `metadata.options` field **only when the flag is given**, so a
plain run posts exactly what Person 4's backend posts. Defaults are the service's own.

| Flag | `metadata.options` field |
|---|---|
| `--seed N` | `seed` (defaults to 1234 for in-process `--backend sam`; unset otherwise) |
| `--candidates K` | `reranking_candidates` (model-level re-ranking) |
| `--window-candidates K` | `candidates` (candidate separations per chunk window) |
| `--select first\|overlap\|mean` | `candidate_select` |
| `--refine off\|mask` | `refine` |
| `--refine-p P` | `refine_p` (mask exponent) |
| `--refine-nfft N` | `refine_n_fft` |
| `--refine-floor F` | `refine_floor` |
| `--refine-blend B` | `refine_blend` (1 = fully refined) |
| `--refine-threshold T` | `refine_threshold` (0..1) |
| `--refine-residual-threshold T` | `refine_residual_threshold` (0..1): mask values above `1 - T` become 1, so the target is cleared from the residual |
| `--refine-gate-db D` | `refine_gate_db` (0..80) |
| `--chunk-seconds S` | `chunk_seconds` (0 = whole file) |

In-process only (a warning is printed and they are ignored with `--url`):

| Flag | Effect |
|---|---|
| `--chunk-overlap S` | `settings.chunk_overlap_seconds` |
| `--ode-step S` | ODE step: the backend's `ode_opt["options"]["step_size"]` when it has one (real model), else `settings.ode_step_size` |
| `--model-id ID` | `settings.model_id` before the engine is built |

`report.json` → `settings` records every knob, the `request_options` actually sent, the
effective engine values (`effective_model_id`, `effective_ode_step`,
`effective_chunk_overlap`) and `peak_rss_bytes` (`getrusage(RUSAGE_SELF).ru_maxrss`,
normalised to bytes: macOS reports bytes, Linux kilobytes).

### Settings sweeps

```sh
.venv/bin/python -m benchmarks.run_benchmark --backend sam --sweep quality-v1 --out benchmarks/out/quality-v1
.venv/bin/python -m benchmarks.run_benchmark --backend sam --sweep my_configs.json
```

`--sweep` loads the engine **once**, then for each config applies its in-process knobs
(ODE step, chunk overlap; a different `model_id` forces one reload) and request options,
runs every fixture and writes `<out>/<config>/report.json` + `REPORT.md`. It also writes
`<out>/sweep.json` and `<out>/SWEEP.md`: one table, one row per config, with per-fixture
SI-SDRi, **leakage and resid-leak** plus mean SI-SDRi, mean leakage, mean resid-leak,
RTF (latency fixture) and peak RSS. Every cell shows the Δ against the **first config (the
baseline)**: SI-SDRi up is better, both leakages down are better. A config's `passes`
(like every other key) applies to that config only; the next config falls back to the
command-line value. Peak RSS is the process high-water mark so far, so it only
grows across rows.

Named sweeps live in `benchmarks/sweeps.py`:

| Sweep | Configs |
|---|---|
| `ode` | ODE step 0.0625 (baseline), 0.03125, 0.015625 |
| `refine` | refine off; mask with p ∈ {1, 2} × n_fft ∈ {1024, 2048, 4096} |
| `refine-floor` | mask p=2 n_fft=2048 with floor ∈ {0, 0.02} × blend ∈ {1.0, 0.5} |
| `leakage` | mask p=2 n_fft=2048 with threshold ∈ {0, 0.3, 0.5} × gate_db ∈ {0, 30}; plus p=4 threshold 0.3 |
| `leakage-v2` | leakage in both directions, all mask p=2 n_fft=2048: `mask` (baseline), `thr0.5`, `rthr0.5` (residual_threshold 0.5), `both0.5`, `both0.7`, `p4-both0.5`, `passes2`, `passes2-both0.5`, `k3auto-both0.5` (window candidates 3, select auto), `k3auto-passes2-both0.5` |
| `candidates` | baseline K=1 at chunk 10; K ∈ {2, 3} × select ∈ {overlap, mean} at chunk 10; K=2 overlap at chunk 15 and whole file |
| `chunks` | chunk 10 / 15 / 20 / 0 (whole file), K=1 |
| `quality-v1` | baseline; refine mask p=2 n_fft=2048; K=2 overlap; both combined |

A JSON sweep file is a list of configs (or `{"name": ..., "configs": [...]}`); each config
is a dict keyed by the flag dest names (`ode_step`, `window_candidates`, `select`,
`refine`, `refine_p`, `refine_nfft`, `refine_floor`, `refine_blend`, `refine_threshold`,
`refine_residual_threshold`, `refine_gate_db`, `chunk_seconds`, `chunk_overlap`, `candidates`, `seed`, `passes`,
`model_id`) plus `"name"`. Keys a config omits keep the command-line value:

```json
[{"name": "baseline"}, {"name": "mask", "refine": "mask", "refine_p": 2, "refine_nfft": 2048}]
```

## Fixtures

All 48 kHz; the quality fixtures are stereo (sources panned apart) so the engine's stereo
path is exercised, and the latency fixture is mono. The mixture is **exactly** the sum of
its ground-truth sources, which is what makes the reconstruction check meaningful.

| Fixture | Prompt | Ground-truth target | Why it exists |
|---|---|---|---|
| `speech_plus_siren` | `siren` | `siren` | The cinematic demo: argument over a background siren, siren 6 dB down and panned right. If this one is bad, the demo is bad. |
| `two_speakers` | `man speaking` | `speaker_low` | Speaker-separation experiment: two talkers at the same level, ~100 Hz vs ~205 Hz f0, different speaking rhythms, panned apart. Deliberately hard -- a text prompt cannot name a person. |
| `music_mix` | `guitar` | `guitar` | Music-separation experiment: percussive kit (kick + bright bursts) against a sustained plucked harmonic tone. Their low bands overlap, so band-based cheating does not score here. |
| `competing_voices` | `man speaking` | `speaker_low` | *Long* fixture (`--long-seconds`, default 24 s). Two voices (low pool vs high pool), panned apart, taking turns and talking over each other. At 24 s, A (low, target) speaks 0-5, 8-13, 16-24 s and B (high) 3-9, 12-15, 18-24 s; the schedule scales with the length. Turns straddle 10 s chunk boundaries, so a separator that follows a different talker per window loses SI-SDR here even when each window looks fine. `extra.other_prompt` is `woman speaking`. Not part of `QUALITY_FIXTURES` (it is in `LONG_FIXTURES`). |
| `long_mixture_<n>s` | `siren` | `siren` | Latency / real-time factor only. Mono so the number is per inference pass. |

**Speech source.** With `--speech-mode auto` (the default) real speech is rendered with
macOS `say` and decoded through `audio_io.decode`; on Linux (CI) or if `say` fails, a
synthetic formant/harmonic voice with syllable-rate modulation is used instead. Every
report row records which was used in `speech_source`, and `--speech-mode synthetic` forces
the portable path so two machines can be compared directly. Numbers from a `say` run and a
synthetic run are **not** comparable with each other.

To hand the same audio to other teams:

```sh
.venv/bin/python -m benchmarks.fixtures --out fixtures/generated --seconds 6
```

That writes `<fixture>/mixture.wav`, one WAV per ground-truth source and a `manifest.json`
listing prompts, source names, roles (`target` / `interference`), durations and file names.
Paths in the manifest are relative to the manifest, so the directory can be moved or
mounted anywhere.

## Metrics, and how to read the report

| Metric | Direction | What it means |
|---|---|---|
| **SI-SDR** (dB) | higher | Scale-invariant signal-to-distortion ratio of the target stem against ground truth. The reference is first least-squares scaled, so a correct-but-quieter stem is not punished. Above ~10 dB the wanted sound clearly dominates; above ~15 dB it is clean to the ear. |
| **input SI-SDR** (dB) | — | The same measure on the untouched mixture: the score for doing nothing at all. |
| **SI-SDRi** (dB) | higher | **The headline number**: SI-SDR minus input SI-SDR. `0` means the stem is no better than the mixture; negative means separation made things worse. |
| **SDR** (dB) | higher | Scale-*dependent* variant. It punishes level errors, which matters because the editor drops the stem straight back into a mix. A large SI-SDR with a poor SDR means the right sound at the wrong gain. |
| **Leakage** (dB) | lower | Energy of non-target sound inside the target stem, relative to the wanted sound, split per time-frequency bin using the ground truth. `-20 dB` is clean; `0 dB` means half the stem is the wrong sound. This is what decides the demo: if the siren stem carries the dialogue, automating the siren's volume ducks the argument too. |
| **Resid leak** (dB) | lower | The other direction: energy of the *target* sound still inside the residual stem, relative to the wanted non-target content (`spectral_leakage_db(residual, interference, target)`). The residual is what the user hears after "Mute target" / "Lower target", so this is the "removed sound is still faintly there" complaint. `n/a` when the service returns no residual stem. Pushed down by `refine_residual_threshold`. |
| **Recon err** | lower | Max absolute sample error of `target + residual - mixture`. `~1e-7` is float32 round-off (in-process), `~1e-4` is 16-bit PCM quantisation from the WAV round-trip over HTTP. Anything larger is a pipeline bug: lost samples, a length drift in chunking, or a double gain. |
| **Latency** (s) | lower | Wall clock for one separation, median over `--repeat`. In-process it is the `Engine.separate` call; over HTTP it includes transfer, decode and WAV encode -- the number the application actually sees. |
| **RTF** | lower | Real-time factor: seconds of processing per second of audio. `< 1` is faster than real time. This is the "latency per second of audio" figure; the `long_mixture` row is the one to quote. |

`report.json` carries everything the markdown shows plus per-repeat latencies, the service
`model` block, the warnings the service returned and the environment (platform, python,
numpy). JSON cannot hold `inf`/`nan`, so non-finite metrics are `null` there and render as
`+inf` / `-inf` / `n/a` in the markdown.

## Mock numbers are not quality numbers

The default backend is `mock`: a prompt-selected band-pass filter, **not** source
separation. SAM Audio checkpoints are gated and not yet approved for this machine, so the
mock is the only backend that runs here.

A mock run proves the *pipeline*: fixtures build, prompts reach the backend, the target and
residual stems come back with the right roles, lengths and sample rate, the stems sum back
to the mixture, and the metrics and timing path work. Every mock report states
`Mock backend: YES` and carries the disclaimer banner.

Do not quote mock SI-SDR, SI-SDRi or leakage as separation quality anywhere -- not in the
README, not in the demo script, not in a status update. Expect the mock to score around
0 dB SI-SDRi on `speech_plus_siren` and `two_speakers` (a band-pass cannot pull a siren out
from under a voice) and positive leakage on the siren case; that is the mock behaving
correctly, not a regression.

## Running against a real SAM Audio service

Once the checkpoints are approved, nothing about the benchmark changes except the backend.

In-process (needs the `sam` extra and a GPU):

```sh
pip install -e '.[sam]'
AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-large \
  .venv/bin/python -m benchmarks.run_benchmark --backend sam --seconds 10 --repeat 3
```

Against a service over HTTP (the same call Person 4's backend makes):

```sh
AUDIO_BACKEND=sam .venv/bin/python -m sam_audio_service &      # or the container
.venv/bin/python -m benchmarks.run_benchmark --url http://127.0.0.1:8001/separate --repeat 3
```

In `--url` mode the runner posts `files={"audio": ...}` + `data={"metadata": "{\"prompts\":
[...]}"}`, exactly like `app/media.py`, and reads `stems[].audio_base64` / `stems[].role`
back. It takes the backend name and `is_mock` from the response's `model` block, so a
report produced this way says truthfully which service answered -- point it at a mock
service and the report still says `Mock backend: YES`.

What to check on the first real run:

1. `Mock backend: no` in `REPORT.md`, and `model_id` / `device` are what you expect.
2. `speech_plus_siren` SI-SDRi clearly positive and leakage well below 0 dB -- that is the
   demo working.
3. Recon err still at quantisation level: the real backend must not change stem lengths.
4. RTF on the `long_mixture` row: with `AUDIO_CHUNK_SECONDS` set, re-run to confirm
   chunking does not cost quality (SI-SDRi should not drop) before relying on it for long
   videos.
