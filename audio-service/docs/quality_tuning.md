# Separation quality tuning (measured 2026-11-13)

What each knob does to the stems, what it costs, and the profiles the backend uses. Numbers are
from `benchmarks/run_benchmark.py --sweep ...` (`benchmarks/results/2026-11-13-quality-sweep*/`),
mean over four fixtures with known clean sources: speech + siren, two speakers, music (guitar
vs drums), and a 24 s two-talker clip whose turns straddle chunk boundaries. Metrics: **SI-SDRi**
(accuracy, higher is better), **leakage** (other sounds left inside the target stem, lower is
better; the "faint background still audible" complaint) and **residual leakage** (the removed
sound left inside the residual, lower is better; what you hear after Mute).

## Headline

| Setup | Mean SI-SDRi | Mean leakage | Mean residual leakage | GPU time |
| --- | --- | --- | --- | --- |
| Raw model output (before this work) | -9.7 dB | -7.1 dB | | 1x |
| Mask refinement (now the default for the real model) | 0.6 dB (+10.3) | -8.6 dB (-1.5) | -3.3 dB | 1x (CPU post-processing) |
| Quality profile: 3 draws per window + mask + both thresholds | 7.7 dB (+17.4) | -14.5 dB (-7.3) | -9.7 dB (-6.4) | 2.6x |

`sam-audio-large`, NVIDIA A40, real speech (macOS `say` voices via `SOUNDCANVAS_SAY_CACHE`).
On `sam-audio-small` (Apple M1 Max, MPS) the mask alone was +3.3 dB SI-SDRi / -3.6 dB leakage
and improved every fixture. The raw large model scores so low on the two-speaker fixtures because
a single draw often extracts the *wrong* speaker for "man speaking" (see Candidates).

## Knobs

### Mask refinement (`options.refine = "mask"`, default for the real backend)
Rebuilds the stem from the mixture through a soft time-frequency mask derived from the model's
own target/residual split (`refine.py`). Removes codec and phase error and the halo around the
decision the model made. `p` (2) sharpens the mask; `n_fft` (2048) sets the frequency/time
trade-off (1024 lost 1.4 dB, 4096 gained 0.2 dB on the small model). Free: numpy on CPU.

### Leakage threshold (`refine_threshold`, quality profile 0.5)
Zeroes mask bins the model only half-attenuated (soft knee). Small model: -1.9 dB leakage for
-2.3 dB SI-SDRi at 0.5; 0.3 is the gentler setting (-1.2 / -1.4). Use it when the stem must be
clean rather than complete.

### Residual threshold (`refine_residual_threshold`, quality profile 0.5)
Mirror image: bins where the target dominates are moved fully out of the residual, so Mute leaves
no ghost. Large model, music fixture: +3.0 dB SI-SDRi and -3.3 dB residual leakage, +1.3 dB
target leakage. Pairs with the leakage threshold (`both0.5`) in the quality profile.

### Frame gate (`refine_gate_db`)
Silences frames where the stem is far below the mixture. Never triggered at 10-30 dB on any
fixture; left available, default off.

### Candidates (`options.candidates`, `candidate_select`)
K noise draws per chunk window, returned to the service and picked by rule. `auto` (default rule):
when chunked, the draw that agrees with the previous window's overlap (seams do not flip
speakers); for a single window, the consensus draw when all draws agree and their average when
they disagree. Why: with the large model, "man speaking" on two TTS voices picks a different
speaker per draw; K=1 produced a -57 dB stem (the woman), K=3 recovered ~0 dB on that clip and
+10.9 dB on the long one. **Use K=3**: with K=2 the first window cannot out-vote a wrong first
draw, and chaining consistency onto it produced -45 dB. Cost: K x per chunk (A40: 1.3x real time
at K=3). Averaging agreeing draws smears music (-4 dB), hence the agreement test.

### Chunk length (`chunk_seconds`)
10 s is safe. 20 s windows collapsed the long two-talker clip by 10 dB (small) and 19 dB (large)
without candidate selection; K=2 overlap selection repaired the small-model case to within 0.2 dB.
Whole-file (0) is best when memory allows (+8.5 dB on the long clip with the large model) and is
what visual prompts already do.

### Solver steps (`AUDIO_ODE_STEP_SIZE`)
1/32 and 1/64 changed SI-SDRi by < 0.05 dB for 1.3-2x the time. Keep 1/16.

### Multi-pass (`options.passes`) and its safeguards
A second pass on the residual helps when the first pass left some of the sound behind (long
two-talker clip: -11.8 dB residual leakage, +8 dB SI-SDRi). It can also hallucinate: a second
"guitar" pass on a residual with no guitar pulled the drums in (19.6 -> 0.9 dB). Two guards now
discard such a pass and stop the chain: louder than `AUDIO_PASS_MAX_RATIO_DB` (-3 dB) relative to
pass 1, or long-term spectrum less similar than `AUDIO_PASS_MIN_SIMILARITY` (0.6) to pass 1.
`refinement.stop_reason` says which fired. Passes stay user-driven (editor "Passes" / "Extract more").

## Backend profiles (`app/guidance.py`, env-overridable)
- **preview**: mask refinement, no thresholds, K=1, 10 s chunks. Accurate and fast.
- **quality** ("Quality retry"): K=3 `auto`, mask with `refine_threshold` 0.5 and
  `refine_residual_threshold` 0.5. Cleanest stems and residuals; ~2.6x the GPU time.
  Works without the judge re-ranker (`capabilities.quality_mode = "consistency"`); with
  `QUALITY_RERANKING=true` and a loaded text ranker it also re-ranks inside the model.

## Reproduce
```sh
# Mac (small model, MPS):
cd audio-service && PYTORCH_ENABLE_MPS_FALLBACK=1 AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-small AUDIO_DEVICE=mps \
  AUDIO_LOAD_SPAN_PREDICTOR=false AUDIO_LOAD_RANKERS=false .venv-sam/bin/python -m benchmarks.run_benchmark \
  --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode say --seed 1234 --sweep quality-v1 --out benchmarks/out/q
# GPU host (large model): render the voices on a Mac once with SOUNDCANVAS_SAY_CACHE, rsync the
# directory to $WS/say-cache, then
bash scripts/runpod_setup.sh bench --sweep quality-v1 --out /workspace/bench/quality-v1
```
