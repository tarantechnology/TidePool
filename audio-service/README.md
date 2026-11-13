# SoundCanvas audio inference service (Person 1)

Text-, span- and visually-prompted sound separation with Meta's SAM Audio, exposed as a
small HTTP service that Person 4's backend calls. See `docs/audio_integration_handoff.md`
at the repository root for the cross-team contract and open questions.

## Run locally (mock backend, no GPU)

```bash
cd audio-service
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m sam_audio_service        # http://localhost:8001/docs
.venv/bin/python -m pytest -q
```

Point Person 4's backend at it with `INFERENCE_ADAPTER=http SAM_AUDIO_SERVICE_URL=http://localhost:8001/separate`,
or run both with the compose overlay: `docker compose -f compose.yaml -f audio-service/compose.audio.yaml up --build`.

## Real model

First real run (2026-11-07, `sam-audio-small`, Apple M1 Max via MPS): SI-SDR improvement 10–13 dB
on the speech+siren, two-speaker and music fixtures, 4–7.5 s processing per second of audio. See
`benchmarks/results/`.

```bash
uv pip install --python .venv/bin/python -e ".[sam]"   # Python >= 3.11, CUDA GPU recommended
hf auth login                                           # after access is approved on huggingface.co/facebook/sam-audio-large
AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-large .venv/bin/python -m sam_audio_service
.venv/bin/python scripts/smoke_sam.py --help            # single-window smoke test, no HTTP
```

`AUDIO_BACKEND=sam` defaults `AUDIO_CHUNK_SECONDS` to 20 (upstream recommends 10-30 s windows;
memory scales with window length). Extra variables, all optional:

| Variable | Default | Effect |
| --- | --- | --- |
| `AUDIO_CHUNK_BATCH` | `1` | How many chunk windows go into ONE model batch (1-8). Batch `k` costs roughly `k x` the activation memory of one window (and buys per-call overhead, not a `k x` speedup); on any GPU allocation failure - `out of memory`, `CUBLAS_STATUS_ALLOC_FAILED`, `CUDNN_STATUS_ALLOC_FAILED`, `cudaErrorMemoryAllocation`, an illegal memory access or `torch.OutOfMemoryError` - the whole signal is retried at batch 1. `AUDIO_CHUNK_WORKERS` is a deprecated alias that now sets this and logs a warning. See `docs/chunk_concurrency.md`. |
| `AUDIO_MAX_PASSES` | `3` | Upper bound (1-5) for `options.passes`; a request above it is `400 INVALID_REQUEST`. Each pass is a full model run. |
| `AUDIO_VISUAL_MASK_MODE` | `keep_object` | `keep_object` keeps the object and zeroes the background (the paper's `V ⊙ M`); `notebook` zeroes the object, as the official notebook does. Upstream issue #75 is unresolved, so this is an A/B switch. |
| `AUDIO_LOAD_SPAN_PREDICTOR` | `true` | `false` drops the multi-billion-parameter PE-A-Frame span predictor (big memory saving, no span prediction). |
| `AUDIO_LOAD_RANKERS` | `true` | `false` drops the CLAP/Judge/ImageBind re-rankers (only used when `AUDIO_RERANKING_CANDIDATES > 1`). |
| `AUDIO_ODE_METHOD` / `AUDIO_ODE_STEP_SIZE` | `midpoint` / `0.0625` | Flow-matching solver. |
| `AUDIO_FRAME_SIZE` | `336` | Frames and masks are pre-resized to this; the vision encoder resizes to 336 regardless. |

Memory, user-reported upstream (not measured by us): small ~21.6 GB to load / ~28 GB peak on
15 s, base ~24 GB to load, large ~38 GB peak on 15 s with ~30 s the practical maximum on an
80 GB A100. Recommended starting points: 24 GB -> small/base, `AUDIO_LOAD_SPAN_PREDICTOR=false`,
`AUDIO_CHUNK_SECONDS=10`-`15`; 40-48 GB -> large, `AUDIO_CHUNK_SECONDS=15`-`20`; 80 GB -> large,
`AUDIO_CHUNK_SECONDS=30`, `AUDIO_RERANKING_CANDIDATES` up to 8. Keep `AUDIO_DTYPE=float32`:
fp16 is reported to break separation quality and is refused; `bfloat16` is an opt-in experiment.

**Dependency pins that matter:** `huggingface_hub<1` and `transformers<5`. With `huggingface_hub`
1.x/2.x the upstream loader fails with `TypeError: BaseModel._from_pretrained() missing 2 required
keyword-only arguments: 'proxies' and 'resume_download'` (found on our first real load); the `sam`
extra pins both.

**Running it on a Mac anyway (development only):** `scripts/build_macos_sam_env.sh` builds
`audio-service/.venv-sam` by installing `sam-audio` and `perception_models` without their
dependencies and guarding the `xformers` imports (the patches from upstream issue #29), then
downloads `facebook/sam-audio-small`. `scripts/run_local_real.sh` at the repository root then starts
the audio service on MPS with the span predictor and re-rankers off, the backend and the editor.
Measured on an M1 Max 64 GB: about 1-1.5 s of processing per second of audio, ~18 GB of memory.

**macOS is not a supported runtime for the real model** (verified 2026-11-06 on Apple Silicon,
Python 3.12): `pip install ".[sam]"` fails because `perception_models` requires `decord`, which has
no macOS/cp312 wheel; installing around it with `--no-deps` then fails at import time on
`xformers`, which also has no macOS wheel. Upstream issue facebookresearch/sam-audio#29 documents
the same two blockers and the source patches people apply. The Linux CUDA image (`Dockerfile`
target `sam`) is unaffected: both packages ship manylinux wheels.

The mock backend (`AUDIO_BACKEND=mock`) is a band-pass filter, not separation; every response
produced by it carries `model.is_mock=true` and a warning.

### Multi-pass separation (`options.passes`)

SAM Audio sometimes only attenuates the described sound: the target stem has it, but the
residual still carries a quieter copy. `options.passes = n` (1..`AUDIO_MAX_PASSES`, default 1)
reruns the same prompt (text, anchors and visual prompt) on that prompt's residual and adds
what it extracts to the target:

| Request option | Default | Effect |
| --- | --- | --- |
| `options.passes` | `1` | Pass 1 separates the mixture; pass `k` separates `mixture - (t1 + ... + t(k-1))` and the stem is `t1 + ... + tk`. Each prompt has its own residual chain; the residual stem is still `mixture - all targets`, so stems sum to the mixture exactly. Chunking, `per_channel` stereo, anchors and visual intervals apply unchanged to every pass. A pass extracting less than -30 dB of pass 1's energy is discarded and stops the chain (with a warning), as does a silent residual. Cost is up to `n x` per prompt. |

Each target stem reports `refinement = {passes_requested, passes_run, per_pass_energy_db,
stopped_early}`; `model.passes` is the most passes run for any prompt. Compare quality across
pass counts with `python -m benchmarks.run_benchmark --backend sam --passes 2`.

### Mask refinement and per-window candidates (opt-in, both off by default)

| Request option (env default) | Default | Effect |
| --- | --- | --- |
| `options.refine` (`AUDIO_REFINE`) | `off` | `mask` rebuilds each prompt's summed target from the mixture through a soft ratio mask `M = \|T\|^p / (\|T\|^p + \|mix - T\|^p)` computed from the model's target, applied to every channel's STFT (`refine.py`). Keeps the model's decision, drops codec artefacts. Runs after all passes; the residual is still `mixture - targets`, and a visual prompt's target stays exactly zero outside its interval. CPU cost: a few STFTs per prompt, no extra model calls. |
| `options.refine_p` (`AUDIO_REFINE_P`) | `2.0` | Mask exponent, (0, 8]. 2 = Wiener (power) ratio, 1 = amplitude ratio, higher = sharper (more separation, more artefacts). |
| `options.refine_n_fft` (`AUDIO_REFINE_NFFT`) | `2048` | 512 / 1024 / 2048 / 4096; hop is `n_fft / 4`. |
| `options.refine_floor` (`AUDIO_REFINE_FLOOR`) | `0.0` | Minimum mask value, [0, 0.5]: fewer musical-noise holes, more leakage. |
| `options.refine_blend` (`AUDIO_REFINE_BLEND`) | `1.0` | [0, 1]: `blend * refined + (1 - blend) * model target`. |
| `options.refine_threshold` (`AUDIO_REFINE_THRESHOLD`) | `0.0` | [0, 1): soft-knee cut, `M' = 0 if M < t else (M - t) / (1 - t)`. 0.3-0.5 removes background bins the model only partially attenuated (the faint, muffled rest of the mix), accepting some artefacts. |
| `options.refine_gate_db` (`AUDIO_REFINE_GATE_DB`) | `0` (off) | [0, 80]: silence the stem where the sound is not really present. STFT frames whose masked target is more than this many dB below the mixture frame are zeroed in every bin, with a 3-frame hold so words are not clipped. |
| `options.candidates` (`AUDIO_CANDIDATES`) | `1` | 1-8 independent noise draws per chunk window; the engine keeps one per window (below). Distinct from `reranking_candidates` (ranked inside the model, needs a ranker); both can be combined at `K x R` cost. Costs `K x` compute and activation memory per window; the window batch is divided by `K` so one model call never exceeds `AUDIO_CHUNK_BATCH` rows. Backends that cannot draw several (the mock) add a warning and use one draw. |
| `options.candidate_select` (`AUDIO_CANDIDATE_SELECT`) | `overlap` | `overlap`: the draw whose head best correlates with what the previous window put into the shared overlap (first window / quiet overlap: medoid for `K >= 3`, else draw 0), so neighbouring windows agree and seams do not flip. `mean`: average of the draws (ensemble). `first`: draw 0. |

`model.refine` reports the effective `{mode, p, n_fft, floor, blend, threshold, gate_db}` (absent when
off); `model.candidates` / `model.candidate_select` report K and the rule (`first` when K = 1).
`/capabilities` `defaults` lists all of them.

## GPU deployment and readiness

For NVIDIA hardware with Docker's NVIDIA Container Toolkit and approved Hugging Face model access, run:

```bash
HF_TOKEN=... docker compose -f compose.yaml -f audio-service/compose.audio.yaml -f audio-service/compose.gpu.yaml up --build
```

The GPU override requests one NVIDIA device and sets `AUDIO_DEVICE=cuda`; model startup fails rather than falling back to CPU if CUDA is unavailable. `/health` is liveness and `/ready` returns 503 until the model is loaded. GPU access, weights and model performance must be checked on the target host. The base overlay remains mock/CPU by default.

The default response contains IEEE float32 WAV stems. This preserves values outside [-1, 1] so the target and residual can reconstruct the mixture without clipping. Explicit `options.output_format=pcm_s16le` is a lossy compatibility option. The backend inference timeout defaults to 3600 seconds and is configurable via `INFERENCE_TIMEOUT_SECONDS`; retry failed jobs by submitting a new process request. A long timeout does not guarantee completion for all inputs and hardware.

Text-prompted audio uses overlapping windows (10 seconds in the cloud configuration) with crossfades for aligned reassembly. `AUDIO_CHUNK_BATCH` bounds how many of those windows go into one model batch; concurrent requests are still serialized so the setting does not multiply GPU usage across requests. Visual-prompted selections run as a single aligned interval and ignore this setting.

**Fixed: the `[16, 32, 3]` inference failure (was "known regression" in PR #18).** Running chunk windows on a thread pool against the one shared model raced `WatermarkEncoderBlock.forward_no_conv` in the DAC-VAE decoder (`dacvae/model/dacvae.py:326-332`), which swaps a `Conv1d(1, 32, 7)` for `nn.Identity()` for the duration of its own forward pass. A second thread then read a 1-channel tensor where 32 channels were expected and raised:

```text
INFERENCE_FAILURE: RuntimeError: Given groups=1, weight of size [16, 32, 3], expected input[2, 1, 480002] to have 32 channels, but got 1 channels instead
```

Windows are now batched instead of threaded: the engine hands up to `AUDIO_CHUNK_BATCH` windows to the backend in one processor batch and one `model.separate` call, and the backend serializes all model access with a lock. `AUDIO_CHUNK_WORKERS` keeps working as a deprecated alias for `AUDIO_CHUNK_BATCH`. The full analysis, the memory implications and the operator guidance are in [`docs/chunk_concurrency.md`](docs/chunk_concurrency.md). Do not include private media or credentials in bug reports.

### Optional candidate ranking and label-free benchmark

The service can generate several candidates and use Meta's pretrained audio judge to
choose one. This costs extra GPU memory and inference time; it is a preference score,
not a guarantee of better quality. On an A40, to avoid also loading the much larger
visual ranker and the CLAP ensemble, set `AUDIO_LOAD_RANKERS=true`,
`AUDIO_JUDGE_ONLY=true`, and `AUDIO_LOAD_VISUAL_RANKER=false`. Send
`options.reranking_candidates` greater than one. The service returns a clear error
if ranking is unavailable rather than claiming to rank candidates. Visual requests
without a visual ranker use the text judge and still need a sound description.

For an automatic reference-based experiment without hand-labeling, install `pyarrow`,
`soundfile`, `scipy`, and `httpx` alongside this package. Download the validation
parquet from `hf-internal-testing/librispeech_asr_dummy` and supply a local music file:

```sh
python benchmarks/real_audio.py --speech-parquet /path/to/validation.parquet \
  --music /path/to/music.wav --out ./benchmark-real --clips 3 --repeat 2
```

The script mixes known clean speech with music and compares the separation to the
known speech using SI-SDR improvement. Add `--candidates 2` to measure ranking;
add `--demucs-python /path/to/python` to compare an installed Demucs model. Its
checkpoint must be obtainable from its trusted official source. These constructed
mixtures do not measure object association or cover all real-world material, so
compare several recordings and listen as well. The script sends constructed mixtures
to the specified inference URL (localhost by default).

A small A40 run is recorded in `benchmarks/real-audio-baseline.json` and
`benchmarks/real-audio-judge.json`: three six-second mixtures, two descriptions,
and a single matched seed averaged 6.88 dB SI-SDR improvement without ranking and
6.89 dB with two candidates. This sample is far too small to demonstrate a reliable
quality advantage. The Demucs comparison could not run because its official
checkpoint server presented an expired TLS certificate at the time of testing.

### Guided preview configuration

The default SAM window is now 10 seconds, with 2-second overlap and batch two. Requests
can set `options.chunk_batch` without mutating shared service settings; the engine still
serializes model access. Responses report requested/effective batch size, window/overlap,
and peak allocated GPU bytes. Batched GPU OOM retries sequentially and reports that fallback.
Automatic span prediction and its heavyweight loader are disabled by default.

The editor's Quality retry uses two judge-ranked candidates. A 20-second/batch-one retry
and per-channel stereo separation are benchmark options, not promoted UI profiles.
See [the guided benchmark procedure](benchmarks/GUIDED.md) for the five-category corpus,
provenance, fixed splits/seeds, metrics, regression gate and staged rollback procedure.
Visual selection intervals no longer become sound anchors. Explicit anchors remain supported;
invalid visual guidance fails instead of silently changing to text-only inference.
