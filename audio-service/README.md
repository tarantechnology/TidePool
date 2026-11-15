# SoundCanvas audio inference service (Person 1)

AudioSep text separation by default, with optional Meta SAM Audio text/span/visual
separation, exposed as a small HTTP service that Person 4's backend calls. See `docs/audio_integration_handoff.md`
at the repository root for the cross-team contract and open questions.

## Run locally (mock backend, no GPU)

```bash
cd audio-service
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
AUDIO_BACKEND=mock .venv/bin/python -m sam_audio_service  # http://localhost:8001/docs
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

Two safeguards stop a chain whose later pass is not a leftover of the first: a pass louder than
`AUDIO_PASS_MAX_RATIO_DB` (default -3 dB) relative to pass 1, or whose long-term spectrum is less
similar than `AUDIO_PASS_MIN_SIMILARITY` (cosine, default 0.6) to pass 1's, is discarded and the
chain stops (`stop_reason` = `too_much` | `dissimilar`). Measured 2026-11-13 with `sam-audio-large`:
without them a second "guitar" pass on a residual with no guitar left pulled the drums in and dropped
the stem from 19.6 to 0.9 dB SI-SDRi. The mock backend runs with both disabled.

Each target stem reports `refinement = {passes_requested, passes_run, per_pass_energy_db,
stopped_early, stop_reason?}`; `model.passes` is the most passes run for any prompt. Compare quality across
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
| `options.refine_residual_threshold` (`AUDIO_REFINE_RESIDUAL_THRESHOLD`) | `0.0` (off) | [0, 1): mirrored soft knee, `M' = 1 if M > 1 - t else M / (1 - t)`, applied after `refine_threshold` and before the gate/floor. Bins where the target dominates go fully to the target, so the residual (what "Mute target" / "Lower target" leaves) no longer carries a faint copy of the removed sound. 0.3-0.5 is a sensible range; with `refine_threshold` also set the mask becomes near-binary (cleaner stems both ways, more artefacts). Measured by the benchmark's `residual_leakage_db`. |
| `options.refine_gate_db` (`AUDIO_REFINE_GATE_DB`) | `0` (off) | [0, 80]: silence the stem where the sound is not really present. STFT frames whose masked target is more than this many dB below the mixture frame are zeroed in every bin, with a 3-frame hold so words are not clipped. |
| `options.candidates` (`AUDIO_CANDIDATES`) | `1` | 1-8 independent noise draws per chunk window; the engine keeps one per window (below). Distinct from `reranking_candidates` (ranked inside the model, needs a ranker); both can be combined at `K x R` cost. Costs `K x` compute and activation memory per window; the window batch is divided by `K` so one model call never exceeds `AUDIO_CHUNK_BATCH` rows. Backends that cannot draw several (the mock) add a warning and use one draw. |
| `options.candidate_select` (`AUDIO_CANDIDATE_SELECT`) | `overlap` | `overlap`: the draw whose head best correlates with what the previous window put into the shared overlap (first window / quiet overlap: medoid for `K >= 3`, else draw 0), so neighbouring windows agree and seams do not flip. `mean`: average of the draws (ensemble). `first`: draw 0. |

`model.refine` reports the effective `{mode, p, n_fft, floor, blend, threshold, residual_threshold, gate_db}` (absent when
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

## Default AudioSep backend

AudioSep is the default **text-only** separator. It uses the AudioSep base checkpoint and
CLAP text encoder; it is not SAM Audio and has no visual conditioning, time-anchor
conditioning, independent candidates or judge-ranked retries. The editor reports those
limitations. A timed mute is still an edit applied after separation. AudioSep may also
extract speech, music or other sounds; review the target and edited mix before applying.

Install its pinned inference dependencies with `pip install -e ".[audiosep]". The runtime is separate from the `sam` extra. Use Python 3.11 or newer; the upstream Conda environment file targets Python 3.10 and must be adapted. It requires a separately
prepared [upstream AudioSep checkout](https://github.com/Audio-AGI/AudioSep), its Python
and PyTorch/torchaudio dependencies, and the official AudioSep and CLAP checkpoints. The
upstream repository is MIT licensed; check the terms for model weights and dependencies
separately. Model checkpoints are never included in this repository.

The A40 integration was tested with upstream revision `944583f18b84589dc965de3ad77525c945334252`,
Python 3.12, PyTorch/torchaudio 2.8, CUDA, and the two small compatibility changes in
`patches/audiosep-checkpoint-compat.patch`. The patch prevents a second implicit CLAP load
and removes a RoBERTa buffer that newer transformers no longer stores. To prepare a **new**
checkout, follow upstream's environment and checkpoint instructions, then apply the patch:

```bash
# Run from the SoundCanvas repository root; do not overwrite an existing checkout.
git clone https://github.com/Audio-AGI/AudioSep.git /path/to/AudioSep
git -C /path/to/AudioSep checkout 944583f18b84589dc965de3ad77525c945334252
git -C /path/to/AudioSep apply "$PWD/audio-service/patches/audiosep-checkpoint-compat.patch"
# Install the service into the prepared AudioSep environment.
python -m pip install -e ./audio-service
```

Place the upstream `audiosep_base_4M_steps.ckpt` and
`music_speech_audioset_epoch_15_esc_89.98.pt` checkpoints under the checkout's
`checkpoint/` directory, as instructed by upstream. The RoBERTa tokenizer/model must also
be available through the Hugging Face cache or network. PyTorch's checkpoint loader is
allow-listed for the NumPy scalar metadata in these weights; use trusted checkpoints only.
Configure custom checkpoint locations using `AUDIO_AUDIOSEP_CHECKPOINT` and
`AUDIO_AUDIOSEP_CLAP_CHECKPOINT` if needed.

```bash
export AUDIO_BACKEND=audiosep
export AUDIO_AUDIOSEP_ROOT=/path/to/AudioSep
export AUDIO_DEVICE=cuda
python -m sam_audio_service
```

Point the project backend at `http://127.0.0.1:8001/separate` with
`INFERENCE_ADAPTER=http` and `SAM_AUDIO_SERVICE_URL`. `/health` should report
`backend: audiosep`, `is_mock: false`, `loaded: true`, and the intended device. The engine
serializes model inference, processes 10-second windows with two seconds of overlap and a
batch size of one by default, and constructs the residual from the original mixture. It
resamples mono model input to 32 kHz and returns it to the 48 kHz service timeline; the
residual retains the original channels. The service default mask refinement is off; an editor preview profile may explicitly request it. Real quality,
latency and memory vary by recording; sample-accurate reconstruction does not guarantee
correct isolation. Do not mistake the optional experimental spectral repairs for the
standard AudioSep backend.

### TidePool voice preservation

AudioSep defaults to `AUDIO_SUPPRESSION=conservative` and
`AUDIO_SUPPRESSION_STRENGTH=0.9`. The engine rebuilds the target from the original
mixture phase through a selective ratio mask and limits processing when the target
estimate is weak. This runs once after chunk reconstruction, preserving the full
stereo timeline. The returned target is what is removed; target + residual still
reconstruct the original. An additional legacy refinement mask is skipped in this mode.

Request `options.suppression=off` for raw comparisons. `suppression_strength=1`
uses stronger cleanup; `.9` retains some target to ease damage to other sounds.
`model.suppression` and `/capabilities.defaults.suppression` report the effective
settings. No source-separation method guarantees unchanged voices during overlap.

The official AudioSep/CLAP checkpoints are reused from the checkout when present.
Otherwise the adapter uses the Hugging Face cache and verified Space revision
`0807a713283035d8b0d0628f1b8e3d6580459f7b`. Explicit missing checkpoint paths fail
clearly. No SAM model or mock fallback is loaded by the AudioSep path.

## Music stems and model routing

A song can be split into one track per instrument (vocals, drums, bass, other; guitar and
piano with a 6-stem model) by a dedicated music-source-separation model instead of the
prompted text model, which is weak on music stems. Every request is routed first
(`sam_audio_service/routing.py`); the decision is returned as `routing` in every response
and can be asked for without separating anything with `POST /v1/route` (multipart, same as
`/separate`). Plan: `docs/plans/music-stems.md`. Contract: `docs/contracts/audio_contracts.md`
(Contract B `options.mode` and Contract C `routing`).

```bash
# CI / local, no weights: frequency-band mock splitter.
AUDIO_STEM_SPLITTER=mock .venv/bin/python -m sam_audio_service
curl -F audio=@song.wav -F 'metadata={"options":{"mode":"music_stems"}}' localhost:8001/separate
# Real model (GPU host): installs demucs + transformers; weights download on first use.
uv pip install --python .venv/bin/python -e ".[music]"
AUDIO_STEM_SPLITTER=demucs AUDIO_STEM_MODEL=htdemucs_ft .venv/bin/python -m sam_audio_service
```

Routing in `auto` (the default): no prompts → split into every stem; every prompt names a
stem ("vocals", "the beat", "karaoke"…) **and** the content classifier says music → split
into those stems plus one `residual`; anything else (mixed prompts, non-stem prompts, span
or visual guidance, not music) → the prompted model exactly as before. Without a classifier,
stem prompts route to the splitter only when the caller sends `hints.source_kind: "audio"`;
a "singing" prompt on a video soundtrack stays prompted. `options.mode` (`prompted` /
`music_stems`) overrides all of it. The full table is in the `routing.py` docstring.

The splitter's stems are made to sum exactly to the input (the error is spread over the
stems in proportion to their energy), so unmuting every track reproduces the upload. A full
split returns no residual; a subset returns one `residual` (the sum of the other stems).

| Variable | Default | Effect |
| --- | --- | --- |
| `AUDIO_STEM_SPLITTER` | `off` | `off`, `mock` (frequency bands; CI only), `demucs`, `chain` (RoFormer vocal model via `audio-separator`, then Demucs on `mixture - vocals`; `pip install -e ".[music,music-chain]"`), or `roformer` (native multi-stem BS-RoFormer such as ZFTurbo's 4-stem checkpoint; `pip install -e ".[music-roformer]"`). Off by default for every backend, the mock included, so existing deployments and the "empty prompts → 400" contract are unchanged until it is switched on. |
| `AUDIO_STEM_MODEL` | `htdemucs_ft` | Demucs model: `htdemucs`, `htdemucs_ft` (4 stems, ~4x slower than `htdemucs`), `htdemucs_6s` (adds guitar and piano). May change after the Phase 0 benchmark. A name ending in `6s` also gives the mock six bands. |
| `AUDIO_STEM_VOCAL_OVERLAP` | `8` | `chain` only: audio-separator `mdxc_params.overlap` for the vocal model (1-16). To be set from the benchmark. |
| `AUDIO_STEM_ROFORMER_CKPT` / `AUDIO_STEM_ROFORMER_CONFIG` | none | `roformer` only (both required): checkpoint and its training YAML. The model is built from the YAML's `model` section (keys the installed `bs_roformer` does not accept are dropped and logged); a checkpoint missing more than 2% of the parameters fails the load. |
| `AUDIO_STEM_ROFORMER_OVERLAP` | `2` | `roformer` only: windows per `audio.chunk_size` (step = chunk / overlap), overlap-added with a triangular window (1-16). |
| `AUDIO_STEM_ROFORMER_BATCH` | `1` | `roformer` only: windows per forward pass (1-16). |
| `AUDIO_STEM_ROFORMER_FP16` | `false` | `roformer` only: run the model in float16. |
| `AUDIO_STEM_VOCAL_MODEL` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | `chain` only: the `audio-separator` vocal model. `AUDIO_STEM_MODEL_DIR` sets where its weights live. |
| `AUDIO_STEM_SHIFTS` | `1` | Demucs random-shift averaging (0-20). Cost scales linearly. |
| `AUDIO_STEM_OVERLAP` | `0.25` | Demucs segment overlap, `[0, 1)`. |
| `AUDIO_STEM_SEGMENT` | model default | Demucs segment length in seconds; smaller uses less GPU memory (`htdemucs` caps it at ~7.8). |
| `AUDIO_ROUTING_DEFAULT` | `auto` | `options.mode` when a request sets none: `auto`, `prompted`, `music_stems` (the last needs a splitter). |
| `AUDIO_ROUTER_CLASSIFIER` | `panns` with `demucs`/`chain`/`roformer`, else `off` | `off`, `mock` (everything is music; tests), `panns` (PANNs Cnn14 via `panns-inference`, 32 kHz; checkpoint `AUDIO_ROUTER_PANNS_CHECKPOINT`, default `~/panns_data/Cnn14_mAP=0.431.pth`), `ast` (`MIT/ast-finetuned-audioset-10-10-0.4593`, 16 kHz). Both score up to eight 10 s windows across the file and load on first use. A classifier that cannot load is disabled for the process and `/health` reports why; routing then uses the prompt rules alone (`scores: null`). |
| `AUDIO_ROUTER_MUSIC_THRESHOLD` | per classifier: `panns` 0.4, `ast` 0.2 | Minimum "song" score (median over up to 8 10 s windows of AudioSet Music minus Speech) for stem-name prompts to go to the splitter. Speech over music (vlogs, soundtracks) scores ~0 and stays on the prompted model; full songs score ~0.8 (PANNs). |
| `AUDIO_STEM_LABELS` | `auto` | `auto` tags every music stem and names the `other` stem after what it sounds like ("Guitar", "Synth & Strings"; `stems[].tags`, see `labeling.py`) whenever a tagger can run. Each stem is described only with its own vocabulary (vocals: Male/Female singing, Rapping, Choir, Synth vocals, Spoken word, else "Singing"; drums: Drum kit/Drum machine + details; bass: Bass guitar/Synth bass/Double bass; other: melodic instruments). `off` leaves `tags: null` and `"Other"`. Labeling failures only add a warning. |
| `AUDIO_STEM_LABEL_MODEL` | `router` | Tagger for stem labels: `router` reuses the router's AudioSet classifier (PANNs; no second model, ~0.8 s warm for 4 stems of a 4-min song on a busy A40), `panns`/`ast` load their own copy when the router uses another, `clap` loads LAION-CLAP zero-shot (`AUDIO_STEM_LABEL_CLAP_MODEL`, default `laion/larger_clap_music_and_speech`, Apache-2.0). PANNs is the default: AST is ~3x slower and weaker on vocals; CLAP is strong on clean strings but mislabels separated vocals ("Synth vocals") and bass (2026-11-14 evaluation, docs/plans/music-stems.md). |
| `AUDIO_STEM_LABEL_MIN_SCORE` | `0` | Floor on the per-family presence thresholds (`labeling.PRESENCE`, calibrated per tagger on synthetic BabySlakh stems) and on drum detail tags; `0` = calibrated thresholds only. |
| `AUDIO_STEM_LABEL_MAX_WINDOWS` | `12` | Active 10 s windows tagged per stem (silent and bleed-only windows are skipped). |

The splitter and classifier load lazily on first use (not at startup), share the GPU and
the inference lock with the prompted model, and are reported under `splitter` and
`classifier` in `/health`; `/capabilities` lists `music_stems` (stems, aliases) and
`routing` (including the splitter's weight `license`). Splitters report progress through
an optional callback (the service logs each 10%; RoFormer reports per chunk, chain per
stage, Demucs only start and end because `apply_model` has no hook) and accept a cancel
event checked between chunks (`409 CANCELLED`; nothing in the synchronous HTTP API sets it
yet). The chain splitter disables audio-separator's peak normalisation
(`normalization_threshold=1.0`), feeds it at most 0.5 peak and undoes that gain on the vocals.
Demucs works at 44.1 kHz stereo: the service resamples 48 kHz ↔ 44.1 kHz inside
the splitter, duplicates mono to stereo and folds the stems back to mono.

## Generate parts that fit the song (chat-fx Phase E)

"add violins", "add a piano part", "add a funky bassline", "add an 80s synth arpeggio", "a trumpet line in the
chorus", "add a choir" ... -> a new `music` layer the same length as the song, plus the MIDI it rendered.
Code: `sam_audio_service/music_analysis.py` and `sam_audio_service/generation/`.

Pipeline: **analyse** (stem-aware: drums -> beats/swing/kick times/density, bass -> chord roots via Basic
Pitch, vocals -> melody MIDI + where the voice is active + room estimate, other stems -> harmony chroma and
register occupancy; per-section local key) -> **resolve the ask** (`vocabulary.py`: instrument table + all
128 GM programs + synth presets -> family, role, register, articulation, density, style) -> **compose**
(rule-based voice-leading arranger, or the Anticipatory Music Transformer restricted to the requested GM
program and snapped to the beat grid) -> **render** (VSCO-2 CE samples through a numpy sampler, FluidSynth +
FluidR3_GM for everything else, a built-in synth for pads/arps/leads/bass) -> **mix in** (role HPF, vocal-keyed
presence-band carve, reverb from the estimated RT60, loudness relative to the song, light ducking, peak guard)
-> objective **checks** in every response. The rule-based arranger is the default and only enabled path; `engine=amt` and `engine=musicgen` are experimental, off by default and return 400 unless enabled. `engine=musicgen` replaces compose+render with MusicGen-melody
(text + song chroma); its weights are CC-BY-NC-4.0 and it is refused unless `AUDIO_GEN_ALLOW_NONCOMMERCIAL=true`.

| Endpoint | Body | Returns |
| --- | --- | --- |
| `POST /v1/analyze-music` | multipart `audio` (mix, optional) + `stem_<name>` files + `metadata` `{"melody": true}`; or JSON `{audio_ref, stems: {name: ref}}` | `{analysis: {duration, tempo, meter, beats, downbeats, key, chords[{start,end,chord,root_pc,quality,confidence}], sections[{start,end,label,name,energy,key}], melody_midi, vocal_activity, groove, register, space, context, warnings}}` |
| `POST /v1/generate-part` | same inputs; `metadata` `{instrument?, prompt?, role?, style?, sections?\|interval?, density?, engine: auto\|symbolic\|amt\|musicgen, seed, level_db, duck_db, analysis?, return_mode, output_format}` | `{part{name,kind:"music",instrument,family,role,...}, audio{wav, sample_rate, sample_count, audio_base64\|audio_ref}, midi_base64, notes, engine{name, license, commercial_use}, renderer{used, license}, windows, mix{loudness, reverb, eq_carve, ducking, peak}, checks{notes, audio}, analysis, warnings}` |

`GET /capabilities` has a `generate` block: `{available, instruments, roles, engines{symbolic, amt, musicgen,
stable_audio_open}, renderers{vsco, gm, synth}, analysis}`. Pass the `analysis` from `/v1/analyze-music` back
into `/v1/generate-part` to skip re-analysis (about 10-20 s per minute of audio on the A40 with Basic Pitch).

| Variable | Default | Effect |
| --- | --- | --- |
| `AUDIO_GEN_ENABLED` | `true` | Turn the feature off. |
| `AUDIO_GEN_SAMPLES_DIR` | unset | VSCO-2 CE root with `manifest.json` (build it with `scripts/build_vsco_manifest.py <root>`). Unset: GM/synth fallbacks. |
| `AUDIO_GEN_SOUNDFONT` | system `FluidR3_GM.sf2` | GM soundfont for the FluidSynth renderer. |
| `AUDIO_GEN_BEATS` / `AUDIO_GEN_TRANSCRIBER` | `auto` | `beat_this` / `basic_pitch` when importable, else numpy tracker / no melody. |
| `AUDIO_GEN_AMT_MODEL` | `off` | Experimental AMT engine, e.g. `stanford-crfm/music-medium-800k`. Off by default: listening tests preferred the rule-based arranger. |
| `AUDIO_GEN_AUDIOGEN` | `off` | `musicgen` loads MusicGen-melody (`AUDIO_GEN_MUSICGEN_MODEL`). |
| `AUDIO_GEN_ALLOW_NONCOMMERCIAL` | `false` | Required to run MusicGen (CC-BY-NC-4.0 weights). |

Licences: beat_this MIT (the JKU checkpoint host returned HTTP 500 on 2026-11-14, so the default loads the
MIT safetensors export `safe-models/beat-this-final0`); Basic Pitch Apache-2.0; AMT code + weights Apache-2.0;
VSCO-2 CE CC0; FluidR3_GM MIT; FluidSynth LGPL-2.1 (run as a separate process); MusicGen weights
CC-BY-NC-4.0; Stable Audio Open is gated behind a licence agreement and not installed. Evaluation:
`benchmarks/generate_parts.py --stems <musdb track dir> --start 0 --seconds 75 --out <dir> --engines symbolic,amt,musicgen`.

## Speech and rhythm tools (`/v1/words`, `/v1/enhance`, `/v1/beats`)

These are the machine-learning operations behind the chat edits in `docs/plans/chat-fx.md`
(Phase B): "bleep the swear words", "cut the ums", "clean up the voice", "add X when he
says 'what'" and "on the beat drop". The endpoints are multipart, like `/separate`. The
contract is Contract F in `docs/contracts/audio_contracts.md`. Each tool is **off** until
its backend variable is set. Each loads on first use (or at startup with
`AUDIO_LOAD_ON_STARTUP`) and has its **own lock**, so these tools never queue behind a
separation, and a separation never queues behind them.

| Tool | Model | Licence |
| --- | --- | --- |
| `/v1/words` | faster-whisper `large-v3-turbo` (CTranslate2 fp16, Silero VAD, `word_timestamps`), then WhisperX-style CTC forced alignment with torchaudio `WAV2VEC2_ASR_BASE_960H` (English) | faster-whisper MIT; Whisper weights MIT; wav2vec2 weights MIT (fairseq) |
| `/v1/enhance` | DeepFilterNet3 (48 kHz) | code and weights MIT OR Apache-2.0 |
| `/v1/beats` | beat_this `final0` (CPJKU, ISMIR 2024), with librosa as the fallback | code and weights MIT; librosa ISC |

```bash
# CI / local, no weights: deterministic mocks (NOT real models).
AUDIO_WORDS_BACKEND=mock AUDIO_ENHANCE_BACKEND=mock AUDIO_BEATS_BACKEND=mock .venv/bin/python -m sam_audio_service
curl -F audio=@clip.wav -F 'metadata={"language":"en"}' localhost:8001/v1/words
curl -F audio=@dialogue.wav -F 'metadata={"atten_lim_db":12}' localhost:8001/v1/enhance
curl -F audio=@song.wav localhost:8001/v1/beats
```

**Install (GPU host, Python 3.12, torch 2.8.0+cu128).** Two of the packages cannot be
resolved normally, so they are installed with `--no-deps`. `deepfilternet` 0.5.6 pins
`numpy<2`, and its Rust feature library (DeepFilterLib) only has wheels up to CPython 3.11.
The service replaces that library with `speech/libdf_shim.py`, a NumPy port that matches
the real wheel within 1.5e-6 (`benchmarks/speech.py check-libdf`).

```bash
uv pip install -e ".[speech-words,speech-enhance,speech-beats]"
uv pip install --no-deps deepfilternet==0.5.6   # the model code only; no DeepFilterLib needed
```

| Variable | Default | Effect |
| --- | --- | --- |
| `AUDIO_WORDS_BACKEND` | `off` | `off`, `mock` (one fixed word per voiced region; tests), or `faster_whisper`. |
| `AUDIO_WORDS_MODEL` | `large-v3-turbo` | Any faster-whisper name (`large-v3`, `distil-large-v3`…), a CTranslate2 directory or an HF repo. |
| `AUDIO_WORDS_COMPUTE_TYPE` | `auto` | `auto` means `float16` on CUDA and `int8` on CPU. |
| `AUDIO_WORDS_BEAM_SIZE` / `AUDIO_WORDS_BATCH_SIZE` | `5` / `8` | A batch size above 1 uses faster-whisper's batched VAD pipeline. 1 uses sequential decoding. |
| `AUDIO_WORDS_VAD` | `true` | Silero VAD filter (it cuts hallucinations in silence and music). |
| `AUDIO_WORDS_ALIGN` | `wav2vec2` | `off` keeps Whisper's own word times. `wav2vec2` re-times English words (see benchmarks). Per request, use `metadata.align`. |
| `AUDIO_WORDS_PROFANITY_FILE` | bundled `speech/profanity_en.txt` | One `word` or `prefix:stem` per line, matched per word. Per request, use `metadata.extra_profanity`. |
| `AUDIO_WORDS_MIN_PAUSE_SECONDS` | `0.7` | Gap reported in `pauses`. |
| `AUDIO_ENHANCE_BACKEND` | `off` | `off`, `mock` (a spectral gate; tests), or `deepfilternet`. |
| `AUDIO_ENHANCE_MODEL` | `DeepFilterNet3` | `DeepFilterNet3`, `DeepFilterNet2` or `DeepFilterNet` (downloaded to `~/.cache/DeepFilterNet`). |
| `AUDIO_ENHANCE_ATTEN_LIMIT_DB` | unlimited | Default strength. Per request, use `metadata.atten_lim_db` (6 is gentle, 12-20 moderate). |
| `AUDIO_ENHANCE_CHUNK_SECONDS` | `60` | Window length for long inputs, with 2 s of context on each side (0 = whole file). |
| `AUDIO_BEATS_BACKEND` | `off` | `off`, `mock` (a 120 BPM grid; tests), `beat_this`, or `librosa`. |
| `AUDIO_BEATS_MODEL` | `final0` | A beat_this checkpoint name, path or URL. If the official host (cloud.cp.jku.at) fails, `final0` falls back to the byte-identical HF mirror `Audante/beat-this-final0`, pinned by SHA-256. |
| `AUDIO_BEATS_DBN` | `false` | madmom DBN post-processing (needs madmom; not installed). |
| `AUDIO_BEATS_FALLBACK` | `librosa` | Tracker used when beat_this cannot load or run. It gives beats and tempo but no downbeats. Set `off` to fail instead. |
| `AUDIO_SPEECH_DEVICE` | `auto` (= `AUDIO_DEVICE`) | Device for these models. |

Benchmarks (A40, 2026-11-14): `benchmarks/speech.py`, with raw JSON in
`benchmarks/results/2026-11-14-speech/raw/`.

| Tool | Measure | Result |
| --- | --- | --- |
| words (turbo fp16, beam 5, batch 8, VAD) | LibriSpeech test-clean, 100 utterances: WER | 3.65% |
| | word boundary error against MFA references (3914 words): Whisper only → with alignment (wav2vec2 plus a 50 ms start shift) | median 50 → 23 ms, p90 180 → 82 ms, within 80 ms 66% → 90% (starts alone: 95% within 80 ms). Without the shift, aligned starts ran +62 ms late: `words_turbo_noshift.json` |
| | spliced TTS target words (sample-exact references), word starts | median 16 ms, 84% within 80 ms. Ends are not scored: the reference keeps about 150 ms of TTS decay |
| | 4-min speech, warm: Whisper only / with alignment | 3.4 s / 4.3 s; process VRAM 5.7 / 6.9 GB peak (CTranslate2 cache and CUDA context included), of which torch 0.86 GB |
| | Scripted Piper TTS, 6 voices × 25 lines: profanity / filler recall, false positives on 204 trap words | 63/78 (81%; 12 of the 15 misses are TTS "shit" heard as "it"/"hit") / 29/42 (69%; "uh" is often dropped) / 0 |
| enhance (DeepFilterNet3) | SI-SDR, 20 LibriSpeech utterances + ESC-50 noise at 0/5/10 dB SNR | −0.0/5.0/10.0 → 14.5/16.7/19.1 dB (12 dB limit: 9.5/13.6/17.3) |
| | same with pink noise; clean pass-through | 0.2/5.2/10.2 → 8.1/11.4/14.6 dB; clean speech kept at 31.3 dB |
| | 4-min stereo, 48 kHz, warm | 3.7 s, process VRAM 2.9 GB (torch 1.4 GB) |
| beats (beat_this final0) | synthetic drum loops, 8 tempos and meters: beat F (70 ms) / downbeat F | 0.988 (librosa: 0.948) / 0.74 (3/4 is weak) |
| | 230 s song, warm | 0.49 s, process VRAM 0.7 GB; 125 BPM, 4/4 |
