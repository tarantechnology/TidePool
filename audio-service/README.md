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

**macOS is not a supported runtime for the real model** (verified 2026-11-06 on Apple Silicon,
Python 3.12): `pip install ".[sam]"` fails because `perception_models` requires `decord`, which has
no macOS/cp312 wheel; installing around it with `--no-deps` then fails at import time on
`xformers`, which also has no macOS wheel. Upstream issue facebookresearch/sam-audio#29 documents
the same two blockers and the source patches people apply. The Linux CUDA image (`Dockerfile`
target `sam`) is unaffected: both packages ship manylinux wheels.

The mock backend (`AUDIO_BACKEND=mock`) is a band-pass filter, not separation; every response
produced by it carries `model.is_mock=true` and a warning.
