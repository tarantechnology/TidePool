# Audio integration handoff (Person 1 → Persons 2, 3, 4)

**Runpod update (2026-11-13):** real SAM 3.1 knife masks and SAM Audio Small TV text/visual/temporal inference run on an L4. Schema 1.0 and the audio HTTP API remain unchanged. Optional `AUDIO_VISUAL_BATCH_SIZE` limits the existing PE encoder microbatch. [Results and reproduction](../vision/docs/RUNPOD.md) report measured execution separately from unverified object-sound correctness and speech/music preservation.

**Issue #4 implementation update (2026-11-07):** the backend now validates frozen
CV selections, attaches stored native-frame detections and a project-owned
`source.original_ref`/SHA256/FPS, propagates model warnings/errors, and enforces
full-timeline visual stems. See [current validation and runtime audit](../vision/docs/ISSUE4.md).
The historical requests/status tables below describe PR #13's original handoff;
they are not the current backend implementation. Real visual-guided inference
remains unverified; the successful M1 Max experiment below was text-only.

Person 1 owns the audio AI pipeline: SAM Audio inference, audio preprocessing for the model,
target/residual stems, audio benchmarking, the audio inference HTTP API and the model-level
interpretation of visual prompts. All of it lives in `audio-service/`. We do not own
uploads, storage, the job queue, detection/segmentation/tracking, the editor or export.

**Every schema in this document and in [`docs/contracts/audio_contracts.md`](contracts/audio_contracts.md)
is PROPOSED, awaiting team agreement.** Nothing here requires another team to change code to
keep working today: the service already implements Person 4's existing `/separate` contract
exactly as documented in `INTEGRATION.md`, and returns a strict superset of what their
worker reads.

**Real SAM Audio inference has run once, on 2026-11-07, with `facebook/sam-audio-small` on an
Apple M1 Max (MPS, float32, span predictor and re-rankers disabled) through this service's
backend.** Measured on our synthetic-but-real-speech fixtures (`benchmarks/results/2026-11-07-sam-small-mps/`):
SI-SDR improvement 11.6 dB isolating a siren from speech, 10.2 dB isolating one of two
speakers by "man speaking", 12.9 dB isolating a guitar-like tone from a percussive mix;
4–7.5 s of processing per second of audio on this hardware. No CUDA run exists yet, so the
GPU figures below remain user-reported upstream numbers. Hugging Face access for the team
account is approved. Everything else below is verified against the code, the mock backend
and the upstream sources.

| | |
| --- | --- |
| Service | `audio-service/` — FastAPI, `python -m sam_audio_service`, default port 8001 |
| Default URL | `http://audio-service:8001/separate` (Person 4's `SAM_AUDIO_SERVICE_URL` default) |
| Contract version | `2025-11-draft-1` (returned in every response and by `/health`) |
| Contract details | [`docs/contracts/audio_contracts.md`](contracts/audio_contracts.md) — Contracts A–E |
| Editor fixture | `audio-service/fixtures/editor/` + `scripts/make_editor_fixture.py` |
| Test suite | `cd audio-service && .venv/bin/python -m pytest -q` — passes on CPU with the mock backend; CI: `.github/workflows/audio-service.yml` |

---

## Cross-team test status

| # | Test | Status | Why |
| --- | --- | --- | --- |
| 1 | Editor → backend → audio (Person 3) | **Blocked** — substituted | No Person 3 code exists in the repository. Substituted by `audio-service/fixtures/editor/` (a Contract C manifest + playable WAVs from `scripts/make_editor_fixture.py`) and `tests/test_editor_fixture.py`, which pin the payload the editor will consume. Executed with the **mock** backend. |
| 2 | Vision → audio (Person 2) | **Translation half executed with Person 2's real output; model half blocked** | `audio-service/fixtures/vision/` holds `detections.json` + `selection.json` produced by Person 2's actual `yolo26n-seg` + ByteTrack pipeline on their documented pedestrian clip; `tests/test_person2_artifacts.py` drives them through our translator, frame/mask preparation and `/separate` (mock backend) with zero mismatches. Since PR #15/#18 the **merged backend does deliver** the original video reference and the per-frame masks for these same artefacts end to end — see [Merged workflow (PR #15/#18)](#merged-workflow-pr-1518) and `tests/test_merged_visual_workflow.py`. Speaker-specific separation itself still cannot run: no CUDA GPU. |
| 3 | Backend → audio (Person 4) | **Executed, mock backend, in-process and in Docker** | `tests/test_integration_person4_backend.py` imports Person 4's real `app/` with `INFERENCE_ADAPTER=http` against a live instance of this service. Additionally the compose overlay was built and run (`docker compose -f compose.yaml -f audio-service/compose.audio.yaml up`) and `scripts/e2e_compose.py` walked create project → upload → `/process` → poll → stem fetch → automation → webhook → MP4 export against the two containers. Stem duration matches the extracted audio, 48 kHz stereo, export probes as h264 + aac with the source duration. Person 4's code is imported, never modified. `tests/test_merged_visual_workflow.py` extends this to the visually guided job on Person 2's real artefacts: `metadata.source.original_ref` + `sha256`, both Person 2 documents under `metadata.visual_prompt`, a real `VisualInput` with per-frame masks at the backend boundary, interval spans and full-length stems. |
| 4 | Cinematic demo (all four) | **Audio half executed, mock backend, in Docker** | The same compose run used a synthetic "two voices + siren over video" clip through separation, a stem automation curve, the natural-language webhook and an MP4 export. The editor half needs Person 3; the visual-prompt half needs Test 2's blockers cleared. |

Every response produced by the mock backend carries `model.is_mock: true` and a warning
saying the stems are band-pass filtered copies of the mixture, not separation.

## Merged workflow (PR #15/#18)

**Status (2026-11-11): the merged backend does deliver the video reference and the
per-frame masks.** This section records exactly what arrives, verified end to end by
`audio-service/tests/test_merged_visual_workflow.py`, which runs Person 4's real `app/`
(`INFERENCE_ADAPTER=http`) against a live instance of this service (`AUDIO_BACKEND=mock`)
on Person 2's real artefacts in `audio-service/fixtures/vision/`, and asserts on *our*
side of the HTTP boundary. No code of Person 4's, Person 2's or Person 3's is changed, and
nothing in `visual_prompts.py` had to be loosened — the merged payload translates as is.

### The request a visually guided job sends

`app/worker.py` POSTs multipart `audio` (the project's `extracted/audio.wav`) plus a
`metadata` form field. For a job started by `POST /api/projects/{pid}/process` with a body
of `{"prompts": [...], "visual_prompt": <Person 2 selection 1.0>}`, `metadata` is exactly:

```jsonc
{
  "prompts": ["person"],                 // app/main.py Process.prompts, list[str] only
  "visual_prompt": {                     // built by app/visual.py::prepare_payload
    "selection": { /* Person 2 selection 1.0, frozen, verbatim */ },
    "detections": { /* Person 2 detections 1.0, from projects.visual, verbatim */ }
  },
  "source": {                            // app/visual.py::project_source
    "asset_id":     "<project_id>",
    "original_ref": "file:///<STORAGE_ROOT>/projects/<project_id>/original/<uid>.mp4",
    "sha256":       "<sha256 of the original uploaded bytes>",
    "duration":     12.0,
    "frame_rate":   10.0
  },
  "asset_id":   "<project_id>",          // legacy top-level correlation field
  "request_id": "<job_id>"               // added by app/worker.py
}
```

Key names, nesting and provenance:

- `visual_prompt.selection` is the client's selection document, unmodified. The backend
  *freezes* it: `prepare_payload` passes it through `validate_selection`, which requires
  schema 1.0, identity agreement with the stored detections (`video_id`, `source_sha256`,
  `analysis_id`), native frame alignment, a 0.1–30 s interval inside
  `processed_duration_seconds`, dense native-FPS coverage of the interval, and non-empty
  masks — so a selection we receive has already been checked against the stored document.
- `visual_prompt.detections` is **the stored document**, read from the `projects.visual`
  column (`PUT /api/projects/{pid}/visual-metadata`, or written by the `vision` job in
  `app/worker.py`), not from the client: `reject_media_refs` refuses any client-supplied
  `video_ref`/`original_ref`/`mask_video_ref`/`detections_ref`/`audio_ref`, and an inline
  `detections` that differs from the stored one is a 422. If the client sends no
  `visual_prompt` at all, the stored document alone is forwarded.
- `source.original_ref` is `Path(project['original']).resolve().as_uri()` — the project's
  immutable upload under `STORAGE_ROOT/projects/<project_id>/original/`. The backend
  re-verifies its SHA-256 in `project_source()` when the payload is built *and* again in
  `app/worker.py` when the queued job starts (`'Uploaded media hash changed after the
  visual job was queued'`), so a swapped file fails the job instead of being separated.
- `source.frame_rate` and `source.duration` come from `ffprobe` on the original.
- There is **no `mask_video_ref` and no `detections_ref`**: masks travel as native-resolution
  RLE inside `visual_prompt.detections`. Contract D's `has_model_input()` (video + mask
  video) is therefore never satisfied by this payload; the envelope path in
  `translate_visual_prompt` is what consumes it, and it builds `VisualInput.frame_masks`
  from the RLE itself.
- `metadata.anchors` and `metadata.options` are never sent (`Process` has only `prompts`
  and `visual_prompt`), so service defaults apply — see "Forward the whole `/process` body".

### What we do with it

`translate_visual_prompt` takes the envelope branch (`isinstance(raw["selection"], dict)`),
reads `video_ref` from `metadata.source.original_ref`, verifies
`detections.source_sha256 == source.sha256`, clamps the interval to
`detections.processed_duration_seconds`, decodes the selected track's per-frame RLE,
re-checks the declared `fps` against `ffprobe` on the resolved video and validates dense
coverage, then returns a `VisualInput(video_path, interval, frame_masks, frame_rate,
frame_size, object_id="track:<id>", label, source_sha256)` plus a `("+", start, end)`
anchor. On Person 2's fixture (768×576, 10 fps, track 7, interval 0–6 s) this produces
**zero warnings**: 49 masked frames of the interval's 60, the rest correctly empty
foreground. The engine then runs the interval as one window (no chunking) and pads the
result back to the full timeline, so stems keep `start_time = 0.0` and the target's
`spans` are `[[0.0, 6.0]]`.

### `AUDIO_MEDIA_ROOT` and the `file://` refs

`original_ref` is an **absolute path produced inside Person 4's container**, and
`sam_audio_service/media.py::resolve_ref` requires the resolved path to live under
`AUDIO_MEDIA_ROOT`. So the two containers must see the original at the *same absolute
path*. `compose.yaml` + `audio-service/compose.audio.yaml` already arrange that:

| | `soundcanvas` (Person 4) | `audio-service` (Person 1) |
| --- | --- | --- |
| Volume | `./storage:/data` | `./storage:/data:ro` |
| Variable | `STORAGE_ROOT: /data` | `AUDIO_MEDIA_ROOT: /data` |

Run it with `docker compose -f compose.yaml -f audio-service/compose.audio.yaml up --build`.
Two consequences worth keeping in mind: overriding `STORAGE_ROOT` without changing the
audio container's mount breaks every `file://` ref with `INVALID_REQUEST: file reference is
outside AUDIO_MEDIA_ROOT`; and leaving `AUDIO_MEDIA_ROOT` unset disables file refs
entirely, which silently degrades visual prompting to span anchors.

### Mismatches with what we expect (none blocking)

1. **No mismatch in the payload itself.** The merged envelope is translated with no
   warnings and no lenient handling added.
2. **`stems[].spans` is dropped on the way back.** `app/media.py` copies only
   `model`, `warnings`, `source`, `request_id` and `contract_version` into the job's
   `inference` report, and `app/core.py::stem()` returns only the `stems` table columns, so
   the editor cannot see which interval the model treated as active. The test therefore
   asserts `spans` on our side (an `Engine.separate` hook). One extra key in that
   `report.update(...)` tuple would expose it — Person 4's call.
3. **The whole detections document rides in every request.** The fixture's 6 s / 60-frame
   document makes the `metadata` form field ≈ 395 KiB (≈ 6.7 KiB per native frame at
   768×576); a 30 s clip at 30 fps would be ≈ 6 MiB per job. Harmless at demo length, but
   `detections_ref` (a `file://` path under the shared volume, which we already support)
   would remove it entirely, and the backend owns the stored document anyway.
4. **One shared `visual_prompt` for all prompts.** `Process.visual_prompt` is a single
   object applied to every entry of `prompts`, so two prompts both get the same mask track.
   Per-prompt guidance needs `prompts[]` as objects (`PromptSpec.visual_prompt`).
5. **`/process` with no selection ships the detections document and no `source`.** When
   `visual_prompt` is absent but the project has stored detections, `prepare_payload` takes
   its `selection is None` branch (`payload['visual_prompt'] = raw; return payload`), so the
   whole document is forwarded *without* `metadata.source`. We cannot use it — no selected
   object, no video reference — and answer with the warning `detections given without a
   selected object (selection, selected_track_id or selected_detection_id)`. The job still
   succeeds as a text-only separation; it is wasted bandwidth, not breakage, and
   `test_process_without_a_selection_forwards_detections_that_cannot_steer` pins exactly
   that behaviour. Suggested fix on Person 4's side: forward the stored document only when
   a selection accompanies it.

### Still unverified

The masks reach the model input — `BackendPrompt.visual` is non-`None` at the backend
boundary and `prepare_masked_video` decodes the frames and masks from the resolved
`file://` video — but with `AUDIO_BACKEND=mock` the backend *ignores* them. Real
visually guided separation remains the one open item on issue #4 and needs a CUDA GPU.

---

## What Person 1 Provides

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/separate` | **Person 4's existing contract.** Multipart: file part `audio`, form field `metadata` (JSON string). |
| `POST` | `/v1/separate` | *Proposed.* Same JSON as `metadata`, plus `audio_ref` instead of uploaded bytes — for when both services share a volume. |
| `GET` | `/health` | `{"status": "ok"\|"degraded", "version", "contract_version", "model": {...}}`. `degraded` means the checkpoint is not loaded; `model.load_error` says why. Use `/ready` for the container readiness check; it returns 503 until the model is loaded. |
| `GET` | `/capabilities` | Machine-readable prompting modes, limits, defaults, output format and the full `error_codes` table. Read this instead of hard-coding our limits. |
| `GET` | `/docs`, `/openapi.json` | FastAPI's generated schema for everything above. |

No authentication: this service is for a trusted network only, exactly as `INTEGRATION.md`
already says about Person 4 → Person 1 traffic. It must not be exposed to the internet.

### Input requirements

- `POST /separate`: multipart with a file part named `audio` and a form field named
  `metadata` containing the request JSON as a **string**. Person 4 already sends precisely
  this.
- Audio: **anything FFmpeg can decode** (WAV/MP3/MP4/MOV, audio or video container). Person
  4 sends 48 kHz stereo PCM16 WAV, which takes a fast path with no FFmpeg subprocess.
  Non-48 kHz input is resampled; more than 2 channels is downmixed to 2; mono stays mono.
- Limits (all env-configurable, all reported by `/capabilities`):
  `AUDIO_MAX_INPUT_BYTES` 500 MiB, `AUDIO_MAX_DURATION_SECONDS` 600 s,
  `AUDIO_MAX_PROMPTS` 8.
- Metadata: Contract B. The only required content is at least one prompt. Unknown keys are
  accepted and ignored, so forwarding extra fields is always safe.
- Media references (`audio_ref`, `video_ref`, `mask_video_ref`, `detections_ref`) must be
  either `file://` paths that resolve **under `AUDIO_MEDIA_ROOT`** (symlink-escape is
  rejected) or `http(s)` URLs whose host is in `AUDIO_ALLOWED_MEDIA_HOSTS`. Nothing else is
  fetched.

### Output schema

Contract C: `stems[]` + `model` + `source` + `request_id` + `warnings[]` +
`contract_version`. Each stem carries `name`, `audio_base64` (or `audio_ref`), `stem_id`,
`role` (`target`|`residual`), `prompt`, `label`, `start_time`, `duration`, `sample_rate`,
`channels`, `sample_count`, `format`, `spans`, `source_asset_id`.

Guarantees, in one place:

- Output defaults to **48 kHz, IEEE float32 WAV** (complete RIFF file, base64, no
  data-URI prefix). Channel count **matches the source** (1 or 2).
- **Every stem is full-timeline**: `start_time == 0.0`, and `duration` / `sample_count`
  equal `source.duration` / `source.sample_count` for every stem including the residual. We
  never return a clip of the detected event, so nothing downstream has to reconstruct an
  offset.
- One stem per prompt, plus one `residual` stem unless `options.residual: "none"`. The stem
  name is `prompts[].name` if given, else **the caller's own wording** of the prompt with
  whitespace collapsed (case and punctuation preserved), else Person 2's visual label. The
  normalised phrase the model saw is reported separately as `stems[].prompt`, so an editor
  that matches stems against the prompt it sent keeps working without sending a name.
- With the default `options.residual: "mixture_minus_targets"` the residual is
  `mixture - sum(targets)`, so **`sum(stems) == source`** exactly in float and within
  float32 rounding after encoding. That is what makes "mute the siren" equal "original minus
  siren" in the editor.
- `warnings[]` is never a reason to fail a job. It is where we say "the mock backend ran",
  "your `track_id` could not steer the model", "chunking was disabled", "the residual mode
  fell back".

### Model capabilities and supported prompting modes

| Mode | Status | What it needs |
| --- | --- | --- |
| **Text** | Implemented; the primary mode | A short lowercase noun/verb phrase, as SAM Audio was trained ("siren", "man speaking", "a dog barking"). We normalise for you: `"Isolate the Siren!"` → `siren`. |
| **Span anchors** | Implemented | `[{"token": "+"\|"-", "start", "end"}]`, seconds on the source timeline. `+` = target audible here, `-` = absent. The main mitigation for SAM Audio emitting unrelated audio when the prompted sound is not actually present. |
| **Span prediction** | Implemented, model-dependent | `options.predict_spans: true`. Needs the multi-billion-parameter span predictor loaded (`AUDIO_LOAD_SPAN_PREDICTOR=true`); costs memory and time. Upstream only predicts when no anchors were supplied. |
| **Visual (interval)** | Implemented end to end | Person 2's `interval` / `time_range` becomes `+` span anchors. **Works today, through Person 4, with no changes on anyone's side** — it is the only visual signal currently reaching the model. |
| **Visual (masks)** | Implemented but **never executed against the model** | The original video *plus* a per-frame binary mask of the selected object. See "What Person 1 Requires from Person 2". |
| Reranking | Implemented, model-dependent | `options.reranking_candidates` 1–16; generates N candidates and re-ranks them. Cost scales roughly linearly. |
| Stereo | Implemented | `duplicate` (one mono pass, copied to the source channel count) or `per_channel` (one pass per channel, 2× cost, keeps the stereo image). SAM Audio itself is mono. |
| Chunking | Implemented | `AUDIO_CHUNK_SECONDS` with `AUDIO_CHUNK_OVERLAP_SECONDS` linear crossfades; anchors are re-based per window. Default 20 s for the real model (upstream recommends 10–30 s), 0 (whole file) for the mock. Disabled automatically for visually prompted requests so frames stay aligned. |

What the model **cannot** do, so that nobody designs around it: bounding boxes, labels and
track ids are not model inputs; a single clicked-frame mask is not enough; object visibility
is not evidence that the object made a sound (Person 2 marks this `audio_association:
"unverified"` and they are right); FP16 degrades quality, so float32 is the supported dtype.

### Expected latency

**No measurement from this project exists** — no GPU, no checkpoint. The numbers below are
*user-reported* figures from upstream issues, collected so Person 4 can size timeouts, and
should be replaced with our own benchmark output (`audio-service/benchmarks/`) the day a GPU
is available.

| Setup | Reported | Implication |
| --- | --- | --- |
| RTX 4090, `sam-audio-large`, reranking on | ≈ 1 min per ~4 s of audio | A 60 s clip is far beyond the earlier 600 s backend timeout. Not viable without reranking off and chunking. |
| A100, `sam-audio-large`, 30 s clip | tens of seconds | A 60 s clip in 20–30 s windows may fit the configurable timeout. |
| `reranking_candidates = 8` | ≈ 8× the above | Treat as an opt-in quality mode, not a default. |
| Apple MPS | minutes per 10 s, memory hungry | Development only; never a demo path. |
| Mock backend (CPU) | ~10–20 ms per 3 s stereo clip, measured | What CI and the fixtures use. |

Multipliers that apply on top: one pass **per prompt**, ×2 for `stereo_mode: "per_channel"`,
×`reranking_candidates`, and chunked runs pay per window (plus overlap). Our wall-clock is
reported per request as `model.elapsed_seconds`, and requests are serialised by a lock
(one inference at a time per process), so a queue of jobs adds up linearly.

### GPU requirements

**Dependency pin found on the first real load:** `huggingface_hub<1` (and `transformers<5`); with 1.x/2.x the upstream loader raises `TypeError ... missing 'proxies' and 'resume_download'`. Both are pinned in the `sam` extra.

**Runtime platform (verified here):** the real model does not install on macOS/Apple Silicon unpatched:
`perception_models` needs `decord` (no macOS wheel) and imports `xformers` (no macOS wheel).
Upstream issue #29 confirms this and lists source patches; we do not patch third-party code
in this service. The Linux CUDA container (`audio-service/Dockerfile` target `sam`) is the
deployment target, which makes Person 4's GPU host the only place the model can run.


User-reported in upstream issue #24; not measured by us:

| Checkpoint | fp32 size | Load | Peak (15 s clip) |
| --- | --- | --- | --- |
| `facebook/sam-audio-small` | 5.1 GB | ≈ 21.6 GB | ≈ 28 GB |
| `facebook/sam-audio-base` | 7.7 GB | ≈ 24 GB | — |
| `facebook/sam-audio-large` | 14.9 GB | — | ≈ 38 GB; ~30 s is about the max on an 80 GB A100 |

Our recommended configuration per GPU class (also in `backends/sam_audio.py`):

- **24 GB** (3090/4090/A10): `AUDIO_MODEL_ID=facebook/sam-audio-small`,
  `AUDIO_LOAD_SPAN_PREDICTOR=false`, `AUDIO_CHUNK_SECONDS=10`–`15`,
  `AUDIO_RERANKING_CANDIDATES=1`.
- **40–48 GB** (A100 40 / L40S): `-large`, `AUDIO_CHUNK_SECONDS=15`–`20`, span predictor on,
  `AUDIO_RERANKING_CANDIDATES=1`–`2`.
- **80 GB** (A100/H100 80): `-large`, `AUDIO_CHUNK_SECONDS=30`,
  `AUDIO_RERANKING_CANDIDATES` up to 8.

Also required: Python ≥ 3.11 (upstream), CUDA (production target; the `sam` Docker target is
CUDA 12.4), float32, approved Hugging Face access to the gated repos, and ~15 GB of disk for
the `-large` checkpoint (mount a persistent HF cache — the compose overlay uses a
`sam-audio-models` volume). CPU "works" but is not a usable path.

### Error codes

`{"detail": "<message>", "error": {"code", "message", "retryable", "details"}}` — `detail`
keeps FastAPI's shape that Person 4 already understands; `error.code` is the stable thing to
branch on. Full table in [Contract E](contracts/audio_contracts.md#contract-e--processing-errors)
and live at `GET /capabilities`.

| Code | HTTP | Retryable |
| --- | --- | --- |
| `INVALID_REQUEST` | 400 | no |
| `INVALID_MEDIA` | 422 | no |
| `UNSUPPORTED_OPERATION` | 400 | no |
| `MISSING_ASSET` | 404 | no |
| `MISSING_MODEL` | 503 | **yes** |
| `INSUFFICIENT_GPU_MEMORY` | 503 | **yes** |
| `INFERENCE_FAILURE` | 500 | no |
| `STORAGE_FAILURE` | 500 | **yes** |
| `SERVICE_BUSY` | 503 | **yes** (reserved; not raised today) |
| `INTERNAL_ERROR` | 500 | no |

### Example requests

**`POST /separate` — exactly what Person 4's worker sends** (`app/media.py`:
`httpx.post(url, files={'audio': ('audio.wav', f, 'audio/wav')}, data={'metadata': json.dumps(payload)})`):

```bash
curl -sS -X POST "$SAM_AUDIO_SERVICE_URL" \
  -F "audio=@storage/projects/$PROJECT_ID/extracted/audio.wav;type=audio/wav" \
  -F 'metadata={"prompts":["dog"],"visual_prompt":{"track_id":"42"}}'
```

With the fields we propose Person 4 start forwarding (still one multipart POST, still
backward compatible):

```bash
curl -sS -X POST "$SAM_AUDIO_SERVICE_URL" \
  -F "audio=@storage/projects/$PROJECT_ID/extracted/audio.wav;type=audio/wav" \
  -F 'metadata={
        "prompts":[{"text":"siren","name":"Police siren",
                    "anchors":[{"token":"+","start":2.35,"end":5.8}]},
                   "man speaking"],
        "source":{"asset_id":"'"$PROJECT_ID"'",
                  "original_ref":"file:///data/projects/'"$PROJECT_ID"'/original/4b1c.mp4",
                  "sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                  "frame_rate":29.97},
        "options":{"predict_spans":true,"residual":"mixture_minus_targets"},
        "request_id":"'"$JOB_ID"'"}'
```

**`POST /v1/separate` — JSON with a media reference** (proposed; needs the storage volume
mounted into our container):

```bash
curl -sS -X POST http://audio-service:8001/v1/separate \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_ref": "file:///data/projects/7f3c9d2e/extracted/audio.wav",
    "prompts": [{"text": "Isolate the siren!", "name": "Police siren"}],
    "source": {
      "asset_id": "7f3c9d2e",
      "original_ref": "file:///data/projects/7f3c9d2e/original/4b1c.mp4",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "frame_rate": 29.97
    },
    "visual_prompt": {
      "selection": {"kind": "visual_selection", "track_id": 42, "detection_id": "frame:71:det:0",
                    "analysis_id": "run-abc", "source_sha256": "e3b0c442...",
                    "interval": {"start_seconds": 2.35, "end_seconds": 5.8}},
      "detections_ref": "file:///data/projects/7f3c9d2e/visual/detections.json"
    },
    "options": {"predict_spans": true, "return_mode": "file"},
    "request_id": "job-0a1b2c3d"
  }'
```

### Example output

Real response from this service (mock backend, 3 s stereo source, `prompts: ["siren"]`,
`visual_prompt: {"track_id": "42"}` — i.e. Person 4's current payload), base64 trimmed:

```json
{
  "stems": [
    {
      "name": "siren",
      "audio_base64": "UklGRiTKCABXQVZFZm10IBAAAAABAAIAgLsAAADuAgAEABAA…<trimmed>",
      "stem_id": "330d4fe2c3ce4856ac2f2a06c2d6a9da",
      "role": "target", "prompt": "siren", "label": "siren",
      "start_time": 0.0, "duration": 3.0,
      "sample_rate": 48000, "channels": 2, "sample_count": 144000,
      "format": "wav/pcm_f32le"
    },
    {
      "name": "residual",
      "audio_base64": "UklGRiTKCABXQVZFZm10IBAAAAABAAIAgLsAAADuAgAEABAA…<trimmed>",
      "stem_id": "d1626662d13d4632b4e09b6233ee6ec5",
      "role": "residual", "label": "everything else",
      "start_time": 0.0, "duration": 3.0,
      "sample_rate": 48000, "channels": 2, "sample_count": 144000,
      "format": "wav/pcm_f32le"
    }
  ],
  "model": {
    "backend": "mock", "model_id": "mock-bandpass", "device": "cpu", "dtype": "float32",
    "predict_spans": false, "reranking_candidates": 1, "stereo_mode": "duplicate",
    "chunked": false, "elapsed_seconds": 0.011, "is_mock": true
  },
  "source": {"duration": 3.0, "sample_rate": 48000, "channels": 2, "sample_count": 144000},
  "request_id": "30fd7dd4840847248053a3a8490135c7",
  "warnings": [
    "MOCK backend: stems are band-pass filtered copies of the mixture, not real separation",
    "visual_prompt has no video_ref+mask_video_ref pair; fields track_id cannot steer SAM Audio and were ignored"
  ],
  "contract_version": "2025-11-draft-1"
}
```

Note both warnings: this is mock output, and `{"track_id": "42"}` alone could not steer the
model. An error response, for comparison:

```json
{"detail":"at least one prompt is required",
 "error":{"code":"INVALID_REQUEST","message":"at least one prompt is required",
          "retryable":false,"details":{}}}
```

A longer, real-model-shaped example (12 s clip, two targets, spans, `is_mock: false`) is
committed at `audio-service/fixtures/editor/example_response.json` — illustrative numbers,
truncated base64, validated against our pydantic models by the test suite.

---

## What Person 1 Requires from Person 2

Reference: **CV integration contract 1.0** on `origin/vision/validated-pipeline`
(`vision/docs/CONTRACT.md`, `vision/docs/SAM.md`, `vision/schemas/selection.schema.json`,
`vision/schemas/detections.schema.json`). We have read it, we adopt its field names, and we
agree with its two central findings: `mask_videos` is the real visual input, and a bbox JSON
object is not. We also agree `audio_association: "unverified"` is the correct claim.

Our translation layer is `sam_audio_service/visual_prompts.py`; the shapes it accepts are
Contract D.

### What we consume today (implemented, unit-tested against synthetic documents)

| Your field | What we do with it |
| --- | --- |
| `interval.start_seconds` / `end_seconds` (selection 1.0) | `+` span anchor, **and** the window the model is run over. `end_seconds: null` → end of source. This is the one visual signal that already works end to end through Person 4 with no changes. |
| `frames[].objects[].mask` (detections 1.0 RLE) | Decoded with your codec (row-major, alternating zero/one runs starting with zeros, `foreground_value: 1`, counts summing to `h*w`) into one `H×W` mask per native frame index. |
| `frames[].frame_index`, `timestamp_seconds`, `valid_until_seconds` | Mask→frame alignment; and when no `interval` is given, the implied interval `[min timestamp_seconds, max valid_until_seconds)`. |
| `track_id`, `detection_id` (+ our `selected_track_id` / `selected_detection_id`, or legacy `object_id: "track:<n>"`) | Selects whose masks to use. We follow your rule: `track_id` when present, else `detection_id`; a frame with no observation of the selected object gets an **all-zero** mask and we never reuse the previous one. |
| `analysis_id` | Selection and detections must share it, or we drop the visual prompt with a warning. We never bridge identity changes. |
| `source_sha256` | Compared against the original media digest Person 4 gives us. Mismatch → visual prompt dropped with a warning. |
| `width` / `height`, `fps` | Mask geometry and frame rate. |
| `label` / `class_name` | Becomes `stems[].label` so the editor can name the stem after the clicked object. |

### What we cannot use

- **`bbox_xyxy` / `bbox_normalized` alone.** Not a model input. We report them in
  `warnings` and fall back to text + anchors. Your `allow_box_masks` rectangle is a coarse
  experiment; we are happy to A/B it once real inference works, but it is not a contract.
- **A selection on its own.** Selection 1.0 carries one mask (the clicked frame). SAM Audio
  needs a mask for **every** frame in the interval. We use the selection's interval as span
  anchors and say so in `warnings`.
- **Sampled (non-native-FPS) detections.** The processor maps frames uniformly onto audio
  features, so a gap in `frames` time-warps the conditioning. We need an observation for
  every native frame in the interval — your `--sample-fps >= native FPS` guidance, exactly.
- **`confidence`, `class_id`, `clicked_at_seconds`, `detector`, `runtime`, `annotated_media`**:
  accepted, ignored. We never read `annotated_media` (annotated previews must not reach the
  model).

### What we ask for

1. **Per-frame instance masks at native FPS** for the selected object over the chosen
   interval, in the detections document (`frames[].objects[].mask`), **or** the generic
   alternative: a losslessly encoded mask video aligned frame-for-frame with the source
   (pixels > 127 = object, all-zero where absent). Either is fine; the RLE path needs no new
   tooling on your side.
2. **A locator for the detections document**, not just the inline object: a
   `detections_ref` (a `file://` path under the shared storage root, or an allow-listed URL)
   alongside the selection. A full-resolution per-frame RLE document for a 1080p clip is
   large; pushing it through Person 4's `visual_prompt` field as inline JSON means it also
   lands in their SQLite `projects.visual` column and in the job payload.
3. **The selection and the detections together** — our preferred envelope:
   `{"selection": {...}, "detections_ref": "...", "video_ref": "..."}`. Sending the
   selection alone is valid and degrades to anchors.
4. **`source_sha256` on both documents** (you already do this) and agreement that it is the
   digest of the *original* uploaded bytes, so it can be compared with Person 4's.
5. **Native frame rate and frame size** (`fps`, `width`, `height` — you already emit these).
6. **Masks at source resolution**, not letterboxed or annotated (you already do this; we
   resize to the model's 336×336 ourselves).

### Verified against your real output

`audio-service/fixtures/vision/` holds `detections.json` + `selection.json` produced by your
pipeline (`yolo26n-seg`, ByteTrack, native 10 fps, your `download_demo.py` pedestrian clip,
commit `0f5654f`) and `tests/test_person2_artifacts.py` drives them through our translator,
frame/mask preparation and `/separate` (mock backend). Result: no field-name, polarity,
frame-alignment or sha256 mismatches; 49 masked frames for the selected track, interval
0–6 s, spans reported back as `[[0.0, 6.0]]`. Three things we noticed:

1. **A raw viewer click is too short.** `select_at_point` yields `interval = [timestamp,
   valid_until)`: one sample interval (0.1 s at 10 fps, 33 ms at 30 fps). We reject
   visual intervals under 100 ms, so the editor must extend the interval before sending
   it (your README documents this as optional; for us it is required).
2. **Inline per-frame RLE does not scale.** ~24 KB per 768×576 frame, 2.97 MB for 12 s.
   Person 4 forwards the whole object inline today. Use `detections_ref` (or a mask video)
   for anything beyond demo length (Open Question 1).
3. **Processed window.** We clamp the selection interval to your `processed_duration_seconds`
   (with a warning) so frames past the processed window are never fed as "object absent";
   your `prepare_visual_inputs` rejects such intervals outright. Either is fine; we chose
   degradation so a text prompt still runs.

### Two points to settle with you

- **Polarity.** We agree on the net effect — the object is kept, the background is zeroed —
  and we implement it the same way you do (`1 - foreground` into `mask_videos`, which keeps
  pixels where the mask is 0). What is unresolved is upstream: the official visual notebook
  passes object masks straight in, which zeroes the object
  (`facebookresearch/sam-audio#75`). We default to the paper reading
  (`AUDIO_VISUAL_MASK_MODE=keep_object`) and ship `notebook` as a one-env-var A/B so the
  first GPU run can settle it empirically. See Open Question 4.
- **Interval cropping and offsets.** Your contract flags that "submitting a cropped clip
  without offset reconstruction would misalign playback". We have closed that on our side:
  when a visual interval is given we run inference on that interval only, then place the
  result back on the full timeline — the target is zero outside the interval, the residual
  carries the untouched mixture there, and `start_time` stays `0.0` with
  `duration == source.duration`. **No offset contract is needed.** Please sanity-check that
  reading.

---

## What Person 1 Requires from Person 3

Person 3's code is not in the repository, so this section is mostly *what we deliver* plus
clearly flagged assumptions about what the editor needs. Everything marked **assumption** is
a guess we would like corrected. See `audio-service/fixtures/editor/README.md` for the
loading guide and `scripts/make_editor_fixture.py` for playable files.

In the shipped product Person 3 never calls this service. Person 4 stores our stems and
serves them from `GET /api/projects/{id}/stems/{stem_id}/audio` (bearer token, fetch as a
blob). What follows is what we guarantee about the audio behind that URL.

### Preferred audio output format

- **48 kHz, IEEE float32 RIFF WAV by default**, complete file with header. Every
  browser decodes it, it is seekable, and it is the format Person 4 already stores.
- Channel count **matches the source** (1 or 2), identical across all stems in a response.
- Over HTTP we return the WAV base64-encoded in `stems[].audio_base64`; with
  `options.return_mode: "file"` we return a `file://` `audio_ref` instead. The fixture
  substitutes a relative `audio_url` so a dev server can serve the files directly.
- **Assumption:** the editor wants plain `<audio>`/`AudioBuffer`-loadable files rather than
  a streaming/segmented format. If you need Opus/AAC, HLS, or range requests for long
  files, say so — that is a transcode in Person 4's layer, not a model change
  (Open Question 7).

### Required stem metadata

Per stem, from Contract C (Person 4 should pass these through):

| Field | Why the editor wants it |
| --- | --- |
| `name` | Track label. Person 4's webhook matches commands against it case-insensitively, so keep names distinct. |
| `role` (`target` / `residual`) | Style "everything else" differently and keep it last, without string-matching the name. |
| `label`, `prompt` | `label` is human-facing (object label when a visual prompt was used); `prompt` is the normalised text the model actually got — useful in a details panel. |
| `start_time` | Always `0.0`. Assert on it. |
| `duration`, `sample_count`, `sample_rate`, `channels` | Timeline length and the sync check. `sample_count` is authoritative; `duration` is a float. |
| `spans` | `[[start, end], …]` seconds where the model believes the target is active. A **UI hint** (where to draw blocks, where to auto-place keyframes), **not** a trim instruction — the audio outside a span is in the file and usually near-silent. |
| `stem_id` | Our correlation id for logs. Person 4 mints its own `stems.id`, which is the one your PATCH calls use. |
| `model.is_mock` (top level) | **`true` means the audio is not real separation.** Please show a "demo separation" badge whenever it is true, so mock output is never mistaken for a model result. |
| `warnings[]` (top level) | Explains degraded runs ("your `track_id` could not steer the model", "chunking disabled"). Worth surfacing somewhere unobtrusive. |

### Playback synchronization requirements

- All stems in one response describe the **same window of the source timeline**: same
  `sample_rate`, same `channels`, same `sample_count`, same `duration`, `start_time = 0.0`.
  Load every stem at offset 0 and start them together — **no per-stem offset, trim, padding
  or resampling is needed**.
- Sync everything to **one clock** (one `AudioContext`, or the `<video>` element's
  `currentTime`). Independent `<audio>` elements drift.
- Timeline origin is second `0.0` of the uploaded media, the same origin Person 4 uses for
  `start_offset`, automation keyframe `time`, and mute ranges, and the same one Person 2
  uses for detection timestamps. One number means one instant everywhere.
- Compare `sample_count`, not `duration`, when asserting alignment.
- If a response ever carries a non-zero `start_time`, treat it as a contract change we must
  agree first (Open Question 5) — assert rather than silently trust.

### Audio layer expectations

- `sum(stems) == source` (exactly in float; within float32 rounding after default WAV encoding),
  because `residual = mixture - sum(targets)` by default. So:
  **mute** a stem = original minus that stem; **solo** = that stem alone; **all stems at
  unity, unmuted** = the original, bit-for-bit modulo rounding.
- Therefore: **no normalisation or make-up gain in the default state**, or "nothing changed
  yet" will not match the original. A master limiter is reasonable but must be bypassed at
  the unity mix for that property to hold.
- Volume/automation multiply the stem before the sum. Person 4's range is `0–2` (`1` =
  unity, `2` = +6 dB), linear interpolation between keyframes, keyframe times strictly
  increasing, in project timeline seconds.
- Summing N stems each boosted above unity can clip. When prompts overlap in content (the
  mock backend especially), targets share energy.
- Edits are **non-destructive**: Person 4 stores them as stem rows and applies them at
  export; the WAV files never change. We assume the editor keeps the same model.
- **Assumption, not implemented:** waveform peak data, a per-stem loudness figure, a colour
  or ordering hint, and a "confidence" number. We can add any of these if you want them —
  peaks and loudness are cheap for us to compute (we already have the arrays and an
  `rms_dbfs`/`peak_dbfs` helper); a confidence score is not something SAM Audio gives us.
  See Open Question 6.

---

## What Person 1 Requires from Person 4

Person 4 already documented our contract in `INTEGRATION.md` and implemented it, and Test 3
passes against their real code. Everything below is **additive and backward compatible**:
if none of it happens, text and interval prompting keep working exactly as today.

### Media storage interface

1. **Mount the storage volume into the audio container, read-only, and set
   `AUDIO_MEDIA_ROOT`.** Already done in our overlay
   (`audio-service/compose.audio.yaml`: `./storage:/data:ro`, `AUDIO_MEDIA_ROOT=/data`), so
   this is really a request to keep it that way and to keep the layout
   `STORAGE_ROOT/projects/<project_id>/{original,extracted,stems,exports}` stable. Without
   it, mask-based visual prompting is impossible — we cannot reach the original video, and
   neither of the alternatives (pushing video bytes through the multipart body, or a signed
   download endpoint) exists.
2. **Keep the extracted audio as it is**: `extracted/audio.wav`, 48 kHz stereo PCM16, which
   is already the ideal input and takes our fast decode path.
3. **If you prefer a URL to a volume:** expose a token-authenticated
   `GET /api/projects/{id}/original` and add its host to `AUDIO_ALLOWED_MEDIA_HOSTS`. We
   support allow-listed `http(s)` references today, but we do not send an `Authorization`
   header — that would need agreeing (Open Question 2).

### Asset identifier format

- Send `metadata.source.asset_id` (and/or the legacy top-level `asset_id`). Your
  `project_id` is a perfectly good value; we treat it as opaque and only echo it back in
  `source.asset_id` and `stems[].source_asset_id`.
- Send `metadata.request_id` = your `job_id`. It is the correlation id between your job
  table and our logs, and we echo it in the response.
- Send `metadata.source.sha256`: the SHA-256 of the **original uploaded bytes**. This is the
  only way we can confirm Person 2's masks were computed on the same media. You already
  compute digests elsewhere (`core.digest` for tokens), so this is one `hashlib` pass during
  upload. Person 2's `source_sha256` is the same value.
- When you pass `--video-id` to Person 2's pipeline, pass your `project_id`, so
  `asset_id == project_id == video_id` and all three documents agree.

### Job execution contract

1. **Forward the whole `/process` body, not two fields.** `Process` currently narrows to
   `prompts: list[str]` and one `visual_prompt: dict`, which drops `prompts` as objects
   (so no per-stem display names and no per-prompt anchors), `options` (so no
   `predict_spans`, `reranking_candidates`, `stereo_mode`, `residual`, `chunk_seconds`) and
   `source`. Minimal change that unblocks all of it:

   ```python
   from pydantic import ConfigDict          # alongside your existing BaseModel/Field imports

   class Process(BaseModel):
       model_config = ConfigDict(extra='allow')          # forward unknown keys unchanged
       prompts: list[str | dict] = Field(default_factory=list, max_length=20)
       visual_prompt: dict | None = None
       options: dict | None = None
       source: dict | None = None
   ```

   We validate everything ourselves and return `INVALID_REQUEST` with per-field detail, so
   passing objects through costs you no validation code. Unknown keys are safe in both
   directions.
2. **Add `source` to the metadata you post**, built from what you already have at upload
   time:

   ```python
   payload['source'] = {
       'asset_id': pid,
       'original_ref': Path(p['original']).resolve().as_uri(),   # file:///data/projects/<pid>/original/<uid>.mp4
       'sha256': sha256_of(p['original']),
       'duration': info['duration'],
       'frame_rate': info.get('frame_rate'),
   }
   payload.setdefault('request_id', job['id'])
   ```

   `original_ref` must be the path **as the audio container sees it** (`/data/...`), which
   is what it already is when `STORAGE_ROOT=/data`.
3. **Read our error body before raising.** Done on `integration/cloud-editor-gpu`: `app/media.py`
   parses `error.code` / `error.message`, so job errors read e.g.
   `INVALID_REQUEST: at most 8 prompts per request` (verified live). `app/media.py` builds
   that string as `f"{code}: {message}"`, so `retryable` and `details` are not persisted at
   all — not part of the string either. `MISSING_MODEL` / `INSUFFICIENT_GPU_MEMORY` /
   `STORAGE_FAILURE` / `SERVICE_BUSY` carry `retryable: true`, so a structured job-error field
   would let the editor offer a retry.
4. **Store our `warnings[]` with the job** (a `warnings` column, or appended to
   `jobs.output`) and pass `role`, `label`, `prompt`, `spans` and `model.is_mock` through to
   the stem/job response, so Person 3 can tell a mock run from a real one and a text-only
   run from a mask-guided one.
5. **Timeout**: the configurable 3600 s default may not be enough for
   `-large` + reranking on a single GPU for anything over ~30 s of audio. Either keep
   `reranking_candidates: 1` and `chunk_seconds: 20`, or raise
   `INFERENCE_TIMEOUT_SECONDS`. Per-prompt, per-channel and per-candidate costs multiply.
6. **Keep one in-flight separation per project** (you already enforce this with the
   `one_active` index). We serialise inference internally, so parallel requests queue rather
   than fail, but a single GPU cannot overlap them.
7. **Health gating**: use `GET /ready` (our compose overlay already waits on
   `service_healthy`). `degraded` means the model has not loaded; a request then returns
   `MISSING_MODEL` with `retryable: true` rather than hanging.

### GPU deployment environment

- `audio-service/Dockerfile` has two targets: `mock` (CPU, no weights, for CI and local
  integration) and `sam` (CUDA 12.4 runtime, SAM Audio installed from GitHub). Bring-up:
  `docker compose -f compose.yaml -f audio-service/compose.audio.yaml up --build`, and
  `AUDIO_TARGET=sam HF_TOKEN=hf_... AUDIO_BACKEND=sam ...` for the GPU variant.
- The GPU host needs: an NVIDIA GPU sized per the table above (≥ 24 GB for `-small`,
  40–80 GB for `-large`), the NVIDIA container toolkit, a persistent Hugging Face cache
  volume (~15 GB for `-large`; the overlay provides `sam-audio-models:/models`), and a
  `HF_TOKEN` for an account **approved** for the gated repos. Tokens are passed at runtime
  and never baked into an image.
- Our container needs no write access to your storage; `:ro` is correct. Stems come back in
  the response body, and `return_mode: "file"` writes to our own `AUDIO_OUTPUT_ROOT`.
- The GPU service and the application backend will usually be on different hosts. That makes
  Open Question 2 (volume vs. authenticated URL) a real decision, not a formality.

### Output storage conventions

- Stems stay yours: you decode `audio_base64` into
  `STORAGE_ROOT/projects/<pid>/stems/<uid>.wav` exactly as you do now. We do not write
  there and do not want to.
- `MAX_STEM_BYTES` (500 MiB) is comfortable: a 48 kHz stereo float32 stem is ~23 MB/minute,
  so even a 10-minute input is ~115 MB per stem. The **response body** is the tighter
  constraint — N+1 stems, base64 (+33%) — e.g. 3 stems of a 10-minute clip is ~460 MB in one
  JSON document. That is the motivation for `options.return_mode: "file"` or `/v1/separate`
  once the volume is shared (Open Question 7).
- If you adopt `return_mode: "file"`, we write `AUDIO_OUTPUT_ROOT/<request_id>/<stem_id>.wav`
  and return `file://` refs; you then move or link them into your layout, and own cleanup.
  We never write inside `projects/`.
- Please keep storing your own probed `duration` per stem (you do) — it is the independent
  check that our timing claim holds.

### API routing expectations

- `SAM_AUDIO_SERVICE_URL` points at the **full path** `http://audio-service:8001/separate`
  (your default already does). We do not depend on being mounted at the root; if you front
  us with a reverse proxy, preserve the path and raise the proxy's body-size and read
  timeouts to match `MAX_UPLOAD_BYTES` and `INFERENCE_TIMEOUT_SECONDS`.
- Port 8001 (`AUDIO_PORT`), container name `audio-service` on your compose network.
- All our env vars are prefixed `AUDIO_` so a shared `.env` cannot collide with yours;
  `audio-service/.env.example` lists every one.
- `/v1/separate` is additive. If you adopt it, `SAM_AUDIO_SERVICE_URL` still points at
  `/separate` and the new path is opt-in per request.
- No authentication between the two services; keep them on a trusted network, as your
  `INTEGRATION.md` already states.

---

## Open Integration Questions

All **awaiting agreement**. Each has our proposed default so nothing is blocked on a
decision — the defaults are what the code does today.

**1. How do per-frame masks reach the audio service?**
*Owners: Person 1 + Person 2 + Person 4.*
Person 2 produces per-frame RLE masks in a detections document; Person 4 forwards only the
`visual_prompt` JSON object, which lands in SQLite and in the job payload. A full-resolution
per-frame document for a 1080p clip is large.
*Proposed default:* Person 2 writes `detections.json` into the project's storage directory
and the `visual_prompt` carries `{"selection": {...}, "detections_ref": "file://…"}`; we
read it through `AUDIO_MEDIA_ROOT`. Inline `detections` also works for short clips, and a
generic `mask_video_ref` is accepted as an alternative.
*Status: awaiting agreement.*

**2. Volume mount or authenticated URL for the original video?**
*Owners: Person 1 + Person 4.*
Mask-based visual prompting needs the original video bytes. Today nothing delivers them.
*Proposed default:* the shared read-only volume (`./storage:/data:ro`,
`AUDIO_MEDIA_ROOT=/data`) plus `metadata.source.original_ref`, which is what our compose
overlay already does. The alternative, needed if the GPU lives elsewhere, is a
token-authenticated `GET /api/projects/{id}/original` on an allow-listed host — we support
allow-listed URLs but send no `Authorization` header, so that needs a small extension.
*Status: awaiting agreement.*

**3. Does `error.code` reach the user?**
*Owner: Person 4.*
Resolved for code + message on `integration/cloud-editor-gpu` (`app/media.py` parses our error
body; verified live: `INVALID_REQUEST: at most 8 prompts per request`). Remaining: `retryable`
and `details` are not persisted at all — the job error is `f"{code}: {message}"`, so neither
field reaches the string.
*Proposed default:* persist `{code, message, retryable, details}` as a structured job-error
field alongside the string.
*Status: partially resolved; structured field awaiting agreement.*

**4. Visual mask polarity.**
*Owners: Person 1 + Person 2 (upstream: `facebookresearch/sam-audio#75`).*
`mask_videos` keeps pixels where the mask is `0`. The paper describes the model seeing the
object with the background zeroed; the official visual notebook passes object masks straight
in, which zeroes the object. Unresolved upstream.
*Proposed default:* `AUDIO_VISUAL_MASK_MODE=keep_object` — pass `1 - object_mask` so the
object survives (the paper reading, the only one under which the ImageBind re-ranker makes
sense, and the same net result as Person 2's `build_processor_batch`). `notebook` ships as a
one-env-var A/B; `scripts/smoke_sam.py --mask-mode both` runs both on one clip. **Cannot be
settled without checkpoint access.**
*Status: awaiting agreement (blocked on GPU + Hugging Face approval).*

**5. Full-timeline stems, or clips with an offset?**
*Owners: Person 1 + Person 3 + Person 4.*
Person 2's contract raises the risk of misalignment if a cropped clip is returned without
offset reconstruction.
*Proposed default:* stems are **always full-length with `start_time = 0.0`**, even for
visually prompted requests (inference runs on the interval; the result is placed back on the
timeline, target zero outside it, residual untouched). This matches Person 4's
`start_offset` default of 0 and needs no offset contract. If anyone needs clipped stems, it
becomes an explicit `options` flag.
*Status: awaiting agreement.*

**6. What extra per-stem metadata does the editor want?**
*Owners: Person 1 + Person 3.*
We propose `role`, `label`, `prompt`, `spans`, `sample_count` and `model.is_mock` in
addition to Person 4's existing stem fields. Not implemented: waveform peaks, per-stem
loudness, colour/ordering hints, a confidence score.
*Proposed default:* ship what is listed above; add peaks and loudness on request (cheap for
us); no confidence score, because SAM Audio does not give us a calibrated one.
*Status: awaiting agreement.*

**7. How do large stem payloads travel?**
*Owners: Person 1 + Person 4 (+ Person 3 for playback format).*
Base64 in one JSON document is ~460 MB for 3 stems of a 10-minute 48 kHz stereo input.
*Proposed default:* keep `audio_base64` for the demo-length clips we actually use, and adopt
`options.return_mode: "file"` (or `/v1/separate`) once the storage volume is shared. If the
editor needs compressed or streamed audio for long files, that is a transcode in Person 4's
layer, not a model change.
*Status: awaiting agreement.*

**8. Which checkpoint, and who holds access?**
*Owners: Person 1, with Person 4 for deployment.*
`facebook/sam-audio-{small,base,large}` and the `-tv` visual variants are gated on Hugging
Face; the team account is now approved and `sam-audio-small` has run here (see top of document).
*Proposed default:* `facebook/sam-audio-large` where a 40–80 GB GPU exists, `-small` on
24 GB, float32 everywhere, span predictor off on 24 GB. The `-tv` variants need evaluating
once access lands — Person 2 also flags that the team must choose. One approved account's
`HF_TOKEN` must be available to the GPU host as a runtime secret.
*Status: awaiting agreement (blocked on Hugging Face approval).*

**9. Prompt vocabulary: who writes the text the model sees?**
*Owners: Person 1 + Person 3 (+ Person 2 for labels).*
SAM Audio expects short lowercase noun/verb phrases. A user typing "Please remove the
annoying police siren!!" is not that, and when the prompted sound is absent the model may
emit unrelated audio.
*Proposed default:* we normalise defensively (lowercase, strip trailing punctuation and
leading "isolate the"/"extract the"/"the") for `stems[].prompt`, and default `stems[].name` to
the user's own wording, so the UI keeps the user's phrasing while the model gets the clean
phrase; `prompts[].name` overrides that name. Person 2's `label` is used when a
visual prompt has no text. A curated suggestion list in the editor would help more than any
parsing we can do.
*Status: awaiting agreement.*

**10. Latency budget and user-facing progress.**
*Owners: Person 1 + Person 4 + Person 3.*
Real latency is unmeasured (no GPU). Reported figures range from ~1 min per 4 s of audio on
a 4090 with reranking to tens of seconds for a 30 s clip on an A100.
*Proposed default:* `reranking_candidates: 1` and `chunk_seconds: 20` as the shipping
configuration, `INFERENCE_TIMEOUT_SECONDS=3600`, and a demo clip under ~30 s. We report
`model.elapsed_seconds` per request; we do **not** stream progress, so Person 4's `progress`
field cannot be finer-grained than queued/processing/completed from our side. Per-chunk
progress callbacks are possible if wanted.
*Status: awaiting agreement.*
