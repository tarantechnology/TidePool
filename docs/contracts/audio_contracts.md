# SoundCanvas audio contracts (Person 1 proposal)

**Issue #4 implementation update (2026-11-07):** the project backend now supplies
trusted source media and stored detections for validated selections, preserves
service warnings/model identity and rejects misaligned visual stems. The service's
more permissive direct-call behavior below is unchanged. See [current integration
and real-model validation status](../../vision/docs/ISSUE4.md); genuine visual-guided
SAM Audio inference remains unverified.

**Status update (2026-11-11), Contracts A + D:** the merged backend
(`integration/cloud-editor-gpu`) is now verified to send `metadata.source`
(`asset_id`, `original_ref`, `sha256`, `duration`, `frame_rate`) and
`metadata.visual_prompt = {"selection": ..., "detections": ...}` — the **envelope** shape
(D.3) — for every visually guided job, and that payload translates into a real
`VisualInput` with per-frame masks end to end on Person 2's real artefacts
(`audio-service/tests/test_merged_visual_workflow.py`). The exact payload, the
`AUDIO_MEDIA_ROOT` requirement and the remaining gaps are documented in
[Merged workflow (PR #15/#18)](../audio_integration_handoff.md#merged-workflow-pr-1518).
Still unverified: real visually guided inference, which needs a CUDA GPU.

**Status: PROPOSED, awaiting team agreement.** Every contract below is Person 1's proposal,
not an agreed interface. Nothing here requires another team to change code today: the audio
service already accepts exactly what Person 4's worker sends and returns a superset of what
it reads, and it accepts Person 2's documents unchanged as free-form JSON. The extra fields
are opt-in, and the names are negotiable — where Person 2 or Person 4 already ship a name
for a thing, that name is adopted here instead of inventing a new one.

Implementation of record: `audio-service/sam_audio_service/contracts.py` (pydantic models),
`errors.py` (Contract E), `visual_prompts.py` + `visual.py` (Contract D translation). Where
this document and the code disagree, the code is right and this document is a bug.

Contract version string returned by the service: `contract_version: "2025-11-draft-1"`
(`sam_audio_service/__init__.py`).

- [Timestamp and identity conventions](#timestamp-and-identity-conventions)
- [Contract A — Media Asset](#contract-a--media-asset)
- [Contract B — Audio Separation Request](#contract-b--audio-separation-request)
- [Contract C — Audio Separation Result](#contract-c--audio-separation-result)
- [Contract D — Visual Prompt](#contract-d--visual-prompt)
- [Contract E — Processing Errors](#contract-e--processing-errors)

---

## Timestamp and identity conventions

Proposed, and already implemented on our side. These are the rules that make four
independently written timelines line up.

| Rule | Value |
| --- | --- |
| Unit | **seconds**, JSON `number` (float). Never milliseconds, never frames, never samples. |
| Origin | `0.0` = first sample / first decoded frame of the **original uploaded media**. Person 2 calls this "seconds from first decoded frame"; Person 4 calls it "project timeline seconds". Same instant. |
| Interval form | half-open `[start, end)`. Person 2's `interval.start_seconds` / `end_seconds` and `valid_until_seconds` already work this way. |
| Stems | **full-length**: `start_time == 0.0` and `duration == source.duration` for every stem in a response, including the residual. We never return a clip of the event. |
| Visual-prompt intervals | narrow where the **model** looks, not what we return. Audio outside the interval is passed through unchanged in the residual and zero in the target. |
| Audio sample index | `round(t * sample_rate)`; `sample_rate` is always `48000` out of this service. `sample_count` is authoritative for length, `duration` is derived (`sample_count / sample_rate`). |
| Video frame index | `frame_index = round(t * fps)`, zero-based, for constant-frame-rate media — the same mapping Person 2 documents (`timestamp_seconds = frame_index / fps`). Variable frame rate is out of scope until someone needs it. |
| Rounding | we emit `duration`/`start_time` rounded to 6 decimals and model/merged `spans` to 3 (millisecond precision). Compare `sample_count`, not floats, when asserting alignment. |
| Identity | `asset_id`/`video_id` identifies the media; `sha256` of the **original bytes** proves two teams are talking about the same file (Person 2: `source_sha256`). Filenames are not identity. |
| Analysis identity | a CV run is `(video_id, analysis_id)`; `track_id`/`detection_id` are only meaningful inside one analysis. We never bridge identities across analyses. |

---

## Contract A — Media Asset

**Status: proposed, awaiting team agreement.** The shared vocabulary for "which media file
are we all talking about, and where is it". There is no asset *service*: Person 4 owns
storage, and this contract is only the identity + locator block that travels in requests.

```json
{
  "asset_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
  "project_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
  "video_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
  "original_ref": "file:///data/projects/7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f/original/4b1c....mp4",
  "audio_ref": "file:///data/projects/7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f/extracted/audio.wav",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "duration": 12.0,
  "sample_rate": 48000,
  "channels": 2,
  "frame_rate": 29.97,
  "width": 1920,
  "height": 1080,
  "has_video": true
}
```

| Field | Type | Status | Notes |
| --- | --- | --- | --- |
| `asset_id` | string | **consumed today** | Opaque. Echoed back as `source.asset_id` and `stems[].source_asset_id`; we never parse it. Person 4's `project_id` is a fine value; so is Person 2's `video_id`. |
| `project_id` | string | Person 4's existing name | Person 4's identifier for the project. Equal to `asset_id` for the one-media-per-project model Person 4 implements (`POST /api/projects/{id}/upload` returns 409 if media already exists). |
| `video_id` | string | Person 2's existing name | Person 2's media identity, default `sha256:<digest>`. Same media, same string, as long as Person 4 passes `--video-id`. |
| `original_ref` | string (`file://` or `http(s)://`) | **consumed today** as `metadata.source.original_ref` | The **original video**, required for mask-based visual prompting. `file://` must resolve under `AUDIO_MEDIA_ROOT`; `http(s)` hosts must be in `AUDIO_ALLOWED_MEDIA_HOSTS`. Enforced in `media.py:resolve_ref`. |
| `audio_ref` | string | **consumed today** as the `audio_ref` body field of `POST /v1/separate` | The extracted 48 kHz WAV. Not needed for `POST /separate`, where the bytes are uploaded. |
| `sha256` | string, 64 lowercase hex | **consumed today** as `metadata.source.sha256` | SHA-256 of the **original media bytes**. Same semantics and same algorithm as Person 2's `source_sha256`; we compare the two and refuse the visual prompt on mismatch. |
| `duration`, `sample_rate`, `channels` | number, int, int | advisory | We measure these ourselves with ffprobe and report what we measured in `source`. A mismatch is worth logging, not failing. |
| `frame_rate` | number | **consumed today** as `metadata.source.frame_rate` | Native CFR frame rate of `original_ref`. Used to map mask frame indices to seconds when the detections document does not carry `fps`. |
| `width`, `height`, `has_video` | int, int, bool | advisory | Person 4 already returns these from `probe()` on upload; Person 2 carries `width`/`height` (aliases `frame_width`/`frame_height`). |

**Compatibility.** Person 4 stores media at
`STORAGE_ROOT/projects/<project_id>/original/<uid><ext>` and the extracted audio at
`.../extracted/audio.wav`, and returns `probe()` output as `metadata` from `/upload`.
**Implemented since PR #15:** `app/visual.py::project_source` forwards exactly
`{asset_id, original_ref, sha256, duration, frame_rate}` as `metadata.source` for visually
guided jobs (`audio_ref` is still not sent; `POST /separate` uploads the bytes instead), and
`app/worker.py` re-verifies the digest when the queued job starts. The `file://` refs resolve
only because `compose.audio.yaml` mounts the same `./storage` at the same absolute path
`/data` in both containers (`STORAGE_ROOT` == `AUDIO_MEDIA_ROOT`). Person 2's exports
deliberately contain no absolute local paths, so `original_ref` has to come from Person 4,
not from Person 2.

---

## Contract B — Audio Separation Request

**Status: proposed, awaiting team agreement.** Implemented by
`contracts.SeparationMetadata` / `SeparationRequest`. The model is `extra="allow"`: unknown
top-level keys are accepted and ignored, so Person 4 can forward whole objects without
coordination.

Minimal request — **exactly what Person 4's worker sends today**, unchanged:

```json
{"prompts": ["dog"], "visual_prompt": {"track_id": "42"}}
```

Full request:

```json
{
  "prompts": [
    "man speaking",
    {
      "text": "Isolate the siren!",
      "name": "Police siren",
      "anchors": [{"token": "+", "start": 2.35, "end": 5.8}, {"token": "-", "start": 0.0, "end": 1.0}],
      "visual_prompt": {"...": "Contract D"}
    }
  ],
  "source": {
    "asset_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
    "original_ref": "file:///data/projects/7f3c.../original/4b1c....mp4",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "frame_rate": 29.97
  },
  "visual_prompt": {"...": "Contract D — applies to every prompt that has none of its own"},
  "anchors": [{"token": "+", "start": 2.35, "end": 5.8}],
  "options": {
    "model_id": "facebook/sam-audio-large",
    "predict_spans": true,
    "reranking_candidates": 1,
    "residual": "mixture_minus_targets",
    "residual_name": "residual",
    "stereo_mode": "duplicate",
    "chunk_seconds": 20,
    "passes": 2,
    "seed": 7,
    "return_mode": "base64"
  },
  "asset_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
  "request_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
}
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompts` | array of string **or** `PromptSpec` | `[]` (then 400) | One stem per entry. Max `AUDIO_MAX_PROMPTS` (8); exceeding it is `INVALID_REQUEST`. Person 4's `Process.prompts: list[str]` is the string form. |
| `prompts[].text` | string ≤ 500 | `""` | Normalised to a lowercase noun/verb phrase, as SAM Audio was trained ("Isolate the Siren!" → `siren`): trailing punctuation and leading `isolate (the)` / `extract (the)` / `the` are stripped. Required unless the entry has a usable visual prompt. |
| `prompts[].name` | string ≤ 500 | the caller's wording of `text` | The **stem name** shown to the user; becomes `stems[].name`. Omit it and we default to your own wording of `text` with whitespace collapsed — case and punctuation preserved ("Isolate the Dog!" stays `Isolate the Dog!` while the model gets `dog`), so an editor that matches stems against the prompt it sent needs no name at all. Pass it to name the stem something else ("Police siren"). Both prompt forms behave identically; a prompt that normalises to nothing (punctuation only) stays unnamed and falls back to the visual label. |
| `prompts[].anchors` | array of `{token, start, end}` | `[]` | Span prompts. `token` is `"+"` (target audible in this span) or `"-"` (target absent). Seconds on the source timeline, clamped to the source duration. |
| `prompts[].visual_prompt` | object | `null` | Contract D, per prompt. Overrides the top-level one for that prompt. |
| `source` | object | `null` | Contract A subset: `asset_id`, `original_ref`, `sha256`, `duration`, `frame_rate`. **Required for mask-based visual prompting** — without `original_ref` we cannot reach the video and fall back to text + anchors with a warning. |
| `visual_prompt` | object (free form) | `null` | Contract D. Applied to every prompt that has no `visual_prompt` of its own. This is the slot Person 4 already fills with the project's stored visual metadata. |
| `anchors` | array | `[]` | Span anchors applied to every prompt that defines none itself. |
| `options.model_id` | string | the loaded model | Requesting a different model is `UNSUPPORTED_OPERATION` (one checkpoint per process; they are 5–15 GB). |
| `options.predict_spans` | bool | `AUDIO_PREDICT_SPANS` (false) | Let the model find the target's spans. Needs the span predictor loaded (`AUDIO_LOAD_SPAN_PREDICTOR`); it is multi-billion-parameter, so it costs memory and time. Upstream only runs prediction when no anchors were supplied, so explicit anchors win. |
| `options.reranking_candidates` | int 1–16 | `AUDIO_RERANKING_CANDIDATES` (1) | Generates N candidates and re-ranks. Cost scales roughly linearly: 8 ≈ 8×. |
| `options.residual` | `mixture_minus_targets` \| `model` \| `none` | `mixture_minus_targets` | See Contract C. `model` needs exactly one prompt and a backend residual, otherwise it falls back with a warning. |
| `options.residual_name` | string | `"residual"` | The residual stem's `name`. |
| `options.stereo_mode` | `duplicate` \| `per_channel` | `AUDIO_STEREO_MODE` (`duplicate`) | SAM Audio is mono. `duplicate` = one mono pass copied to the source channel count. `per_channel` = one pass per channel (2× cost, keeps the stereo image). |
| `options.chunk_seconds` | number ≥ 0 | `AUDIO_CHUNK_SECONDS` (0 for mock, 20 for the real model) | `0` = whole file in one pass. Overlapping windows with a linear crossfade otherwise; anchors are re-based per window. Disabled automatically for visually prompted requests so frames stay aligned. |
| `options.passes` | int 1–`AUDIO_MAX_PASSES` (3, at most 5) | `1` | Multi-pass separation for when the model only attenuates the sound. Pass 1 separates the mixture → t₁; pass k ≥ 2 runs the **same** prompt (text, anchors, visual prompt) on that prompt's residual `mixture − (t₁+…+tₖ₋₁)` → tₖ; the stem is `t₁+…+tₖ`. Several prompts each run their own chain from the mixture; the residual stem is still `mixture − all targets` (`residual: "model"` uses the last pass's model residual). Chunking, `stereo_mode`, anchors and visual intervals apply unchanged to every pass. **Early stop:** if pass k extracts < −30 dB of pass 1's energy it is discarded and the chain stops with the warning `"pass {k} extracted negligible energy; stopped after {k-1} pass(es)"`; a silent pass 1 (`"pass 1 extracted negligible energy; stopped after 1 pass(es)"`) or a silent residual also stops it. Must be a JSON integer; out of range → `INVALID_REQUEST`. Cost: up to n× per prompt; `elapsed_seconds` covers all passes. |
| `options.seed` | int ≥ 0 | `null` | Forwarded to the backend for reproducibility. |
| `options.return_mode` | `base64` \| `file` | `base64` | `base64` inlines the WAV in `stems[].audio_base64` (Person 4's contract today). `file` writes WAVs under `AUDIO_OUTPUT_ROOT` and returns `stems[].audio_ref` instead — proposed for large payloads. |
| `asset_id` | string | `null` | Echoed as `source.asset_id` / `stems[].source_asset_id`. Superseded by `source.asset_id`; both are accepted. |
| `request_id` | string ≤ 128 | generated | Echoed in the response; the correlation id for logs across services. Person 4's `job_id` is the natural value. |

**Transport.** Two endpoints, same metadata model:

- `POST /separate` — multipart: file part `audio` + form field `metadata` (the JSON above as
  a string). **Person 4's existing contract**, unchanged.
- `POST /v1/separate` — JSON body = the same object plus `audio_ref` (Contract A). Proposed;
  avoids pushing hundreds of MB through the request body when both services share a volume.

**Compatibility.** `{"prompts": [...], "visual_prompt": {...}}` — the only two keys Person
4's `Process` model forwards — is a valid Contract B request, and the only one exercised in
the executed integration test. Everything else is additive.

---

## Contract C — Audio Separation Result

**Status: proposed, awaiting team agreement.** Implemented by `contracts.SeparationResponse`.
HTTP 200. `null` fields are omitted from the wire response
(`response_model_exclude_none=True`), except `stems[].refinement`, which residual stems always carry as `null`. `GET /capabilities` reports `defaults.passes` (1) and `limits.max_passes` (`AUDIO_MAX_PASSES`).

```json
{
  "stems": [
    {
      "name": "Police siren",
      "audio_base64": "UklGRiRbIgBXQVZF...<whole WAV, base64>...AAA=",
      "stem_id": "9c1f2a7b4e2d4f0c8a6b3d5e7f901234",
      "role": "target",
      "prompt": "siren",
      "label": "Police siren",
      "start_time": 0.0,
      "duration": 12.0,
      "sample_rate": 48000,
      "channels": 2,
      "sample_count": 576000,
      "format": "wav/pcm_f32le",
      "spans": [[2.35, 5.8], [7.1, 10.95]],
      "source_asset_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
      "refinement": {"passes_requested": 2, "passes_run": 2, "per_pass_energy_db": [-8.412, -24.03], "stopped_early": false}
    },
    {
      "name": "residual",
      "audio_base64": "UklGRiRbIgBXQVZF...AAA=",
      "stem_id": "c4d7e0f3a6b94182d5e8f1a4b7c0d369",
      "role": "residual",
      "label": "everything else",
      "start_time": 0.0,
      "duration": 12.0,
      "sample_rate": 48000,
      "channels": 2,
      "sample_count": 576000,
      "format": "wav/pcm_f32le"
    }
  ],
  "model": {
    "backend": "sam",
    "model_id": "facebook/sam-audio-large",
    "device": "cuda:0",
    "dtype": "float32",
    "predict_spans": true,
    "reranking_candidates": 1,
    "stereo_mode": "duplicate",
    "chunked": false,
    "passes": 2,
    "elapsed_seconds": 34.812,
    "is_mock": false
  },
  "source": {
    "asset_id": "7f3c9d2e1a5b4c6d8e0f1a2b3c4d5e6f",
    "duration": 12.0,
    "sample_rate": 48000,
    "channels": 2,
    "sample_count": 576000
  },
  "request_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
  "warnings": [],
  "contract_version": "2025-11-draft-1"
}
```

A committed copy with truncated base64 lives at
`audio-service/fixtures/editor/example_response.json` and is validated by
`audio-service/tests/test_editor_fixture.py`.

| Field | Type | Status | Notes |
| --- | --- | --- | --- |
| `stems[].name` | string | **Person 4 reads this today** | `prompts[].name` if given, else the caller's wording of the prompt with whitespace collapsed (case and punctuation preserved), else Person 2's visual label; the residual uses `options.residual_name`. Person 4 stores it as `stems.name` and matches webhook commands against it case-insensitively, so the default keeps the caller's own phrasing matchable. The **normalised** phrase the model saw is `stems[].prompt`, not this. |
| `stems[].audio_base64` | string | **Person 4 reads this today** | A complete RIFF WAV (header included), base64, no data URI prefix. Present unless `return_mode: "file"`. Capped by Person 4's `MAX_STEM_BYTES` after decoding. |
| `stems[].audio_ref` | string | proposed | `file://` URL under `AUDIO_OUTPUT_ROOT`, present instead of `audio_base64` when `return_mode: "file"`. |
| `stems[].stem_id` | string (32 hex) | proposed | Stable **within one response**. It is not Person 4's `stems.id` (they mint their own) — it is the correlation id between our logs and a stem. |
| `stems[].role` | `"target"` \| `"residual"` | proposed | Exactly one `residual` unless `options.residual: "none"`. Lets the editor style "everything else" differently without string-matching the name. |
| `stems[].prompt` | string \| null | proposed | The **normalised** text actually given to the model. `null` for the residual. |
| `stems[].label` | string \| null | proposed | The human-facing name of the stem: Person 2's object label when the prompt carried its own visual prompt, otherwise the same value as `stems[].name` (so for a text prompt it is the caller's wording, not the normalised phrase); `"everything else"` for the residual. |
| `stems[].start_time` | number | proposed, maps to Person 4's `start_offset` | **Always `0.0`.** Stems are full-timeline. Person 4 defaults `stems.start_offset` to 0 and lets the editor change it; our value is the initial one. |
| `stems[].duration` | number | **Person 4 computes its own** | Equal to `source.duration` for every stem. Person 4 re-probes each WAV and stores its own `duration`; the two agree to within ffprobe's rounding (asserted < 50 ms in the executed test, in practice exact). |
| `stems[].sample_rate` | int | proposed | Always `48000`. |
| `stems[].channels` | int | proposed | Matches the source (1 or 2). Identical across stems. |
| `stems[].sample_count` | int | proposed | Frames per channel. **Authoritative length.** Identical across stems and equal to `source.sample_count`. |
| `stems[].format` | string | proposed | `"wav/pcm_f32le"` by default; optional PCM16 can clip. |
| `stems[].spans` | `[[start, end], ...]` \| null | proposed | Seconds on the source timeline where the target is believed active: predicted spans when `predict_spans`, otherwise the `+` anchors. A **UI hint**, not a trim instruction — audio outside a span is in the file and usually near-silent. |
| `stems[].source_asset_id` | string \| null | proposed | Echo of the request's asset id. |
| `stems[].refinement` | object \| null | proposed | Multi-pass bookkeeping (`options.passes`). Target stems: `{"passes_requested": n, "passes_run": k, "per_pass_energy_db": [e₁, …, eₖ], "stopped_early": bool}` where eᵢ = 10·log10(energy of tᵢ / energy of the mixture), floored at −120, one entry per pass **kept** (a pass discarded by early stop is not listed). Present on every target stem, also for `passes: 1`. Residual stems: always `null` — the one field emitted even though it is null. |
| `model.passes` | int | proposed | Most passes run for any prompt (≥ 1). |
| `model.*` | object | proposed | What actually ran: `backend` (`mock`\|`sam`), `model_id`, `device`, `dtype`, the effective `predict_spans` / `reranking_candidates` / `stereo_mode`, whether chunking kicked in, wall-clock `elapsed_seconds`, and `is_mock`. |
| `model.is_mock` | bool | proposed | **`true` means the stems are band-pass filtered copies of the mixture, not separation.** Any UI that can show mock output must surface this. |
| `source.*` | object | proposed | What we measured after decoding: `duration`, `sample_rate` (48000), `channels`, `sample_count`, `asset_id`. |
| `request_id` | string | proposed | Echo or generated. |
| `warnings` | array of string | proposed | Non-fatal: mock backend in use, visual fields that could not steer the model, chunking disabled, residual fallback. **Never** a reason to fail a job; worth storing next to the job so the user can be told "the siren was isolated from text only — masks were missing". |
| `contract_version` | string | proposed | `"2025-11-draft-1"`. Bumped on any breaking change here. |

### The residual, and why it matters to the editor

With the default `options.residual: "mixture_minus_targets"` the residual is literally
`mixture - sum(targets)`, computed in float before WAV encoding. Therefore:

```
sum(all stems) == source        # within float32 rounding after default WAV encoding
```

This is what makes the editor's semantics exact: muting a stem is the original minus that
stem, and the default all-stems-at-unity mix reproduces the original. Verified by
`tests/test_api.py::test_stems_reconstruct_mixture_exactly`,
`::test_chunking_preserves_length_and_reconstruction` and
`tests/test_editor_fixture.py::test_stems_sum_back_to_the_source`.

`options.residual: "model"` returns the model's own residual instead (single prompt only) —
then the stems do **not** sum to the mixture. `"none"` returns targets only.

**Compatibility.** `{"stems": [{"name", "audio_base64"}]}` is a subset of the above, so
Person 4's `app/media.py` works against this response unmodified — that is the executed
Test 3. Person 4's stem row (`id`, `name`, `path`, `duration`, `start_offset`, `volume`,
`muted`, `solo`, `automation`, `mute_ranges`; API hides `path` and adds `audio_url`) is the
editor's view; our `start_time` → their `start_offset`, our `duration` → their `duration`,
our `name` → their `name`, and `role`/`spans`/`label`/`prompt` are the fields we propose
they pass through so Person 3 can use them.

---

## Contract D — Visual Prompt

**Status: proposed, awaiting team agreement.** This is the contract with the most open
questions; see the handoff doc. Implemented by `visual_prompts.translate_visual_prompt`
(shape translation) and `visual.py` (frames, masks, polarity).

### What SAM Audio actually consumes

Verified against the upstream `sam_audio` sources, not inferred: visual prompting goes
through `processor.mask_videos(videos, masks)` → `processor(..., masked_videos=[...])`. For
the audio interval the model needs **every native video frame** *and* **a per-frame binary
mask** of the selected object. The processor maps the N frames uniformly onto the audio
feature frames, so the frames must cover exactly the audio interval at the source frame
rate. Frames are resized to 336×336 by the vision encoder.

Consequence: **a bounding box, a `track_id`, a label or a single clicked-frame mask cannot
steer the model.** They are accepted and reported in `warnings`, never rejected.

### Accepted shapes

All four are accepted in the `visual_prompt` slot (top level or per prompt); unknown fields
are ignored with a warning rather than failing the request.

**1. Person 2 selection 1.0** (`kind: "visual_selection"`) — yields **span anchors only**,
because it carries a mask for the clicked frame alone:

```json
{
  "schema_version": "1.0", "kind": "visual_selection",
  "video_id": "7f3c...", "source_sha256": "e3b0c442...",
  "analysis_id": "run-2026-11-06T12:00:00Z-abc", "detection_id": "frame:71:det:0",
  "track_id": 42, "object_id": "track:42", "label": "dog", "class_name": "dog",
  "frame_index": 71, "timestamp_seconds": 2.369, "clicked_at_seconds": 2.5,
  "interval": {"start_seconds": 2.35, "end_seconds": 5.8},
  "mask": {"encoding": "rle", "order": "C", "size": [1080, 1920], "counts": [12345, 67, "..."], "foreground_value": 1},
  "audio_association": "unverified"
}
```

**2. Person 2 detections 1.0** (`frames[].objects[].mask` RLE for every frame) plus a
selected object — this is the shape that **can** drive the model:

```json
{
  "schema_version": "1.0", "video_id": "7f3c...", "source_sha256": "e3b0c442...",
  "analysis_id": "run-...", "media_type": "video", "fps": 29.97,
  "width": 1920, "height": 1080, "processed_duration_seconds": 12.0,
  "selected_track_id": 42,
  "frames": [
    {"frame_index": 70, "timestamp_seconds": 2.336, "valid_until_seconds": 2.369,
     "objects": [{"track_id": 42, "detection_id": "frame:70:det:0", "label": "dog",
                  "mask": {"encoding": "rle", "order": "C", "size": [1080, 1920], "counts": ["..."], "foreground_value": 1}}]}
  ]
}
```

**3. Envelope** — the recommended shape, because it carries the selection, the per-frame
masks and the video locator together. **This is what Person 4's backend sends today**, as
`{"selection": ..., "detections": ...}`, with the video locator arriving separately in
`metadata.source.original_ref`; `detections_ref`, `video_ref` and `time_ranges` are
accepted but unused by the merged workflow:

```json
{
  "selection": {"...": "shape 1"},
  "detections": {"...": "shape 2"},
  "detections_ref": "file:///data/projects/7f3c.../visual/detections.json",
  "video_ref": "file:///data/projects/7f3c.../original/4b1c....mp4",
  "time_ranges": [{"start": 2.35, "end": 5.8}]
}
```

**4. Generic mask video** — no CV vocabulary, for anyone who can render a mask track:

```json
{
  "video_ref": "file:///data/.../original/clip.mp4",
  "mask_video_ref": "file:///data/.../masks/track_42.mkv",
  "time_range": {"start": 2.35, "end": 5.8},
  "label": "dog", "object_id": "track:42", "frame_rate": 29.97
}
```

| Field | Source | Status | How we use it |
| --- | --- | --- | --- |
| `interval.start_seconds` / `end_seconds` | Person 2 selection | **consumed** | `+` span anchor, and the window the model is run on. `end_seconds: null` → end of source. Works today with text prompts through Person 4 unchanged. |
| `time_range` / `time_ranges` `{start, end}` | ours | **consumed** | Additional `+` span anchors. |
| `frames[].objects[].mask` (RLE) | Person 2 detections | **consumed** | Decoded with Person 2's codec (row-major, alternating zero/one runs, `foreground_value: 1`, counts sum to `h*w`) into one mask per frame index. Frames with no observation of the selected object get an **all-zero** mask; we never reuse the previous mask. |
| `frames[].frame_index` / `timestamp_seconds` / `valid_until_seconds` | Person 2 | **consumed** | Mask → frame alignment and, when no `interval` is given, the implied interval `[min timestamp, max valid_until)`. |
| `track_id` / `detection_id` / `selected_track_id` / `selected_detection_id` / `object_id` (`track:<n>`) | Person 2 | **consumed as a selector** | Chooses which object's masks to use. Alone, without masks, it **cannot** steer the model. If exactly one track in the document has masks, we use it and say so in `warnings`. |
| `source_sha256` | Person 2 | **consumed** | Compared against `metadata.source.sha256`; a mismatch drops the visual prompt with a warning (masks computed on different bytes are worse than no masks). |
| `analysis_id` | Person 2 | **consumed** | Selection and detections must agree, or the visual prompt is dropped with a warning. |
| `width` / `height`, `fps` | Person 2 | **consumed** | Mask geometry and frame rate when `source.frame_rate` is absent. |
| `video_ref` / `original_ref` (or `metadata.source.original_ref`) | Person 4 | **consumed** | The original video. **Required for masks to do anything.** Must resolve under `AUDIO_MEDIA_ROOT` or an allow-listed host. |
| `mask_video_ref` | ours | **consumed** | Alternative to RLE: a losslessly encoded video aligned frame-for-frame with the source, pixels > 127 = object. |
| `label` / `class_name` | Person 2 | **consumed** | Becomes `stems[].label`; also used as the mock backend's prompt when there is no text. |
| `bbox_xyxy` / `bbox_normalized` | Person 2 | **accepted, cannot steer the model** | Reported in `warnings`. A box-derived rectangular mask is available in Person 2's own helper behind an explicit flag and is a coarse experiment, not a contract. |
| `confidence`, `class_id`, `clicked_at_seconds`, `audio_association`, `runtime`, `detector`, … | Person 2 | ignored | Accepted and ignored; `audio_association: "unverified"` is correct and we make no claim otherwise. |

### Behaviour we commit to

- **Degradation, never rejection.** Missing masks or missing video → the request still runs
  with text + span anchors, and `warnings` states exactly what was missing. The only hard
  failures are a visual interval shorter than 100 ms or outside the audio
  (`INVALID_REQUEST`) and an unreadable/out-of-root reference (`MISSING_ASSET` /
  `INVALID_REQUEST`).
- **Stems stay full-length.** When a visual interval is given, inference runs on that
  interval only; the target is zero outside it and the residual carries the untouched
  mixture there. `start_time` is still `0.0` and `duration` still equals the source. This
  removes the offset-reconstruction problem Person 2 flags in their contract.
- **Chunking off** for visually prompted requests, so frames and audio stay aligned.
- **Mask polarity.** `mask_videos` keeps pixels where the mask is `0`. We default to
  `AUDIO_VISUAL_MASK_MODE=keep_object`, which passes `1 - object_mask` so the object
  survives and the background is zeroed — the paper's `V ⊙ M`, and the same net result as
  Person 2's `build_processor_batch`. `AUDIO_VISUAL_MASK_MODE=notebook` passes the object
  mask unchanged, reproducing the official notebook. Upstream
  `facebookresearch/sam-audio#75` is unresolved; the switch exists so one clip can be run
  both ways and compared. **No such comparison has been run** (no GPU, no checkpoint access
  yet).

---

## Contract E — Processing Errors

**Status: proposed, awaiting team agreement.** Implemented by `errors.py`; the live table is
served at `GET /capabilities` under `error_codes`.

```json
{
  "detail": "audio exceeds AUDIO_MAX_INPUT_BYTES",
  "error": {
    "code": "INVALID_MEDIA",
    "message": "audio exceeds AUDIO_MAX_INPUT_BYTES",
    "retryable": false,
    "details": {"max_input_bytes": 524288000}
  }
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `detail` | string | The same message. Present so the shape stays FastAPI's `{"detail": ...}`, which Person 4's backend and INTEGRATION.md already assume. |
| `error.code` | string | Stable machine-readable code from the table below. The thing to branch on. |
| `error.message` | string | Human-readable, safe to show a user after review. Never contains tokens or absolute host paths beyond the reference the caller supplied. |
| `error.retryable` | bool | `true` = the same request may succeed later (model still loading, GPU busy/OOM, transient storage). `false` = the request itself must change. |
| `error.details` | object | Code-specific context: limits, the offending interval, pydantic validation errors, `model_id`. Free-form; do not parse blindly. |

| Code | HTTP | Retryable | Raised when |
| --- | --- | --- | --- |
| `INVALID_REQUEST` | 400 | no | Bad/absent prompts, unparseable `metadata` JSON, failed validation, a reference outside `AUDIO_MEDIA_ROOT`, a visual interval under 100 ms. |
| `INVALID_MEDIA` | 422 | no | Audio cannot be decoded, has no audio stream, is empty, exceeds `AUDIO_MAX_INPUT_BYTES`, or decoding timed out. |
| `UNSUPPORTED_OPERATION` | 400 | no | A different `options.model_id`, `predict_spans` on a backend without a span predictor, audio longer than `AUDIO_MAX_DURATION_SECONDS`. |
| `MISSING_ASSET` | 404 | no | A `file://`/`http(s)` reference resolves to nothing (deleted upload, wrong volume mount, 404 from the host). |
| `MISSING_MODEL` | 503 | **yes** | Checkpoint not downloaded, Hugging Face access not granted, `sam_audio` not installed, load failed. `/health` reports `degraded` with `model.load_error`. |
| `INSUFFICIENT_GPU_MEMORY` | 503 | **yes** | CUDA OOM on load or during inference. `details` carries the window length; retry with a shorter input, `chunk_seconds`, or fewer reranking candidates. |
| `INFERENCE_FAILURE` | 500 | no | The model raised. `message` is `TypeName: text`; the traceback is in our logs only. |
| `STORAGE_FAILURE` | 500 | **yes** | Could not write a stem (`return_mode: "file"`) or fetch an http(s) reference. |
| `SERVICE_BUSY` | 503 | **yes** | Reserved for admission control. Not raised today — one request at a time is serialised by a lock instead. |
| `INTERNAL_ERROR` | 500 | no | Anything unhandled. `message` carries the exception type only. |

**Compatibility.** Person 4's worker (`app/media.py` on `integration/cloud-editor-gpu`) parses
`error.code` and `error.message` and stores `"<CODE>: <message>"` as the job error (verified
live: `INVALID_REQUEST: at most 8 prompts per request`); `retryable` and `details` are not yet
persisted (Open Question 3 in the handoff doc). Retry policy: Person 4 has no automatic retry ("to
retry, submit a new processing job"), so `retryable` is advice for a human or a future
scheduler, not a promise that we will be retried.
