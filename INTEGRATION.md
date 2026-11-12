# Integration contracts

Base URL `http://localhost:8000`; OpenAPI and executable examples are at `/docs` and `/openapi.json`. Except project creation and local webhook, requests require `Authorization: Bearer <project token>`. A token grants full access to one project and must be sent for media downloads as well. Protect it like a password. FastAPI errors are `{"detail":"message"}`; validation errors return a detail array. IDs are opaque.

## Person 1: audio AI

Set `INFERENCE_ADAPTER=http` and `SAM_AUDIO_SERVICE_URL` to your endpoint. The worker POSTs multipart form data: `audio` is 48 kHz, stereo, float32 PCM WAV; `metadata` is a JSON string such as `{"prompts":["dog"],"visual_prompt":{"track_id":"42"}}`. Return HTTP 200 with JSON:

```json
{"stems":[{"name":"dog","audio_base64":"<base64-encoded WAV bytes>"}]}
```

Return at least one valid WAV containing audio. Float32 WAV preserves stem headroom and is supported by the backend; PCM16 may clip individual stems. In HTTP mode at least one nonempty text prompt is required before a job is queued. Duration is probed locally. The default HTTP timeout is 3600 seconds (`INFERENCE_TIMEOUT_SECONDS`), and each decoded stem is capped by `MAX_STEM_BYTES`. Non-2xx, timeout, malformed JSON, invalid base64 and invalid audio fail the job; its error appears at `GET /api/jobs/{id}`. When the service returns `{"error":{"code","message","retryable"}}`, the job keeps `"<code>: <message>"` in `error` and also reports `error_code` and `retryable` (null when the service said nothing structured). At most `MAX_PROMPTS` prompts (default 8, mirroring the service's `AUDIO_MAX_PROMPTS`) are accepted; over-limit requests get HTTP 422 before a job is queued. Per-stem `stem_id`, `role`, `prompt`, `label`, `sample_rate`, `channels`, `format`, `spans` and `refinement` are stored when present and returned to the editor. When a client asks for multi-pass separation the backend adds `options.passes` (1..`MAX_PASSES`) to `metadata`; it is omitted otherwise, so the service default of one pass applies. Each target's `refinement` block (`passes_requested`, `passes_run`, `per_pass_energy_db`, `stopped_early`) is persisted verbatim. To retry, submit a new processing job. Service responses are trusted and should come from a trusted network. In fake mode each prompt produces a copy of the original audio, for integration only.

## Person 2: vision

`PUT /api/projects/{id}/visual-metadata` stores detection metadata. A `/process` request only forwards a visual prompt when it carries a selection: without one the backend sends `visual_prompt: null` and the job runs text-only, so a stored detections document is never shipped to a request that cannot use it. Client media-reference fields (`video_ref`, `original_ref`, `mask_video_ref`, `detections_ref`, `audio_ref`) are rejected, including nested fields. The backend can also queue the existing CPU vision pipeline with `POST /api/projects/{id}/visual-analyze`; poll `GET /api/visual-jobs/{job_id}` and retrieve results with `GET /api/projects/{id}/visual-metadata`. Detection does not itself infer sound associations.

Object detection availability and jobs:

- `GET /api/projects/{id}/capabilities` includes `"vision": {"available", "reason", "model", "weights_cached", "device"}`. `available` means `soundcanvas_cv` and `ultralytics` are installed and import in the worker's interpreter (checked once per process and cached; a failed job that reports the package missing clears the cache). `reason` is an actionable sentence when unavailable, otherwise `null`. `weights_cached` says whether `VISION_MODEL` is already on disk; if not, the first analysis downloads it, which requires network access. The check never downloads.
- `POST .../visual-analyze` returns 503 `{"detail": <reason>, "code": "vision_unavailable"}` when detection is unavailable. Media that fails the CFR/zero-start/rotation checks (typical of phone recordings) is not rejected: the job first prepares a compatible copy exactly as `prepare-visual` does, then detects on it. The 202 response is `{"job_id", "status": "queued", "prepared": bool}`. It returns 409 if a copy is needed but the project already has stems, and 422 for problems re-encoding cannot fix (rotation that is not a multiple of 90°).
- `GET /api/visual-jobs/{id}` adds `stage` (`queued` → `preparing`, only when a copy is needed → `detecting` → `completed`) and `progress` (5 / 15 / 30 / 100; the detector prints no intermediate progress). A failed job keeps the stage it failed in, and `error` is the detector's last stderr line, with a missing package mapped to the capabilities `reason` and a weights download failure mapped to an offline hint. `result` adds `prepared`.
- The worker runs the detector with its working directory at `$STORAGE_ROOT/vision`, where ultralytics downloads weights, and with `YOLO_CONFIG_DIR=$STORAGE_ROOT/vision/config` unless that is already set, so weights and settings persist in the Docker volume. A relative `VISION_MODEL` that exists relative to the backend's working directory is passed as an absolute path.

For actual visual prompting, store the [CV schema-1.0 detections](vision/docs/CONTRACT.md) generated at native FPS with instance masks, then send `{"prompts":["dog barking"],"visual_prompt":<exported visual_selection>}`. The backend validates media/analysis identity, the observed selection, mask geometry/RLE, complete native-frame coverage and a 0.1–30 s frame-aligned interval. It snapshots `{"selection":...,"detections":...}` without changing those documents and adds server-owned `source` metadata (`original_ref`, SHA256, project asset ID, native FPS, duration). A stored envelope with both documents also works. Analyzed track absence yields empty foreground; missing frame records or masks fail validation. Bounding boxes alone cannot guide the model. CFR media must start audio/video at zero and have no rotation metadata; normalize before upload and analyze those same bytes.

Both services must see the same absolute storage path: the existing audio Compose overlay mounts `/data` read-only into the audio service with `AUDIO_MEDIA_ROOT=/data`. Local processes should set `AUDIO_MEDIA_ROOT` to the backend's absolute `STORAGE_ROOT`. No project token or client file path is forwarded. The worker rechecks uploaded bytes before inference and accepts only zero-offset, full-duration visual stems with matching rate/channels. Job responses add `inference` with model identity, warnings and service source/request metadata; `model.is_mock=true` means no real separation. See [Issue #4 validation and exact commands](vision/docs/ISSUE4.md). **Real visual-guided model isolation remains unverified; mock transport/export tests do not establish sound association.**

## Person 3: frontend

1. `POST /api/projects` returns `{"project_id":"...","access_token":"...","status":"created"}`. Save the token securely.
2. `POST /api/projects/{id}/upload` with multipart `file`; allowed extensions MP4/MOV/WAV/MP3, size defaults to 500 MiB. Response includes original media metadata and status. Upload extracts 48 kHz stereo WAV synchronously.
3. `POST /api/projects/{id}/process` with `{"prompts":["dog","music"]}` returns `{"job_id":"...","status":"queued"}`. At most `MAX_PROMPTS` prompts (default 8) are accepted; more is HTTP 422 and no job. Poll `/api/jobs/{job_id}` until `completed` or `failed`; fields include progress, error, `error_code`, `retryable` and stems. Only one active process job per project is allowed.
4. `GET /api/projects/{id}` and `/stems` provide metadata. Project metadata includes the latest processing/visual job and last completed processing job for restoring a session. `GET /api/projects/{id}/original` serves the protected original, and `/preview` serves a protected browser-compatible H.264 preview that upload already transcoded (odd pixel dimensions are rounded down to even, keeping the aspect ratio); projects uploaded before that transcode it on first request. Both require the bearer token; fetch as a blob in the browser. Each stem contains `audio_url` requiring the same bearer header. Fetch as a blob for browser playback. `PATCH /api/projects/{id}/stems/{stem_id}` accepts `{"volume":0.5,"muted":false,"solo":false,"start_offset":0,"automation":[{"time":0,"value":1},{"time":10,"value":0.5}]}`. Volume and keyframe values range 0–2; keyframe times are strictly increasing. Automation multiplies volume. Solo restricts playback in exports to unmuted soloed stems. Edits never change source files.
5. Besides the editable fields, each stem carries the audio service's own metadata, or `null` where it reported none (always `null` for `source_stem_id`, `sample_rate`, `channels`, `format` and `spans` under the fake adapter): `role` (`target` or `residual`), `source_stem_id` (the service's correlation ID; `id` is still the one PATCH uses), `prompt` (the normalised phrase the model saw), `label` (human-facing, the clicked object's label for visual prompts), `sample_rate`, `channels`, `format` and `spans` (`[[start, end], ...]` seconds where the target is active — a UI hint, not a trim instruction). **Use `role == "residual"` to find the residual**; never string-match the name. The same fields appear in `GET /api/projects/{id}`, `GET /api/projects/{id}/stems` and the job output.
6. `POST /api/projects/{id}/export` with `{"format":"mp4","include_video":true}` returns an export ID. Poll `/api/exports/{id}` and GET its `download_url` with the bearer token. Audio-only formats use `{"format":"wav","include_video":false}` or `mp3`.
7. **Separation passes and refine.** `/process` and `/candidates` accept optional `"passes": 1..MAX_PASSES` (default 3, `GET /api/projects/{id}/capabilities` reports `max_passes` and `refine: true`); out of range is HTTP 422 and no job. SAM Audio sometimes only attenuates a sound, leaving a quieter copy in the residual; extra passes rerun the prompt on the residual inside one request. Each target stem then carries `refinement` (`{"passes_requested","passes_run","per_pass_energy_db":[...],"stopped_early"}`, `null` for residuals and older stems) in `GET /api/projects/{id}`, `/stems` and job output; candidate `provenance` adds `passes` and `refinement`. The fake adapter reports `passes_run = passes` with zero energies.
   To refine one layer afterwards, `POST /api/projects/{id}/stems/{stem_id}/refine` with optional `{"passes":1}` returns `202 {"job_id","status":"queued"}`; poll `GET /api/refine-jobs/{job_id}`. The job separates the current residual with the stem's `prompt` (or its name), reusing the anchors and, if still valid, the visual selection of the separation that created the stem (otherwise a warning says it ran text-only). It then rewrites both files: target' = target + extracted audio and residual' = the new residual, so the layers still sum to the same mix. Both stem IDs stay; `revision` advances on both; volume, mute, solo, automation and mute ranges are kept. `refinement` gains the new passes and energies plus a `refine_jobs` entry. The completed job's `result` is `{"stem_id","residual_stem_id","extracted_energy_db","passes","refinement","stems","stems_job_id","warnings","inference"}`; `stems_job_id` is a completed processing job for the existing stem loader. A failed job leaves both stems and their files unchanged. The route returns 409 when the stem is the residual or not a target, when the project does not have exactly one residual, when any other job for the project is queued or running, or when the stem or residual has a non-zero `start_offset`. While a refine job is active, new `/process` and `/candidates` requests and candidate apply are rejected with 409. Old audio files of the refined pair are deleted after the update, so an earlier candidate's preview audio can return 404.
8. `POST /api/webhooks/local` accepts `{"project_id":"...","token":"...","text":"Mute the dog from 10 to 15 seconds"}`. It returns `applied`, `accepted`, or `clarification`. Supported commands: mute/unmute, lower/set volume to a percentage, make louder, mute a range, export the video. Matching multiple stems requires clarification. `GET /api/projects/{id}/messages` retrieves completion notifications and confirmations.

## Messaging provider

`app.messaging.MessagingProvider` is the outgoing boundary and `LocalMessaging` persists an outbox. Photon is a placeholder, never selected automatically. Several distinct products use the Photon name; select the intended product, verify its current official SDK/webhook, signature and credentials, then implement `PhotonMessaging` and an authenticated provider-specific incoming adapter. The local webhook is a test protocol and must not be registered as a Photon webhook as-is. Photon Chat's [official webhook documentation](https://doc.photonengine.com/chat/current/reference/webhooks) describes its own contract and custom-header security, which is not necessarily the requested messaging product.

Stem metadata includes a monotonically increasing `revision`. To avoid overwriting another editor or a remote command, send `expected_revision` with a stem PATCH using the revision last read; the API returns HTTP 409 if it has changed. A successful write returns the new revision. Legacy PATCH requests without `expected_revision` are accepted but do not have conflict protection. The API accepts at most 64 automation keyframes per stem; older projects with longer curves must reduce them before updating or exporting reliably.

The stem PATCH also accepts `mute_ranges` as an array of `[start, end]` timeline seconds; send an empty array to clear remote range mutes. Values must be finite and ordered, and cannot end past the stem timeline. The editor flushes pending changes and uses `expected_revision` before applying settings.

## AI command planner

`POST /api/projects/{id}/agent` requires the project bearer token and takes `{"prompt":"make the music quieter","stems":[{"id":"...","label":"music","kind":"music","duration":10}],"tracks":[{"stemId":"...","gain":1,"muted":false,"solo":false,"automation":[]}],"duration":10,"has_project_stems":true}`. It returns a structured `kind` of `edit`, `separate` or `clarify`. The frontend validates action IDs, values, and timeline boundaries before applying them. The backend derives whether a project already has stems from its database. Configure `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` only in the backend environment; `OPENAI_AGENT_MODEL` defaults to `gpt-6-luna`. Media and bearer tokens are not included in the OpenAI request.

To add a sound without replacing current edits, submit `{"prompts":["dog barking"],"append":true}` to `/process`. The worker separates the existing unshifted `residual` stem and atomically replaces only that stem with its target and new residual, inheriting the residual mix settings. Concurrent residual edits fail the job without deleting existing layers. The `GET /api/projects/{id}/original-audio` endpoint serves the authenticated extracted original for A/B comparison. The audio service accepts bounded multipart metadata up to 64 MiB by default (`AUDIO_MAX_METADATA_BYTES`) for visual masks.

`POST /api/projects/{id}/prepare-visual` (also run automatically by `visual-analyze` when needed) queues a re-encoded compatible copy for a video with invalid start times, rotation or frame timing. Poll `/api/prepare-jobs/{job_id}`. It retains the uploaded original file, atomically updates the project to use the normalized copy, clears prior detections, and rejects projects with existing stems. Exports use the copy after preparation.

### Previewing a different separation

The guided candidate contract below supersedes the earlier text-only two-layer retry.

### Guided candidates (current contract)

The editor uses candidates for new separations and visual actions as well as replacement
retries. `/process` remains a direct-apply compatibility API for string-prompt clients.
Both requests accept `guidance`, `edit_scope`, and `profile`. `/process.prompts` also accepts
`{"text":"dog barking","name":"dog","anchors":[...]}` objects.

```json
{
  "prompt": "dog barking",
  "stem_id": null,
  "guidance": {"anchors": [{"token":"+","start":3,"end":4}]},
  "edit_scope": {"action":"mute","interval":{"start":10,"end":15}},
  "profile": "preview",
  "visual_prompt": {}
}
```

All times are finite source-timeline seconds. Intervals must be ordered and within the
extracted audio. `-` anchors indicate absence. Guidance never becomes an edit range.
Anchors guide overlapping inference windows, not persistent speaker identity. The example
separates the dog across the source, then adds `[10,15]` to the target's mute ranges while
preserving automation. Whole-layer actions are `keep`, `mute`, or `lower` (50%);
only `mute` accepts an interval. An edit through `/process` requires one target.

Omitting `stem_id` previews from the original when there are no stems, or splits the
current unshifted residual in a multitrack project. A supplied `stem_id` replaces a target
only when there are exactly two layers. Replacing an older multitrack target is rejected:
it needs an earlier source snapshot, not arbitrary recombination of current stems.

Candidate status returns protected `audio_url` (isolated target), `mix_url` (proposed mix),
`original_url` (mix before this edit), sanitized `provenance`, and actual `inference` settings.
The comparisons use the same timeline without automatic loudness normalization. Candidate
records retain source hash/reference, all stem revisions and edits, complete visual guidance,
explicit anchors, requested options, random seed, and resulting audio references.

Apply is one SQLite transaction. It rejects changed media, changed layer sets/revisions,
missing audio, repeat applies, and active media replacement jobs. It returns a completed
`job_id` suitable for the existing stem loader. `POST /api/candidates/{id}/undo` restores the
previous audio and edits only if every revision still matches the applied preview. Revisions
advance on undo; intervening edits require a new preview. The additive `candidate_history`
table preserves previous references across restarts. Apply and undo delete no stem files (a refine job deletes the two files it replaces).
`uploaded_original` preserves the upload separately from a normalized working copy;
`GET /api/projects/{id}/uploaded-original` serves it with project authorization. For older
projects, migration initializes this field to their currently referenced original.

`GET /api/projects/{id}/capabilities` exposes a whitelist of usable options, never internal
paths, credentials, service addresses or raw load errors. `preview` requests one candidate;
`quality` requests two judge-ranked candidates. Both use 10 s windows, 2 s overlap, batch
two, automatic spans off, and a mixture-derived residual. Quality controls stay disabled
when the text judge is unavailable. Ranking expresses preference, not calibrated confidence.

The planner separates `description`, `guidance`, `edit_scope`, and existing-layer actions.
The backend reloads authoritative stems/edits, validates every generated operation, and
rejects generated anchor/edit timestamps absent from the supplied instruction. Existing
layer timed mutes use `mute_range` operations and append ranges without replacing automation.
Multiple plausible layer matches produce a clarification listing the matches.

Uploads and candidate mixes use float32 WAV intermediates; originals remain available.
Upload metadata reports peak/RMS findings, channel downmixing and incompatible visual timing.
These findings do not imply restoration of clipped or absent information. Visual guidance
is experimental: native frame/mask/track/source checks remain enforced, the UI displays the
selected mask, and visible intervals no longer automatically create positive sound anchors.
Unusable visual input fails explicitly; suspicious output offers a text-plus-time retry.
