# Integration contracts

Base URL `http://localhost:8000`; OpenAPI and executable examples are at `/docs` and `/openapi.json`. Except project creation and local webhook, requests require `Authorization: Bearer <project token>`. A token grants full access to one project and must be sent for media downloads as well. Protect it like a password. FastAPI errors are `{"detail":"message"}`; validation errors return a detail array. IDs are opaque.

## Person 1: audio AI

Set `INFERENCE_ADAPTER=http` and `SAM_AUDIO_SERVICE_URL` to your endpoint. The worker POSTs multipart form data: `audio` is 48 kHz, stereo, signed 16-bit PCM WAV; `metadata` is a JSON string such as `{"prompts":["dog"],"visual_prompt":{"track_id":"42"}}`. Return HTTP 200 with JSON:

```json
{"stems":[{"name":"dog","audio_base64":"<base64-encoded WAV bytes>"}]}
```

Return at least one valid WAV containing audio. Float32 WAV preserves stem headroom and is supported by the backend; PCM16 may clip individual stems. In HTTP mode at least one nonempty text prompt is required before a job is queued. Duration is probed locally. The default HTTP timeout is 3600 seconds (`INFERENCE_TIMEOUT_SECONDS`), and each decoded stem is capped by `MAX_STEM_BYTES`. Non-2xx, timeout, malformed JSON, invalid base64 and invalid audio fail the job; its error appears at `GET /api/jobs/{id}`. To retry, submit a new processing job. Service responses are trusted and should come from a trusted network. In fake mode each prompt produces a copy of the original audio, for integration only.

## Person 2: vision

`PUT /api/projects/{id}/visual-metadata` stores detection metadata. Legacy label/time-range objects pass unchanged as `visual_prompt` unless `/process` supplies an override. Client media-reference fields (`video_ref`, `original_ref`, `mask_video_ref`, `detections_ref`, `audio_ref`) are rejected, including nested fields. The backend does not run detection or infer sound associations.

For actual visual prompting, store the [CV schema-1.0 detections](vision/docs/CONTRACT.md) generated at native FPS with instance masks, then send `{"prompts":["dog barking"],"visual_prompt":<exported visual_selection>}`. The backend validates media/analysis identity, the observed selection, mask geometry/RLE, complete native-frame coverage and a 0.1–30 s frame-aligned interval. It snapshots `{"selection":...,"detections":...}` without changing those documents and adds server-owned `source` metadata (`original_ref`, SHA256, project asset ID, native FPS, duration). A stored envelope with both documents also works. Analyzed track absence yields empty foreground; missing frame records or masks fail validation. Bounding boxes alone cannot guide the model. CFR media must start audio/video at zero and have no rotation metadata; normalize before upload and analyze those same bytes.

Both services must see the same absolute storage path: the existing audio Compose overlay mounts `/data` read-only into the audio service with `AUDIO_MEDIA_ROOT=/data`. Local processes should set `AUDIO_MEDIA_ROOT` to the backend's absolute `STORAGE_ROOT`. No project token or client file path is forwarded. The worker rechecks uploaded bytes before inference and accepts only zero-offset, full-duration visual stems with matching rate/channels. Job responses add `inference` with model identity, warnings and service source/request metadata; `model.is_mock=true` means no real separation. See [Issue #4 validation and exact commands](vision/docs/ISSUE4.md). **Real visual-guided model isolation remains unverified; mock transport/export tests do not establish sound association.**

## Person 3: frontend

1. `POST /api/projects` returns `{"project_id":"...","access_token":"...","status":"created"}`. Save the token securely.
2. `POST /api/projects/{id}/upload` with multipart `file`; allowed extensions MP4/MOV/WAV/MP3, size defaults to 500 MiB. Response includes original media metadata and status. Upload extracts 48 kHz stereo WAV synchronously.
3. `POST /api/projects/{id}/process` with `{"prompts":["dog","music"]}` returns `{"job_id":"...","status":"queued"}`. Poll `/api/jobs/{job_id}` until `completed` or `failed`; fields include progress, error and stems. Only one active process job per project is allowed.
4. `GET /api/projects/{id}` and `/stems` provide metadata. Each stem contains `audio_url` requiring the same bearer header. Fetch as a blob for browser playback. `PATCH /api/projects/{id}/stems/{stem_id}` accepts `{"volume":0.5,"muted":false,"solo":false,"start_offset":0,"automation":[{"time":0,"value":1},{"time":10,"value":0.5}]}`. Volume and keyframe values range 0–2; keyframe times are strictly increasing. Automation multiplies volume. Solo restricts playback in exports to unmuted soloed stems. Edits never change source files.
5. `POST /api/projects/{id}/export` with `{"format":"mp4","include_video":true}` returns an export ID. Poll `/api/exports/{id}` and GET its `download_url` with the bearer token. Audio-only formats use `{"format":"wav","include_video":false}` or `mp3`.
6. `POST /api/webhooks/local` accepts `{"project_id":"...","token":"...","text":"Mute the dog from 10 to 15 seconds"}`. It returns `applied`, `accepted`, or `clarification`. Supported commands: mute/unmute, lower/set volume to a percentage, make louder, mute a range, export the video. Matching multiple stems requires clarification. `GET /api/projects/{id}/messages` retrieves completion notifications and confirmations.

## Messaging provider

`app.messaging.MessagingProvider` is the outgoing boundary and `LocalMessaging` persists an outbox. Photon is a placeholder, never selected automatically. Several distinct products use the Photon name; select the intended product, verify its current official SDK/webhook, signature and credentials, then implement `PhotonMessaging` and an authenticated provider-specific incoming adapter. The local webhook is a test protocol and must not be registered as a Photon webhook as-is. Photon Chat's [official webhook documentation](https://doc.photonengine.com/chat/current/reference/webhooks) describes its own contract and custom-header security, which is not necessarily the requested messaging product.

Stem metadata includes a monotonically increasing `revision`. To avoid overwriting another editor or a remote command, send `expected_revision` with a stem PATCH using the revision last read; the API returns HTTP 409 if it has changed. A successful write returns the new revision. Legacy PATCH requests without `expected_revision` are accepted but do not have conflict protection. The API accepts at most 64 automation keyframes per stem; older projects with longer curves must reduce them before updating or exporting reliably.
