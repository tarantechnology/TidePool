# Integration contracts

Base URL `http://localhost:8000`; OpenAPI and executable examples are at `/docs` and `/openapi.json`. Except project creation and local webhook, requests require `Authorization: Bearer <project token>`. A token grants full access to one project and must be sent for media downloads as well. Protect it like a password. FastAPI errors are `{"detail":"message"}`; validation errors return a detail array. IDs are opaque.

## Person 1: audio AI

Set `INFERENCE_ADAPTER=http` and `SAM_AUDIO_SERVICE_URL` to your endpoint. The worker POSTs multipart form data: `audio` is 48 kHz, stereo, signed 16-bit PCM WAV; `metadata` is a JSON string such as `{"prompts":["dog"],"visual_prompt":{"track_id":"42"}}`. Return HTTP 200 with JSON:

```json
{"stems":[{"name":"dog","audio_base64":"<base64-encoded WAV bytes>"}]}
```

Return at least one valid WAV containing audio. Duration is probed locally. The default HTTP timeout is 600 seconds (`INFERENCE_TIMEOUT_SECONDS`), and each decoded stem is capped by `MAX_STEM_BYTES`. Non-2xx, timeout, malformed JSON, invalid base64 and invalid audio fail the job; its error appears at `GET /api/jobs/{id}`. To retry, submit a new processing job. Service responses are trusted and should come from a trusted network. In fake mode each prompt produces a copy of the original audio, for integration only.

## Person 2: vision

`PUT /api/projects/{id}/visual-metadata` accepts an arbitrary JSON object containing detections, tracks, timestamps, bounding boxes and selection IDs, for example `{"tracks":[{"id":"42","label":"dog","frames":[{"time":1.0,"bbox":[0.1,0.2,0.3,0.4]}]}]}`. Agree coordinate conventions with Person 1. That object is passed unchanged as `visual_prompt` on processing unless the process request explicitly supplies a `visual_prompt` object. Alternatively send `{ "prompts":["dog"], "visual_prompt":{"track_id":"42"} }` directly to `/process`. The backend does not run detection or infer associations.

The independent [Part 2 CV package](vision/README.md) proposes [schema 1.0](vision/docs/CONTRACT.md) for these arbitrary objects, with source/analysis identity, pixel and normalized boxes, bounded intervals, and optional instance-mask RLE. This is additive documentation for review in issues #3/#4; no backend validation or audio inference behavior changes. The WAV+JSON inference boundary does not transmit original video; real SAM Audio visual prompting still requires an agreed video resolver and aligned audio/mask preparation.

## Person 3: frontend

1. `POST /api/projects` returns `{"project_id":"...","access_token":"...","status":"created"}`. Save the token securely.
2. `POST /api/projects/{id}/upload` with multipart `file`; allowed extensions MP4/MOV/WAV/MP3, size defaults to 500 MiB. Response includes original media metadata and status. Upload extracts 48 kHz stereo WAV synchronously.
3. `POST /api/projects/{id}/process` with `{"prompts":["dog","music"]}` returns `{"job_id":"...","status":"queued"}`. Poll `/api/jobs/{job_id}` until `completed` or `failed`; fields include progress, error and stems. Only one active process job per project is allowed.
4. `GET /api/projects/{id}` and `/stems` provide metadata. Each stem contains `audio_url` requiring the same bearer header. Fetch as a blob for browser playback. `PATCH /api/projects/{id}/stems/{stem_id}` accepts `{"volume":0.5,"muted":false,"solo":false,"start_offset":0,"automation":[{"time":0,"value":1},{"time":10,"value":0.5}]}`. Volume and keyframe values range 0–2; keyframe times are strictly increasing. Automation multiplies volume. Solo restricts playback in exports to unmuted soloed stems. Edits never change source files.
5. `POST /api/projects/{id}/export` with `{"format":"mp4","include_video":true}` returns an export ID. Poll `/api/exports/{id}` and GET its `download_url` with the bearer token. Audio-only formats use `{"format":"wav","include_video":false}` or `mp3`.
6. `POST /api/webhooks/local` accepts `{"project_id":"...","token":"...","text":"Mute the dog from 10 to 15 seconds"}`. It returns `applied`, `accepted`, or `clarification`. Supported commands: mute/unmute, lower/set volume to a percentage, make louder, mute a range, export the video. Matching multiple stems requires clarification. `GET /api/projects/{id}/messages` retrieves completion notifications and confirmations.

## Messaging provider

`app.messaging.MessagingProvider` is the outgoing boundary and `LocalMessaging` persists an outbox. Photon is a placeholder, never selected automatically. Several distinct products use the Photon name; select the intended product, verify its current official SDK/webhook, signature and credentials, then implement `PhotonMessaging` and an authenticated provider-specific incoming adapter. The local webhook is a test protocol and must not be registered as a Photon webhook as-is. Photon Chat's [official webhook documentation](https://doc.photonengine.com/chat/current/reference/webhooks) describes its own contract and custom-header security, which is not necessarily the requested messaging product.
