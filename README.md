# SoundCanvas backend

An independent FastAPI backend for uploading media, running an asynchronous separation adapter, editing stems and rendering media with FFmpeg. The default adapter creates **copies**, not separated sounds. No frontend, CV model or SAM model is included.

## Team collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch and review conventions, and [INTEGRATION.md](INTEGRATION.md) for the contracts between audio AI, vision, frontend and backend. The backend MVP is present; the other three workstreams are ready to integrate.

## Start

Install Docker and run `docker compose up --build`. Open http://localhost:8000/docs for the interactive API. Data persists in `./storage`. Alternatively install Python 3.11+, FFmpeg and ffprobe, then run `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app`. Copy `.env.example` to `.env` and export its values when running without Compose; Compose reads `.env` itself.

Create a project at `POST /api/projects`. Save its `access_token` immediately: it is shown only once. Use `Authorization: Bearer <token>` for all project, job, stem and export endpoints, including playback and download. Upload with multipart field `file`, start `/process` with e.g. `{"prompts":["dog", "music"]}`, poll `/api/jobs/{job_id}`, patch a stem, then start `/export` and poll `/api/exports/{export_id}`. The local remote-control webhook accepts `{"project_id":"...","token":"...","text":"Mute the dog"}` at `POST /api/webhooks/local`. Read local notifications at `/api/projects/{project_id}/messages` with the project token. See [INTEGRATION.md](INTEGRATION.md) for complete contracts.

Run tests with `pip install pytest && pytest -q`. Tests require FFmpeg and ffprobe on PATH.

## Operation and limitations

Use one API process and one replica: its embedded worker claims persistent SQLite jobs and recovers interrupted work on startup. Jobs are at-least-once across a crash, and only one active job of each kind per project is allowed. Retry a failed job by submitting a new request. Uploaded files and completed exports remain on local disk; manage retention and backups externally. The fake adapter copies the whole input once per prompt, so mixing multiple fake stems increases loudness. Time-based mute ranges and automation apply in project timeline seconds. Automation values multiply the stem volume; linear interpolation is evaluated per audio frame. Video export copies the video stream where the MP4 container supports it, and encodes audio as AAC; unsupported codecs fail with a job error.

This is a local-development MVP: terminate TLS at a trusted proxy, protect tokens and webhook request bodies, restrict access to local storage, and add rate limits and a real identity/credential distribution system before internet exposure. Upload extraction is synchronous; separation and export are asynchronous. Responses contain FastAPI's standard `{"detail": ...}` error shape. Local webhook authentication uses the project token; it is not a Photon signature. Photon is intentionally disabled pending a specific provider and verified authentication/delivery contract. The local outbox stores notifications; it does not deliver SMS or iMessage.
