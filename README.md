# TidePool

**AI-powered sound editing for video.**

TidePool separates the audio in a video into individual layers (dialogue, a barking dog, rain, music) so each sound can be edited on its own. Describe a change in plain language, select an object in the frame, or work directly on the timeline, then export the finished result as MP4, WAV or MP3.

TidePool pairs an **audio separation model** (AudioSep) with **computer vision models** (YOLO and SAM 3.1): vision identifies what is on screen and when, and audio isolates what it sounds like. See [Models](#models).

---

## Table of contents

1. [Features](#features)
2. [Getting started](#getting-started)
3. [Using TidePool](#using-tidepool)
4. [Example commands](#example-commands)
5. [Architecture](#architecture)
6. [Models](#models)
7. [Project structure](#project-structure)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Limitations](#limitations)
11. [Further reading](#further-reading)

---

## Features

| Feature | Description |
| --- | --- |
| **Current** · command bar | Edit by typing or speaking, e.g. *"make the siren quieter after 10 seconds"* or *"trim from 3 to 4 seconds and export as MP4"*. |
| **Whirlpool** · separation | Extracts a described sound onto its own layer. Every layer set sums exactly to the original audio. |
| **Explode** · full-scene split | Splits a clip into dialogue (one layer per speaker), weather, distinct effects and background in a single step, with optional per-instrument music layers. |
| **Ripple** · visual selection | Detects and tracks people and objects. Select one to edit its sound, or trigger a sound each time it appears. |
| **Reflection** · A/B comparison | Switches between the original, the edited mix and the isolated sound at the same point in time. |
| **Deep dive** and **Undertow** | A higher-quality separation pass, and a follow-up pass that extracts more of a partially separated sound. |
| **Tide chart** · timeline | Per-layer mute, solo, volume (up to 500%) and volume automation, plus cutting, splitting and rearranging the video with keyboard shortcuts. |
| **Sound replacement** | Finds repeated sounds and replaces or augments them with uploaded or generated effects, each on its own layer. |
| **Effects** | Per-layer EQ and compression; time-ranged deepen, impact, disorient and muffle; whole-mix creative treatments. |
| **Moon pull** | Rules that give a sound a consistent role across the clip, driven by dialogue, marked moments or loudness. |
| **Text-based filtering** | Transcribes speech with word-level timing so edits can target what is said: beep or replace profanity, add a sound whenever a chosen word is spoken, or drive Moon pull rules from dialogue. |
| **Music stems** | Splits a song into vocals, drums, bass, guitar, piano and other. |
| **Outflow** · export | MP4 with the edited audio and all video cuts, trims and rotation applied, or WAV/MP3 audio. |
| **Projects** | Project library, save, full undo/redo history, video rotation and a responsive phone layout. |

All editing is non-destructive. The original upload is always retained, every change is previewed before it is applied, and every applied change can be undone.

---

## Getting started

### Requirements

- Python 3.11+
- Node.js 22+
- FFmpeg

On macOS: `brew install python@3.12 node ffmpeg`. Real separation requires an NVIDIA GPU; all other components run locally.

### Option A: Local development (no GPU)

Runs the full application with a mock separator that returns filtered copies of the audio. Useful for UI and integration work.

```sh
scripts/dev_setup.sh              # create .venv and install the API and vision packages
(cd frontend && npm ci)           # install the editor
scripts/run_local_real.sh mock    # start the mock audio service, backend and editor
```

Open **http://localhost:5173/#/app**. Stop all services with `scripts/run_local_real.sh stop`.

### Option B: Local editor with a remote GPU (recommended)

The GPU host (e.g. a RunPod A40) runs the AudioSep separator and SAM 3.1 object selection. The editor, backend and all project data stay on your machine.

**1. Start the GPU service** on the GPU host, from a checkout of this repository:

```sh
bash scripts/runpod_audiosep.sh setup     # first run only
bash scripts/runpod_audiosep.sh start     # serves privately on 127.0.0.1:8011
bash scripts/runpod_audiosep.sh status
```

**2. Open an SSH tunnel** on your machine and keep it running:

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8001:127.0.0.1:8011 -p <SSH_PORT> -i ~/.ssh/id_ed25519 root@<SSH_HOST>
```

**3. Start TidePool:**

```sh
scripts/run_local_real.sh    # verifies the GPU service, then starts the backend and editor
```

Open **http://localhost:5173/#/app**. Logs are written to `logs/`.

> **Troubleshooting:** *"Object selection service is unreachable"* means the tunnel or GPU service is down. Run `curl http://127.0.0.1:8001/ready`, restart the service on the GPU host if needed, and reopen the tunnel.

See [GPU setup](docs/tidepool_audiosep.md) for the full procedure.

### Option C: Docker

With the GPU service tunnelled to `localhost:8001` as in Option B:

```sh
docker compose up --build
```

For a single Linux NVIDIA host running every component, use `compose.cloud.yaml`. See the [architecture guide](docs/ARCHITECTURE.md).

### Optional components

| Capability | Setup |
| --- | --- |
| Natural-language commands | Set `OPENAI_API_KEY` in `./.env` (backend only) |
| Object detection | Installed by `scripts/dev_setup.sh`; YOLO weights (~6 MB) download on first use |
| Text-based filtering | Local faster-whisper runtime: [setup](INTEGRATION.md#local-transcription-and-profanity-beeps) |
| Sound effect generation | Stable Audio CLI via `SOUNDCANVAS_SFX_CLI`: [setup](INTEGRATION.md#add-or-replace-sound-effects) |
| Music stem splitting | `AUDIO_STEM_SPLITTER=demucs` on the audio service: [setup](INTEGRATION.md#music-stems-song--one-track-per-instrument) |

---

## Using TidePool

The editor guides each project through four steps:

1. **Upload:** an MP4, MOV, WAV or MP3 file of up to 500 MB.
2. **Pick a sound:** describe it, run **Explode** on the whole clip, or select an object in the video.
3. **Preview and apply:** compare with Reflection, then apply or discard.
4. **Export:** MP4, WAV or MP3.

Layers can then be refined on the timeline or through the command bar.

### Keyboard shortcuts

Shortcuts apply when the timeline has focus.

| Shortcut | Action |
| --- | --- |
| `Space` | Play or pause the edited sequence |
| `R` / `A` | Range tool / clip selection tool |
| `Delete` | Remove the selection from the video and close the gap |
| `⌘C` `⌘X` `⌘V` | Copy, cut, paste at the playhead |
| `⌘B` | Split at the playhead |
| `⌘Z` / `⇧⌘Z` | Undo / redo |
| `I` / `O` | Set selection start / end at the playhead |
| `←` `→` | Move the playhead one frame (`⇧` for one second) |
| `⌥←` `⌥→` | Move the selected clip earlier or later |
| `⌘A` / `Esc` | Select all / clear selection |
| `⌘E` | Export |
| `⌘S` | Save project |

The context menu offers the same actions plus **Trim to selection**. Video edits apply to all audio layers, and each project's clip arrangement is saved in the browser.

---

## Example commands

| Goal | Command |
| --- | --- |
| Isolate and mute a sound | `isolate the dog barking` → `mute the dog from 10 to 15 seconds` |
| Timed volume change | `make the dialogue quieter from 2 to 5 seconds` |
| Local removal | `remove the chopping at 4.8 seconds` |
| Edit within a layer | `separate barking from the music layer at 4 seconds` |
| Replace repeated sounds | `replace every punch with a quack` → `replace all quacks with loud gongs` |
| Sound tied to an object | `every time person 5 appears, play a gong sound` |
| Creative treatment | `make it more dramatic` |
| Trim and export | `trim 3 sec to 4 sec and export as mp4` |
| Text-based filtering | `beep the swear words` · `add a gong whenever the word "hello" is said` |

Detection and separation are approximate; review each preview before applying it.

---

## Architecture

```
 Browser · React editor (:5173)
    │   /api proxy · per-project bearer token
    ▼
 FastAPI backend + job worker (:8000) ─── SQLite + media in ./storage
    │   projects · jobs · previews · undo checkpoints · FFmpeg export
    │
    ├──▶ Audio service · GPU (:8001 via SSH)    AudioSep · Demucs · Explode · SAM 3.1 selection
    ├──▶ Vision pipeline · local CPU/MPS        YOLO detection and tracking
    └──▶ OpenAI (optional)                      command planning
```

**Processing flow**

1. **Ingest:** the backend stores the original, extracts 48 kHz stereo audio and creates a browser preview.
2. **Separate:** a background job sends audio and a text prompt to the audio service, which returns a *target* and a *residual*. Together they reconstruct the input exactly.
3. **Preview:** each change is built as a candidate for comparison. Applying it records a checkpoint for undo.
4. **Edit:** mix settings are stored per layer and never modify source files; effects render new audio alongside the old.
5. **Export:** FFmpeg mixes all layers, applies video edits and encodes MP4 (H.264/AAC), WAV or MP3.

The language model receives only the command, layer names and timeline state, never media files or project tokens. Every planned action is validated before it is applied.

---

## Models

★ marks the core models.

### Audio

| Model | Role | Runs on |
| --- | --- | --- |
| ★ **AudioSep** (base 4M) | Text-prompted separation of any described sound | GPU host |
| **Demucs** `htdemucs_ft` | Dialogue extraction for Explode | GPU host |
| **Demucs** `htdemucs_6s` | Music stem separation | GPU host |
| **ECAPA-TDNN** | Speaker embeddings for per-speaker dialogue layers | GPU host |
| **CLAP**, **PANNs** Cnn14 | Sound tagging, verification and song/scene routing | GPU host |
| **SAM Audio** (small, large) | Alternative separator with time and visual prompting | GPU host, optional |
| **YAMNet** | Sound event timing | Backend CPU, optional |
| **faster-whisper** | Word-timed transcription for text-based filtering | Backend, optional |
| **Stable Audio 3 Small SFX** | Sound effect generation | Configured host, optional |

### Vision

| Model | Role | Runs on |
| --- | --- | --- |
| ★ **YOLO26n-seg** + ByteTrack | Object detection, segmentation and tracking (first 30 s of video) | Apple GPU (MPS) or CPU |
| ★ **SAM 3.1** | Precise object segmentation from a click | GPU host |
| **Qwen3-VL 2B** | Scene captions for creative requests | Local GPU, optional |

### Language

| Model | Role | Runs on |
| --- | --- | --- |
| **OpenAI** `gpt-6-luna` (default) | Converts commands into structured edit plans and resolves references to existing effects | OpenAI API, optional |

**How audio and vision work together:** vision determines *where* an object is and *when* it is on screen; AudioSep isolates *what it sounds like*. AudioSep accepts only text and audio, so selecting an object produces a text prompt for its sound, applied across the clip or a chosen time range. For object-driven effects, tracked on-screen times determine placement.

---

## Project structure

| Path | Contents |
| --- | --- |
| `frontend/` | React and TypeScript editor built with Vite: UI components, playback engine, command routing, timeline and undo state, Moon pull rules, pages |
| `app/` | FastAPI backend: routes, job worker, FFmpeg media and export, command planner, previews, undo checkpoints, sound replacement, effects, Moon pull, text-based filtering, SAM bridge |
| `audio-service/` | GPU inference service: AudioSep, SAM Audio, Explode, music stems and tagging ([README](audio-service/README.md)) |
| `vision/` | Detection, tracking and mask pipeline, `soundcanvas_cv` ([README](vision/README.md)) |
| `scripts/` | Local setup, service launcher, GPU host setup, demo and benchmark tooling |
| `tests/` | Backend test suite |
| `docs/` | Architecture, setup guides, contracts and design plans |
| `compose*.yaml`, `Dockerfile` | Container configurations for local and GPU deployments |

---

## Configuration

Configuration uses environment variables, loaded from `./.env` by `scripts/run_local_real.sh`. See `.env.example` for the full list.

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | none | Enables natural-language commands (`OPENAI_API_KEY_FILE` also supported) |
| `OPENAI_AGENT_MODEL` | `gpt-6-luna` | Model used for command planning |
| `INFERENCE_ADAPTER` | `http` | `http` for the audio service; `fake` for UI-only development |
| `SAM_AUDIO_SERVICE_URL` | `http://127.0.0.1:8001/separate` | Separation endpoint |
| `VISION_SERVICE_URL` | `http://127.0.0.1:8001` | SAM 3.1 object selection endpoint |
| `STORAGE_ROOT` | `./storage` | Project data and SQLite database |
| `MAX_UPLOAD_BYTES` | 500 MiB | Maximum upload size |
| `VISION_DEVICE` | `mps` | Device for local object detection |
| `INFERENCE_TIMEOUT_SECONDS` | `3600` | Timeout for GPU requests |

Audio service settings (`AUDIO_BACKEND`, `AUDIO_CHUNK_SECONDS`, `AUDIO_STEM_SPLITTER` and others) are documented in [audio-service/README.md](audio-service/README.md).

---

## Testing

```sh
.venv/bin/pytest -q                                   # backend (requires FFmpeg)
(cd frontend && npm run verify)                       # editor: types, lint, tests, build
(cd audio-service && .venv/bin/python -m pytest -q)   # audio service
```

Vision tests are described in [vision/README.md](vision/README.md). Continuous integration runs the backend and vision suites on every pull request, and the frontend and audio-service suites when those directories change. Interactive API documentation is available at **http://localhost:8000/docs** while the backend is running.

---

## Limitations

- **Deployment:** TidePool is a development MVP. It runs as a single backend process with an embedded SQLite worker and is not intended for multi-replica or public hosting.
- **Accuracy:** separation is approximate, and overlapping sounds (particularly speech) can bleed between layers. Object tracking covers the first 30 seconds and may miss or swap identities.
- **Security:** each project's bearer token grants full access to that project and is stored only in the browser. Access the app through an SSH tunnel or a secured HTTPS deployment; do not expose service ports.
- **Cost:** rented GPUs bill until stopped. Separated layers remain playable after the GPU host is shut down.
- **Not yet supported:** moving a split section of audio onto a separate track, and outbound notifications.

---

## Further reading

| Document | Covers |
| --- | --- |
| [Architecture guide](docs/ARCHITECTURE.md) | Code map, data flow, verification |
| [Integration contracts](INTEGRATION.md) | APIs, jobs and feature contracts |
| [GPU setup](docs/tidepool_audiosep.md) | GPU service and local setup |
| [Object selection](docs/object_selection_audiosep.md) | Visual selection with AudioSep |
| [Audio service](audio-service/README.md) · [Vision](vision/README.md) · [Frontend](frontend/README.md) | Component documentation |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |
