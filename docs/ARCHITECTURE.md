# Architecture and development guide

TidePool is a browser-based media editor backed by a Python API, an asynchronous worker, an audio inference service, and an optional computer-vision pipeline. The editor and API can run locally with mock inference; model-backed separation requires separately installed models and suitable hardware.

## Code map

- `frontend/src/`: React and TypeScript editor. `components/` contains UI panels, `api/` handles backend requests and jobs, `audio/` and `export/` handle playback and output, and `state/` and `agent/` coordinate project state and command plans.
- `app/main.py`: FastAPI routes, project authentication, uploads, jobs, and exports. `core.py` owns storage and database helpers; `worker.py` processes queued work; `media.py` handles media operations. `agent.py`, `director.py`, `discovery.py`, and `replacement.py` implement command planning, creative previews, detection, and repeated-event editing.
- `audio-service/sam_audio_service/`: inference API and engine. `config.py` defines runtime settings; `backends/` contains separation backends; `routing.py`, `classifier.py`, and `stems/` support music stem routing. See the service README for model dependencies and limitations.
- `vision/soundcanvas_cv/`: object detection, tracking, masks, and visual-selection contracts. The API starts vision analysis in a separate process using the same Python interpreter.
- `tests/`, `audio-service/tests/`, `vision/tests/`, and `frontend/src/**/*.test.ts`: backend, service, vision, and editor tests. `docs/contracts/` and `INTEGRATION.md` describe interfaces; `scripts/` contains setup and operational helpers.

## Data flow

A user uploads media and receives a project ID and bearer token. The API stores project state in SQLite and files under `STORAGE_ROOT`, then enqueues processing. A single embedded worker extracts or prepares media and calls the configured inference adapter. The audio service reads shared media, runs the selected backend, and returns stem metadata. The editor polls job state, renders layers and previews, and submits edits or export requests. Optional vision analysis supplies selections for supported visual guidance. Audio model output and detection are approximate and require review.

The worker is designed for one API process and one replica. In HTTP inference deployments, the API and audio service need compatible access to the configured media root; container examples use a shared volume. Keep project tokens, media, and model credentials private. The local notification implementation is a test webhook/outbox, not a phone messaging service.

## Configuration and execution

Copy `.env.example` only when needed and supply deployment-specific values without committing secrets. `compose.yaml` runs the local stack with fake inference; `compose.cloud.yaml` describes an NVIDIA GPU deployment. For individual process commands, prerequisites, cloud caveats, and model setup, follow the root README and the audio service README. Model weights, runtime data, generated outputs, virtual environments, and credentials are not source files.

## Verification

- Backend: install `requirements.txt` and pytest, then run `pytest -q` from the repository root. FFmpeg and ffprobe are required for media tests.
- Frontend: run `npm ci` and `npm run verify` in `frontend/` for TypeScript, lint, unit tests, and production build.
- Audio service: install its development dependencies and run `pytest` in `audio-service/`. GPU and real-model tests have additional requirements; consult its README.
- Vision: see `vision/README.md` for optional dependencies and test commands.

For API exploration, start the backend and open `/docs`. For contract changes, update both the relevant service documentation and the client integration, and exercise the affected tests.
