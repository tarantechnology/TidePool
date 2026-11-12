# SoundCanvas

SoundCanvas is a browser-based audio-layer editor for video. Create a project, upload media, separate sounds into stems, adjust and automate the layers, and export the result. The repository includes the editor, FastAPI backend, audio service and vision pipeline. See [INTEGRATION.md](INTEGRATION.md) for their contracts and [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration.

## Run locally on a Mac

Install Docker Desktop, start it, and run:

```sh
docker compose up --build
```

Open **http://localhost:5173**. The API documentation is at http://localhost:8000/docs. Data persists in `./storage`. The first build installs CPU vision dependencies; the first visual analysis downloads detector weights. A network connection and sufficient disk space are needed.

1. Choose an MP4, MOV, WAV or MP3 and click **Create & upload**. Copy the project ID and access token shown in the project panel; they are needed to restore the project later.
2. Enter comma-separated sound descriptions and start processing. The default local inference adapter **copies the entire audio into each test stem**; it does not separate sounds. This mode is useful for exercising the editor without a GPU.
3. Play and edit the layers, then choose MP4, WAV or MP3 and export. For video object selection, run visual analysis, pause the video, and click a detected object. A sound prompt is still required.
4. The local message panel can test commands such as `Mute the dog` or `Mute the dog from 1 to 2 seconds`. It is a local webhook and outbox, not a messaging delivery service.

## Use a rented cloud GPU from a Mac

The Mac runs the browser. The Linux GPU machine runs the editor, backend, CPU vision pipeline and NVIDIA SAM Audio service together, sharing project files. The GPU service is internal to Compose and isn't exposed publicly. This avoids mounting your Mac filesystem or sending local paths to another host.

1. Rent a Linux x86-64 NVIDIA GPU VM with enough GPU memory and disk for the model, Docker images, uploaded media and outputs. Install Docker Compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and confirm a GPU container can see the device. Open SSH to your trusted IP; keep application ports private.
2. Obtain access to the gated [`facebook/sam-audio-small`](https://huggingface.co/facebook/sam-audio-small) model and a Hugging Face token with that access. Clone this repository on the VM and start it there:

   ```sh
   cd SoundCanvas
   read -rs HF_TOKEN; export HF_TOKEN   # paste token, then press Enter
   docker compose -f compose.cloud.yaml up --build -d
   docker compose -f compose.cloud.yaml logs -f
   ```

   Initial image builds and model downloads can take a while. The API waits for the audio service's readiness check. Model files persist in a Docker volume; projects persist in the VM's `./storage`. The defaults use the small model, CUDA and 10-second audio chunks; inspect `compose.cloud.yaml` for overrides.
3. On your Mac, create an SSH tunnel, replacing the placeholders with your VM details:

   ```sh
   ssh -L 5173:127.0.0.1:5173 USER@GPU_VM
   ```

   Keep SSH open and visit **http://localhost:5173** on your Mac. The editor uses its same-origin `/api` proxy. Use a separate tunnel to `8000` if you need API docs.
4. When finished, run `docker compose -f compose.cloud.yaml down` on the VM. GPU rental charges may continue until you stop or terminate the VM. Back up `./storage` and any needed Docker volumes before deleting the VM.

The cloud Compose configuration has been checked syntactically, but a real GPU deployment and gated model download have not been validated in this environment. Check service logs if model access, GPU drivers, memory or container startup fails. Real visual-guided audio isolation remains unverified; see [vision/docs/ISSUE4.md](vision/docs/ISSUE4.md). The backend returns inference model identity and warnings so the editor can distinguish mock results.

## APIs and tests

FastAPI serves `/docs` and `/openapi.json`. Project media and edits require `Authorization: Bearer <project token>`; the token is shown once at creation. The UI keeps credentials in memory, so save them securely before closing the page. Jobs and exports run asynchronously; uploads and first preview conversion can take time.

With Python 3.11+, FFmpeg and ffprobe installed, run backend tests using `pip install -r requirements.txt pytest` and `pytest -q`. Run editor checks with `cd frontend && npm ci && npm run verify`. Vision's optional tests are documented in [vision/README.md](vision/README.md). An optional Playwright browser smoke test is in `tests/test_editor_browser.py`; point `SOUNDCANVAS_BROWSER_URL` at a running editor configured with fake inference and install Playwright plus Chromium to run it.

The embedded SQLite worker is intended for one backend process and one replica. Jobs can be retried by submitting a new request after failure; interrupted work is recovered on startup. Video export copies the video stream where the MP4 container supports it and encodes audio as AAC. Incompatible video codecs may cause export failure. Automation and remote mute ranges use timeline seconds.

This is a development MVP. Keep project tokens and uploaded media private; use a trusted SSH tunnel or a properly secured HTTPS deployment with authentication, access controls and rate limits. Photon integration is a placeholder pending an identified provider and verified API; local notifications do not send messages to a phone.

## Natural-language commands

Set `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` on the backend to enable GPT-6 Luna in the editor command bar. `OPENAI_AGENT_MODEL` defaults to `gpt-6-luna`. Project authentication is required. The request sends the command, timeline state and stem names to OpenAI; it does not send uploaded media or project tokens. The backend asks for a structured plan and the editor validates each edit before applying it.

Commands can request separation of a sound in a new project or edit existing stems. Creating new stems currently replaces existing ones, so the planner asks for a new project instead of discarding edits. It cannot perform effects or infer audio content or timestamps that the editor does not provide. OpenAI usage incurs charges and requires model access.
