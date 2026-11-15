# TidePool

TidePool is a browser-based audio-layer editor for video. Create a project, upload media, separate sounds into stems, adjust and automate the layers, and export the result. The repository includes the editor, FastAPI backend, audio service and vision pipeline. See [INTEGRATION.md](INTEGRATION.md) for their contracts and [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration.

## Repository guide

Start with the [architecture and development guide](docs/ARCHITECTURE.md) for the code map, request flow, configuration, and test commands. [INTEGRATION.md](INTEGRATION.md) describes service contracts and feature limitations; component-specific setup lives in [audio-service/README.md](audio-service/README.md), [frontend/README.md](frontend/README.md), and [vision/README.md](vision/README.md).

## Default separator: AudioSep

TidePool uses AudioSep for text-prompted separation, with conservative source-phase
suppression enabled to reduce unwanted changes to voices. Requests use one candidate,
10-second windows, 2-second overlaps and sequential processing. The editor reports the
serving model. Preview, apply/undo, timed edits and export use the same processed stems.
AudioSep has no visual masks, sound anchors or ranked quality retries.

See [TidePool + AudioSep startup](docs/tidepool_audiosep.md) for the RunPod service,
private SSH tunnel and Mac editor. Keep the Pod running while using inference.

## Run locally on a Mac

With the AudioSep GPU service forwarded to localhost:8001, install Docker Desktop and run:

```sh
docker compose up --build
```

Open **http://localhost:5173**. The API documentation is at http://localhost:8000/docs. Data persists in `./storage`. The first build installs CPU vision dependencies; the first visual analysis downloads detector weights to `./storage/vision`. A network connection and sufficient disk space are needed.

1. Choose an MP4, MOV, WAV or MP3 to upload it. Copy the project ID and access token shown in the project panel; they are needed to restore the project later.
2. Enter comma-separated sound descriptions and start processing. The backend sends the request to the configured AudioSep service. For explicit UI-only development, set `INFERENCE_ADAPTER=fake`; that mode copies audio and performs no separation.
3. Play and edit the layers, then choose MP4, WAV or MP3 and export. For video object selection, run visual analysis, pause the video, and click a detected object. A sound prompt is still required.

Open **Trim and export** to keep a start/end range, either by entering seconds or setting each boundary at the playhead. Reset restores the full range. Trimming is applied to the exported copy, with the mixed audio and video kept together; the timeline and original media remain intact. You can also export a trimmed original before separating sounds.

In the Current command bar, say or type “trim from 3 seconds to 4 seconds” to keep that range, then “export the video” to download it. You can also say “trim 3 to 4 seconds and export as MP4”. The microphone button appears in browsers that support speech recognition and requires microphone permission; recognition is provided by the browser. Natural-language trim replaces the clip sequence (use Undo clip edit to restore it). The source timeline still plays in full; the chosen trim applies to export.

To edit visually, choose **Select range** above the timeline and drag across a waveform. **Play selection** plays the highlighted section, and **Keep selection** sets it as the export trim. **Split selection**, also available by right-clicking a selected waveform, makes clips at both boundaries; use the clip list to reorder or delete the corresponding video and audio together. **Play edited sequence** previews the clip list in order. On separated project layers, **Mute selection** and **Lower selection** change that layer's volume within the highlighted range; the main Undo and Redo buttons restore or reapply those edits. The transitions use short ramps to avoid clicks. These layer edits require a separated project layer so the playback and export use the same saved settings. The source timeline itself remains intact.

For a sequence, use **Add range as clip**, select a clip, and copy/paste, split at the source playhead, delete, or move it. Clip undo/redo buttons cover these sequence edits; the header undo/redo controls cover layer settings. Exports use the clips in the listed order and preserve the audio mix at the original source times. Clearing the list returns to exporting the single trim range. Clip sequences and their history are kept in the open browser session only; they are not restored after a refresh. The waveform uses source time; **Play edited sequence** previews the assembled sequence.
4. The local message panel can test commands such as `Mute the dog` or `Mute the dog from 1 to 2 seconds`. It is a local webhook and outbox, not a messaging delivery service.

### Without Docker

With FFmpeg and Node.js installed (and any Python 3.11+ on the machine):

```sh
scripts/dev_setup.sh          # creates ./.venv from python3.12/3.11/3.13 if no 3.11+ env is active,
                              # installs API requirements + ./vision, downloads detector weights
source .venv/bin/activate     # (or audio-service/.venv if the script chose that one; it prints which)
```

Then three terminals, all from the repo root with the environment activated:

```sh
# 1. audio service (mock separation: stems are filtered copies, no model download)
cd audio-service && AUDIO_BACKEND=mock AUDIO_MEDIA_ROOT=$PWD/../storage AUDIO_PORT=8001 python -m sam_audio_service
# 2. backend, talking to the audio service
STORAGE_ROOT=$PWD/storage INFERENCE_ADAPTER=http SAM_AUDIO_SERVICE_URL=http://127.0.0.1:8001/separate python -m uvicorn app.main:app --port 8000
# 3. editor, proxying /api to the backend
cd frontend && npm ci && VITE_API_BASE_URL=/api npm run dev    # http://localhost:5173
```

To use the built-in fake adapter explicitly, set `INFERENCE_ADAPTER=fake` (stems are plain copies of the audio). An unavailable real separator fails with an error; it does not fall back to fake audio. The script never installs into a conda or system interpreter: it uses the active virtualenv if it is 3.11+, else `.venv`, else `audio-service/.venv`, else creates `.venv`; `PYTHON=/path/to/python` overrides.

`scripts/dev_setup.sh` installs the object-detection package (`pip install -e ./vision`) into the same interpreter that runs the API, because the worker launches the detector as `python -m soundcanvas_cv` with that interpreter. It also downloads the default weights (`VISION_MODEL`, `yolo26n-seg.pt`, about 6 MB) to `$STORAGE_ROOT/vision`. Without the package the editor's object detection is reported as unavailable (`GET /api/projects/{id}/capabilities` → `vision.available: false`) rather than failing a job. Without cached weights the first analysis downloads them, so it needs network access once.

## Use a rented cloud GPU from a Mac

The Mac runs the browser. The Linux GPU machine runs the editor, backend, CPU vision pipeline and NVIDIA AudioSep service together, sharing project files. The GPU service is internal to Compose and isn't exposed publicly. This avoids mounting your Mac filesystem or sending local paths to another host.

1. Rent a Linux x86-64 NVIDIA GPU VM with enough GPU memory and disk for the model, Docker images, uploaded media and outputs. Install Docker Compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and confirm a GPU container can see the device. Open SSH to your trusted IP; keep application ports private.
2. Clone this repository on the VM and start it there. AudioSep uses public, pinned official checkpoints; no gated SAM token is required:

   ```sh
   cd TidePool
   docker compose -f compose.cloud.yaml up --build -d
   docker compose -f compose.cloud.yaml logs -f
   ```

   Initial image builds and model downloads can take a while. The API waits for the audio service's readiness check. Model files persist in a Docker volume; projects persist in the VM's `./storage`. The defaults use AudioSep base 4M, CUDA and 10-second audio chunks, with one chunk window per model batch (`AUDIO_CHUNK_BATCH=1`; batch `k` costs roughly `k x` the activation memory and is unverified on the A40, see [audio-service/docs/chunk_concurrency.md](audio-service/docs/chunk_concurrency.md)); inspect `compose.cloud.yaml` for overrides.
3. On your Mac, create an SSH tunnel, replacing the placeholders with your VM details:

   ```sh
   ssh -L 5173:127.0.0.1:5173 USER@GPU_VM
   ```

   Keep SSH open and visit **http://localhost:5173** on your Mac. The editor uses its same-origin `/api` proxy. Use a separate tunnel to `8000` if you need API docs.
4. When finished, run `docker compose -f compose.cloud.yaml down` on the VM. GPU rental charges may continue until you stop or terminate the VM. Back up `./storage` and any needed Docker volumes before deleting the VM.

The cloud Compose configuration has been checked syntactically; that Compose deployment has not been tested end to end. A separate manual RunPod A40 deployment has run SAM Audio Large and the pretrained audio judge. Check service logs if model access, GPU drivers, memory or container startup fails. Real visual-guided audio isolation quality remains unverified; see [vision/docs/ISSUE4.md](vision/docs/ISSUE4.md). The backend returns inference model identity and warnings so the editor can distinguish mock results.

## APIs and tests

FastAPI serves `/docs` and `/openapi.json`. Project media and edits require `Authorization: Bearer <project token>`; the token is shown once at creation. The UI stores the current project credentials in this tab’s session storage to restore it after a refresh. Copy the project ID and token securely from the Upload panel to resume after closing the tab; anyone with the token can access the project. Jobs and exports run asynchronously; uploads and first preview conversion can take time.

With Python 3.11+, FFmpeg and ffprobe installed, run backend tests using `pip install -r requirements.txt pytest` and `pytest -q`. Run editor checks with `cd frontend && npm ci && npm run verify`. Vision's optional tests are documented in [vision/README.md](vision/README.md). An optional Playwright browser smoke test is in `tests/test_editor_browser.py`; point `SOUNDCANVAS_BROWSER_URL` at a running editor configured with fake inference and install Playwright plus Chromium to run it.

The embedded SQLite worker is intended for one backend process and one replica. Jobs can be retried by submitting a new request after failure; interrupted work is recovered on startup. Untrimmed video export copies the video stream where the MP4 container supports it; trimmed and assembled video is re-encoded to H.264 for accurate boundaries. Audio is encoded as AAC. Incompatible video codecs may cause untrimmed export failure. Automation and remote mute ranges use timeline seconds.

This is a development MVP. Keep project tokens and uploaded media private; use a trusted SSH tunnel or a properly secured HTTPS deployment with authentication, access controls and rate limits. Photon integration is a placeholder pending an identified provider and verified API; local notifications do not send messages to a phone.

## Natural-language commands

Set `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` on the backend to enable GPT-6 Luna in the editor command bar. `OPENAI_AGENT_MODEL` defaults to `gpt-6-luna`. Project authentication is required. The request sends the command, timeline state and stem names to OpenAI; it does not send uploaded media or project tokens. The backend asks for a structured plan and the editor validates each edit before applying it.

Commands can request separation of a sound or edit existing stems. Adding a sound separates the remaining residual while preserving previous layers and their edits. A project needs one unshifted residual stem to add another sound. The optional sound detector can suggest approximate events and intervals; results must be reviewed. Creative requests can produce an audio treatment and selected sound adjustments for preview. OpenAI usage incurs charges and requires model access.

The editor can analyze the first 30 seconds of a video for tracked objects. Select a highlighted object to isolate, mute, or lower a described sound. Visual prompting requires constant-frame-rate, unrotated media whose audio and video both start at zero; Incompatible uploads are rejected before analysis; open Visual selection options and choose Prepare compatible copy to re-encode a separate copy before creating any stems. The uploaded file is retained, and export uses the prepared copy. A 4-second excerpt from a real uploaded video passed visual handoff and export on SAM Audio Large, but the selected visual target was nearly silent, so isolation quality remains unverified. The editor warns for near-silent visual results and provides an original/edited comparison.

### Describe, guide, preview, apply

Describe one sound, optionally mark **I hear it here** by dragging the guidance timeline,
and choose the edit separately. A guidance interval shows where the sound occurs; the
edit interval controls when its isolated layer is muted. For example, “mute the dog from
10 to 15 seconds” preserves other sounds and existing automation.

**Preview** lets you switch between the current mix, proposed edited mix, and isolated
target at the same position without loudness normalization. **Apply preview** accepts it;
**Undo applied preview** restores the prior audio if no intervening edits occurred.
**Quality retry** uses two pretrained judge-ranked candidates when the service supports it.
Judge preference is not confidence in correctness. A time anchor guides overlapping model
windows and cannot establish speaker identity across a long recording.

New sounds are previewed from the current residual. **Try another separation** replaces
a target only in a two-layer project; replacing an older multitrack target requires its
earlier source snapshot. Stale candidates fail safely. Visual guidance remains experimental:
the selected mask is displayed, visibility does not imply audibility, and suspicious results
offer an explicit text-plus-time retry before any edit is applied.

SAM Audio remains opt-in. The default AudioSep profile uses 10-second windows, 2-second
overlap, batch one, one candidate and conservative suppression. Longer windows and independent stereo
channels remain benchmark options. See [guided contracts](INTEGRATION.md#guided-candidates-current-contract)
and [benchmark procedure](audio-service/benchmarks/GUIDED.md).

See [AudioSep setup and limitations](audio-service/README.md#default-audiosep-backend). Its runtime is installed with the `audiosep` extra; official model source and cached weights are prepared separately or by the AudioSep container.

Try **“Make it more dramatic”**, **“Make it cool”**, or **“Emphasize the funny sounds”**. The creative preview can combine a whole-mix treatment with adjustments to locally detected sounds. Review the original and edited mix, disable individual suggested sound adjustments, and apply or discard the preview. Undo restores the previous layers. Sound detection and separation are approximate; the preview does not add music or sound effects and may not match your intent. Funny-sound requests fail clearly when no suitable sound can be isolated.

For a local edit, say **“Remove the chopping at 4.8 seconds”** or mark a processing range. The detector looks near the timestamp and, when it finds a matching supported event, proposes its approximate interval; otherwise the editor uses a short window around the time and reports that fallback. AudioSep processes that window with a little context, and the resulting edit is confined to the selected interval. **Discover sounds** suggests categories and intervals for review; a suggestion is not a separated layer.

The optional YAMNet ONNX detector runs on CPU (`onnxruntime`, model and class map configured separately). Optional local Qwen3-VL scene captions use GPU memory and can inform creative planning; images stay local, while the resulting caption and detected labels may be sent to the configured language model. Captions and detections can be wrong. See [integration details](INTEGRATION.md#creative-director-and-local-sound-discovery).


For repeated effects, try **“Replace every punch with a quack”**. If the detector
cannot find the sound, provide an example time; SoundCanvas can look for similar
sounds, separate short candidate windows, and preview suggested placements. The
experimental text detector and example matching may miss or misidentify events. In
the preview you can listen, disable or nudge individual placements, add a placement
within a processed range, and compare peak and attack alignment. On apply, each added
effect becomes a separate layer. Muting an effect layer later does not restore the
original sound reduction; use Undo to restore the previous mix. See
[repeated-event integration and model setup](INTEGRATION.md#experimental-repeated-event-discovery-and-editing).

Follow up with **“Replace all quacks with loud gongs”** to replace matching effect
layers at their current positions, without another audio separation. The preview
preserves existing layer edits and warns that original video audio was not searched.
Less literal references can use the configured LangChain/OpenAI resolver; it chooses
only among known project effect labels and asks for clarification when uncertain.
