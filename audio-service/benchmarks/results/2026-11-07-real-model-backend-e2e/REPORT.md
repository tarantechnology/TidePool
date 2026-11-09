# Real-model validation through the SoundCanvas backend (issue #1)

Date: 2026-11-07/08. Model: `facebook/sam-audio-small` (1.38 B parameters), float32,
span predictor and re-rankers disabled, 1 candidate, `AUDIO_CHUNK_SECONDS=0` unless stated.
Hardware: Apple M1 Max, 64 GB unified memory, PyTorch 2.15 on **MPS**. This is **not** a CUDA
run; the CUDA acceptance items of issue #1 remain open.

Everything below went through Person 4's real backend (`app/`, `INFERENCE_ADAPTER=http`)
talking to this service over HTTP, except the prompt sweep, which posted to the service
directly. No mock output is reported here.

## Environment notes

- The model cannot be installed on macOS from the `sam` extra (`decord` and `xformers` have no
  Apple Silicon wheels). For this run a throwaway virtualenv was built and the three
  `xformers` imports in `perception_models` were wrapped in `try/except`, as described in
  upstream issue facebookresearch/sam-audio#29. Nothing in the repository was patched.
- `huggingface_hub` had to be `< 1.0` (the upstream loader fails otherwise). Pinned in
  `pyproject.toml` since PR #13.
- Model load: 64 s cold (first run, text/vision encoders downloading), 12–24 s warm.

## 1. Backend → service → stems → edit → export (acceptance item 2)

Input: `demo_argument_siren.mp4`, 8.0 s, 640×360 h264 + 48 kHz stereo AAC. Audio = two real
spoken voices (macOS `say`, Daniel then Samantha, overlapping, panned apart) plus a synthetic
sinusoidal siren (850 ± 350 Hz FM at 0.6 Hz). Ground-truth stems kept for scoring.

Command (service and backend already running):

```sh
python audio-service/scripts/e2e_compose.py --backend http://127.0.0.1:8000 \
    --video demo_argument_siren.mp4 --prompt siren
```

Result: `RESULT: PASS`. Upload → `/process` → job `completed` with stems `siren`, `residual`
(8.0 s each) → stem audio 200 (3.07 MB float32 WAV) → `PATCH` automation → webhook
"Lower the siren volume to 30%" applied → MP4 export `completed`; `ffprobe`: h264 video +
aac audio, 8.000 s. Wall clock 11.4 s with the model warm.

## 2. Stem format, alignment, headroom, reconstruction (acceptance item 3)

| Check | Result |
| --- | --- |
| WAV format stored by Person 4 | IEEE float32, 2 ch, 48 kHz (`format tag 3, 32 bit`) |
| Length | 384 000 samples each = exactly the backend's extracted audio |
| Peaks | target −4.2 dBFS, residual −5.8 dBFS, input −1.2 dBFS; no clipping |
| `sum(stems) − input` | max abs error 1.5 × 10⁻⁸ (float round-off) |
| Timeline | `start_time 0`, full length, `spans` null (no anchors) |

Failure and retry (backend restarted with `INFERENCE_TIMEOUT_SECONDS=3`):

| Case | Job outcome |
| --- | --- |
| 3 s timeout against a ~9 s inference | `failed`, error `Inference timed out; retry the job or increase INFERENCE_TIMEOUT_SECONDS` |
| New `/process` after restoring 3600 s | `completed`, stems `siren`, `residual` |
| 9 prompts (service limit 8) | service returns `400 INVALID_REQUEST`; the backend records httpx's status text (see note) |
| Empty prompt | rejected by the backend with `422 At least one nonempty audio prompt is required for HTTP inference` |

Note: during one attempt the host disk filled while the service spooled the upload, which
surfaced as a 500. That path now raises `STORAGE_FAILURE` (retryable). Person 4's worker still
reads only the HTTP status, so our `error.code` does not reach the job error text (Open
Question 3 in `docs/audio_integration_handoff.md`).

## 3. Processing time and memory (acceptance item 4)

Warm model, text prompt `siren`, direct `POST /separate`, `elapsed_seconds` from the response:

| Clip | Chunking | Elapsed | Processing s per audio s |
| --- | --- | --- | --- |
| 8 s stereo | none | 8.9 s | 1.11 |
| 15 s stereo | none | 17.1 s | 1.14 |
| 30 s stereo | none | 33.9 s | 1.13 |
| 30 s stereo | 10 s windows, 2 s crossfade | 38.8 s | 1.29 |
| 5 s mono (first call after load) | none | 34.6 s | 6.9 (MPS warm-up) |

Memory: service process 2.5 GB resident; `top` reports ~18 GB for the process once the model
is on MPS (unified memory, not visible in RSS). System swap stayed under 1 GB during the
sweep. Note for operators: with less than ~6 GB of free disk the host hit ENOSPC twice during
these runs; keep ≥10 GB free when running the model locally.

## 4. Separation quality on this clip (prompt sweep)

SI-SDR improvement over the mixture, target stem vs. the named ground truth (dB). Higher is
better; ≥10 dB means the wanted sound clearly dominates.

| Prompt | vs siren | vs both voices | vs speaker A (man) | vs speaker B (woman) |
| --- | --- | --- | --- | --- |
| `siren` | −2.6 | −40.1 | −42.5 | −37.8 |
| `police siren` | −5.4 | −49.4 | −47.0 | −48.5 |
| `siren wailing` | 0.5 | −49.2 | −74.2 | −45.4 |
| `alarm` | −2.9 | −48.7 | −49.1 | −46.9 |
| `beeping tone` | −2.1 | −44.7 | −34.2 | −52.9 |
| `man speaking` | −49.0 | **11.1** | 3.1 | 6.3 |
| `woman speaking` | −50.5 | 3.4 | −26.0 | **12.9** |
| `speech` | −17.1 | 6.6 | 2.4 | 4.5 |

Reading: speech prompts work (the woman's voice is isolated at 12.9 dB; "man speaking" pulls
both voices rather than only the lower one). **No siren prompt isolates this clip's siren**:
the 500–1300 Hz band energy stays equal in target and residual, and the siren stem is mostly
speech. The same model scored 11.6 dB on the benchmark's `speech_plus_siren` fixture
(`../2026-11-07-sam-small-mps/`), whose siren was a different synthetic sweep, so the model's
notion of "siren" does not reliably cover pure synthetic FM tones. The negative numbers in
the other columns are expected (the stem is not that source).

Implications for the cinematic demo:

- Use a recorded siren (real harmonics and Doppler) rather than a synthetic tone, and verify
  the stem before the demo.
- Enable the span predictor and `reranking_candidates ≥ 2` on the GPU host; both were off
  here for memory, and upstream recommends them for sound events.
- Try `-base`/`-large`; only `-small` has run.
- Residual is `mixture − targets`, so a speech stem plus its residual still reconstructs the
  original exactly even when the siren stem is poor.

## 5. Still open (needs a CUDA host)

- Build the `sam` Docker target and confirm `/ready` and `/health` report the model and a
  `cuda` device; record GPU, driver/toolkit, memory.
- Repeat sections 1–4 there with `sam-audio-large`, span predictor on, re-ranking on, 20–30 s
  windows, and a recorded siren.
- The real visual-prompt handoff is tracked in issue #4.

## Reproduce

```sh
# service (Linux CUDA: use the Docker target; macOS needs the patched venv described above)
AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-small AUDIO_DEVICE=auto \
AUDIO_LOAD_SPAN_PREDICTOR=false AUDIO_LOAD_RANKERS=false python -m sam_audio_service
# backend
STORAGE_ROOT=/tmp/sc INFERENCE_ADAPTER=http SAM_AUDIO_SERVICE_URL=http://127.0.0.1:8001/separate \
python -m uvicorn app.main:app --port 8000
# flow
python audio-service/scripts/e2e_compose.py --video demo.mp4 --prompt siren
# quality
python -m benchmarks.run_benchmark --backend sam
```
