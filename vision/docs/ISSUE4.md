# Issue #4: project-owned visual/audio handoff

Implementation and local validation, 2026-11-07. **Real visual-guided SAM Audio
separation remains blocked. Keep [Issue #4](https://github.com/AkhilNam/SoundCanvas/issues/4)
open.** A passing mock test proves transport, mask/frame preparation and editing;
it does not establish an association between an object and a sound.

## What was missing and what changed

PRs #12/#13 already implemented detection, tracking, instance masks, the picker,
visual translation, model input preparation and full-timeline audio reconstruction.
The normal `/process` route still sent a selection without its stored detections
or a trusted reference to the original upload. The audio service consequently
could use text/spans but could not resolve that selection into masked video.

The backend now snapshots the stored detections with a schema-1.0 selection and
adds its own `source.original_ref`, `source.sha256`, `source.asset_id`, native FPS,
duration, `asset_id` and job `request_id`. Selection/detection fields are preserved
inside the envelope, including analysis, video, track/detection IDs, timestamps,
interval, dimensions and RLE masks. File references come exclusively from the
authenticated project's original upload. Both services must see the same absolute
storage path; the existing Compose overlay already mounts `/data` appropriately.

The backend validates source/analysis identity, the clicked observation, dimensions,
FPS, frame-aligned interval, complete native-frame coverage and instance-mask RLE.
It rejects client media references recursively, conflicting inline detections,
sparse records, missing/invalid masks and fabricated selections. A recorded frame
without the track is valid absence; its foreground stays empty. Track identities
are never bridged across interruptions. Upload hashes are rechecked before queued
inference. Rotation, VFR and nonzero stream origins require normalization before
upload/analysis. Analyze and upload the **same normalized bytes**.

The existing audio-service translator/resolver and SAM backend remain responsible
for decoding frames, mask polarity and inference. Frame-count disagreement now fails
instead of continuing with potentially shifted masks. The backend rejects cropped,
offset, wrong-rate or wrong-channel visual results. Full-timeline stems retain
offset zero; volume/mute/solo/export continue through the existing editor API.
Jobs expose `inference.model`, `warnings`, `source`, `request_id` and contract version
so fake/mock/degraded results can be identified. Historical job-output arrays still
read correctly. Legacy label/time-range metadata continues to pass through unchanged;
the stricter validation applies to actual schema-1.0 selections.

No frontend editor is present in current main. The standalone browser picker and
the backend's editing/export routes were exercised. At least one nonempty sound
description is still required by the existing HTTP `/process` contract; a visual
selection does not invent a verified sound label.

## Executed validation

From repository root, using the existing `.venv`:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -c audio-service/pyproject.toml audio-service/tests
SC_CV_REAL_TESTS=1 SC_CV_BROWSER_TESTS=1 \
  .venv/bin/python -m pytest -q -c vision/pyproject.toml vision/tests
```

- Backend: **55 passed**. Includes project/source/analysis identity, malicious media
  references, malformed selections/masks, sparse coverage, genuine analyzed track
  absence, changed upload bytes, old upload metadata, unsupported timelines,
  one-sample-short/offset/wrong-rate/wrong-channel returned stems, and service errors.
- Audio service: **153 passed**. Includes a live localhost HTTP service, actual
  FFmpeg frame/mask preparation, target/residual reconstruction and edited export.
  Added native-frame RGB/mask correspondence at **30000/1001 FPS**, plus fail-fast
  coverage of decoded frame-count mismatch. The inference backend is **mock**.
- Vision: **35 passed**, including actual CPU YOLO26n and YOLO26n-seg inference on
  one second of the real pedestrian clip, tracker reset, native masks, actual
  Chromium clicks/downloads/hash rejection, and real-YOLO metadata through the backend.
  The documented pedestrian clip has **silence**, so it cannot measure isolation.
- All suites emit one existing FastAPI/Starlette TestClient deprecation warning.

Alignment evidence from `test_backend_visual_masks_and_export`: selected interval
`[0.2, 0.8)` has 18 frames at 30 FPS; returned stems each have **48,000 samples**.
Target samples outside that interval are zero; target + residual reconstruct the
source within `1e-7`. The MP4 has **30 video frames, duration 1.000000 s, 48,000
decoded audio samples**, and both streams start at **0.000000 s**. Its decoded audio
matches `0.5 * target + residual` with measured RMSE **0.0007871** (AAC is lossy).
WAV mute/solo exports match the intended mix within PCM16 quantization.

A separate live two-process run of `examples/visual_audio_flow.py` with the existing
12-second real YOLO segmentation artifacts exported H.264/AAC, **120 video frames**,
**12.000000 s** on both streams and **576,000 samples** per stem, all starting at zero.
Local, ignored artifacts: `vision/outputs/issue4-live/{edited.mp4,stem-0.wav,
stem-1.wav,report.json,alignment.json}`. Its model is explicitly **mock**, and its
pedestrian input is silent. These artifacts demonstrate the workflow, not quality.

## Real SAM Audio diagnosis

Active interpreter: `<repository>/.venv/bin/python`, Python **3.12.1**, base interpreter
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`, macOS 26.3 arm64,
Apple **M2**, **8 GiB unified RAM**, approximately **12 GiB free disk**, PyTorch
**2.9.1**, **MPS available**, **CUDA unavailable**. This venv was created with
`include-system-site-packages=true`; no installed packages were changed.

`sam-audio-service` and `soundcanvas-cv` are installed editable. `sam_audio`,
`torchaudio`, `torchcodec`, `huggingface_hub`, `transformers`, `decord` and `xformers`
are absent. `sam_audio` is the upstream model package, distinct from this repository's
`sam_audio_service`. It belongs to `audio-service/pyproject.toml`'s optional **`sam`**
extra, not the base/mock/dev install. The immediate failure is an uninstalled
optional runtime, not an interpreter mismatch.

An offline call of the **existing** `scripts/smoke_sam.py`, with
`facebook/sam-audio-small`, `mps`, `float32`, `--no-span-predictor --no-rankers`, fails:

```text
MissingModel: sam_audio is not importable
(ModuleNotFoundError: No module named 'sam_audio'); install the service
with the 'sam' extra. The processor also needs torchcodec.
```

A real-backend `/separate` preflight carrying the video/selection/detections returned
**503 `MISSING_MODEL`**, `/health` reported **`loaded=false`**, and no inference ran.
`model.is_mock=false` in an unloaded health response is **not** evidence of inference.
Machine-readable local evidence: `vision/outputs/issue4-sam-preflight/report.json`.
The silent clip was used only to reach the loader; no sound-quality claim is possible.

Bounded installation feasibility checks used a clean, isolated scratch venv:

```bash
.venv/bin/python -m venv /private/tmp/soundcanvas-sam-preflight-20261107
/private/tmp/soundcanvas-sam-preflight-20261107/bin/python -m pip install \
  --dry-run --no-deps --only-binary=:all: --no-cache-dir decord==0.6.0
```

Result: **`No matching distribution found for decord==0.6.0`**. The current upstream
[SAM dependency declaration](https://github.com/facebookresearch/sam-audio/blob/main/pyproject.toml)
uses `perception_models@unpin-deps`, whose
[requirements](https://github.com/facebookresearch/perception_models/blob/unpin-deps/requirements.txt)
still include `decord` and `xformers` unconditionally. No full dependency install,
source build, checkpoint download or GPU job was attempted.

Unlike the earlier team note, today's wheel query finds a small generic
`xformers-0.0.35.dev1148-py3-none-any` wheel. Its metadata requires **torch>=2.10**,
NumPy and `mslk`; its presence alone does not prove macOS model compatibility.
That wheel was inspected in `/private/tmp` without installation. The primary
confirmed blocker is `decord`; additional xformers/runtime compatibility was not
claimed or patched. The scratch venv still contains only pip.

The team's [successful M1 Max report](../../audio-service/benchmarks/results/2026-11-07-sam-small-mps/REPORT.md)
used Python 3.12.6/MPS/float32, small, no span predictor/rerankers and an explicitly
patched scratch environment. Its text-only SI-SDR improvements were 11.60/10.19/12.94 dB
on synthetic siren/speech/music fixtures. It contains **no visual experiment**, no
reusable environment lock/patch bundle, and no measured small-plus-vision memory
budget for this 8-GiB machine. The existing
[upstream MPS discussion](https://github.com/facebookresearch/sam-audio/issues/29)
documents removing decord and modifying Perception's xformers imports. Dropping
the visual encoder to make a text-only port work would defeat this validation.

Given the confirmed dependency blocker, no cached checkpoint, the 8-GiB memory
budget and several-GB model/dependency footprint, a patched native install was not
justified here. Existing `pip check` also reports inherited NumPy conflicts for
mediapipe, pytorch-kinematics and opencv-python-headless; these predate this work and
were left alone. No global or working-venv installation was performed.

## Reproduce the integrated workflow

Use the setup in `vision/README.md` and `audio-service/README.md` for optional
dependencies. With the existing demo video/weights already downloaded, native-FPS
segmentation can be generated with:

```bash
cd vision
../.venv/bin/python -m soundcanvas_cv --input examples/media/pedestrians.mp4 \
  --model yolo26n-seg.pt --output-dir outputs/issue4-seg \
  --sample-fps 10 --max-seconds 12 --device cpu
cd ..
```

Open `vision/viewer.html`; load the original video and `issue4-seg/detections.json`,
click an object, adjust the frame-aligned interval if desired, and export the
selection to `vision/outputs/issue4-seg/selection.json`. This is the existing browser
picker. Use a real audible-source clip for model evaluation; the pedestrian demo
is only a silent integration fixture. For other videos, export metadata at their
**native** FPS. Normalize VFR/rotation/nonzero timelines **before** analysis/upload.

Start the two existing services in separate terminals from repository root:

```bash
AUDIO_BACKEND=mock AUDIO_MEDIA_ROOT="$PWD/storage" AUDIO_PORT=8101 \
  .venv/bin/python -m sam_audio_service
```

```bash
STORAGE_ROOT="$PWD/storage" INFERENCE_ADAPTER=http \
SAM_AUDIO_SERVICE_URL=http://127.0.0.1:8101/separate \
  .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

Submit the actual picker export and edit its target:

```bash
.venv/bin/python vision/examples/visual_audio_flow.py --url http://127.0.0.1:8100 \
  --video vision/examples/media/pedestrians.mp4 \
  --detections vision/outputs/issue4-seg/detections.json \
  --selection vision/outputs/issue4-seg/selection.json \
  --prompt 'person speaking' --volume 0.25 --output-dir vision/outputs/issue4-live
ffprobe -v error -show_streams -show_format -of json vision/outputs/issue4-live/edited.mp4
```

The example creates/uploads an authenticated project, stores detections, submits
the frozen selection, polls, downloads target/residual, patches target volume and
exports MP4. It keeps the token in memory and excludes it from reports/logs. All
media and generated artifacts remain ignored. `--muted` and `--solo` exercise the
same existing edit API. `report.json` includes model identity/warnings and selection
IDs; **MOCK** output is labeled explicitly.

## Remaining acceptance gate and recommended next execution

Run on the teammate's working patched M1 Max environment, or the existing
Linux/CUDA `sam` deployment with approved model access and an adequate memory
budget. Keep `facebook/sam-audio-small`, float32, `AUDIO_LOAD_SPAN_PREDICTOR=false`,
`AUDIO_LOAD_RANKERS=false`, `AUDIO_PREDICT_SPANS=false`,
`AUDIO_RERANKING_CANDIDATES=1`. Preserve the visual encoder and shared storage path.
The Docker overlay requires explicit `-e` overrides on `docker compose run` for
the load flags; it does not forward every shell `AUDIO_*` variable automatically.
Do not build/download weights on this Mac to simulate a CUDA host.

Before any several-GB download, confirm model access, available RAM/VRAM/disk and
authorization. Use a consented/licensed short clip of a visibly audible source
with an unrelated competing sound. Analyze and click that exact file, then run
the integrated workflow against the real service. Require `/ready=200`, a completed
response with `model.is_mock=false`, saved stems, and actual listening/evaluation.

For a fair comparison, use the existing `audio-service/scripts/smoke_sam.py` on
the **same interval**, prompt, seed and model settings: text only versus
`--video ... --frame-masks ... --frame-rate ... --mask-mode both`. Generate the
frame-mask JSON from the chosen track's native RLE observations (empty for analyzed
absence). Compare target intelligibility, unrelated leakage and visual benefit or
regression against the original mixture. Measure runtime and peak memory; measure
SI-SDR/leakage only where isolated reference sources exist. Recheck full-timeline
reconstruction and edited MP4 alignment through the backend afterward.

**Acceptance:** schema examples, unchanged selection/detection fields through the
authenticated boundary, selected-object frame/mask preparation, editing and export
are verified. Real visual-guided isolation and any improvement over text remain
**unverified**. The previous real text benchmark and the mock tests cannot close
that gate; this PR must not auto-close Issue #4.
