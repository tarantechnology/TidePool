# SoundCanvas Part 2 — Vision

**Runpod update (2026-11-13):** official SAM 3.1 and SAM 2.1 video inference now run on the L4, preserving schema 1.0 and YOLO fallback. Real SAM Audio Small TV text/visual/temporal comparisons also ran. [GPU results, adapters and reproduction](docs/RUNPOD.md) distinguish execution from unverified isolation quality.

Independent, working CV pipeline: MP4/image decoding, real YOLO detection, ByteTrack tracking, optional YOLO instance masks, click selection and versioned JSON. This package does not separate audio. See [validation evidence](docs/VALIDATION.md), [integration contract](docs/CONTRACT.md) and [SAM feasibility](docs/SAM.md).

[Issue #4 integration](docs/ISSUE4.md) adds project-owned video resolution, stored-mask/selection pairing, job diagnostics and an upload → edit → MP4 example. The historical report includes real YOLO/browser and mock-audio integration tests and the earlier local dependency blockers. The Runpod results linked above now establish real visual-guided model execution; isolation quality remains unverified.

## Setup

From the SoundCanvas repository root, with Python 3.11 or 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e './vision[dev]'
# macOS:
brew install ffmpeg
# Ubuntu instead: sudo apt-get update && sudo apt-get install -y ffmpeg
python -m playwright install chromium
cd vision
```

The backend and vision dependencies are separate; root `requirements.txt` and the backend Docker image do not install YOLO. On Windows use `.venv\Scripts\activate`, and install FFmpeg on PATH. CPU is the validated device. `--device mps` is available but was not benchmarked.

## Reproduce real inference

```bash
python examples/download_demo.py
python -m soundcanvas_cv --input examples/media/pedestrians.mp4 --model yolo26n.pt \
  --output-dir outputs/real_yolo --sample-fps 5 --max-seconds 12 --device cpu
# Real instance masks at every native 10-FPS frame:
python -m soundcanvas_cv --input examples/media/pedestrians.mp4 --model yolo26n-seg.pt \
  --output-dir outputs/real_seg --sample-fps 10 --max-seconds 12 --device cpu
# Runs and validates detection-only, tracking, segmentation and controlled dropout tests;
# also writes aligned visual tensors and an explicitly mocked handoff:
python -m examples.validate_real
# Or validate already generated outputs (still runs controlled tracking experiments):
python -m examples.validate_real --reuse
```

Pretrained weights download automatically if absent. The previously downloaded `yolo26n.pt` can also be supplied with an absolute `--model` path. `--no-track` reduces inference to the requested sample rate. Tracking runs on **every decoded frame**, so lowering sample FPS reduces JSON density but does not reduce tracking inference cost. `--tracker botsort.yaml` is an alternative; ByteTrack is the tested default. `--video-id` accepts a backend-owned media ID; otherwise identity is the source SHA256.

The test video is real pedestrian footage from OpenCV's pinned `vtest.avi`: [source, commit 5728f796](https://github.com/opencv/opencv/blob/5728f796f6860246134066b4338c31c612adde77/samples/data/vtest.avi). Downloaded AVI SHA256: `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`. The script verifies that digest, transcodes its first 12 seconds to 768×576 H.264 at 10 FPS, and adds **silence** for backend upload compatibility. It records provenance in `examples/media/source.json`. No original sounds, arguing scene or siren are present. No additional license for the footage is asserted; media is downloaded locally, not redistributed in git.

Outputs include `annotated.mp4` (silent MPEG-4 Part 2 preview), `preview.jpg`, `detections.json`, `outputs/validation_report.json`, and, for the mask demo, `visual_prompt.npz`, `selection.json`, `mock_handoff.json` and `masked_preview.jpg`. MP4 preview playback depends on browser codecs; the picker uses the H.264 **original** video. Downloaded media, weights and generated outputs are ignored, as required by CONTRIBUTING.md.

## Click-to-select

Open `viewer.html` directly in a modern browser; no frontend build or backend is required. Choose `examples/media/pedestrians.mp4` and `outputs/real_yolo/detections.json`. For mask selections choose `outputs/real_seg/detections.json` instead. Play/seek and click a box. Playback pauses, JSON shows the frozen observation, and **Export selected object JSON** saves it. Loading different media or invalid metadata clears the previous selection.

The viewer verifies SHA256, decoded dimensions and processed duration. It accounts for letterboxing and prefers the smallest overlapping box, then highest confidence. Boxes are held causally only until the next sampled observation, and disappear at the processing boundary. An exported click records both the observation timestamp and the click time; seeking later does not alter it. Fullscreen native video hides the overlay; use the inline player for selection. Video/metadata validation hashes the whole local file into browser memory, so this viewer is a small-clip demo, not the upload UI.

For a selection made against **the same segmentation run**, prepare CV inputs:

```bash
python -m soundcanvas_cv.prepare --video examples/media/pedestrians.mp4 \
  --metadata outputs/real_seg/detections.json \
  --selection outputs/real_seg/selection.json --output outputs/prepared.npz
```

The validation script creates that example selection and extends its interval to the full clip explicitly. Viewer clicks initially cover only one sampled interval. To prepare longer clips, an integrating editor may deliberately extend `interval` to native frame boundaries. Masks must exist at every native frame in that interval; missing observations of the selected track become empty foreground. Sparse metadata and bounding-box-only prompts are rejected. `--allow-box-masks` enables a clearly labeled coarse rectangular experiment, with unmeasured audio quality.

## Tests

From repository root after setup:

```bash
python -m pytest -q                         # existing backend tests
python -m pytest -q -c vision/pyproject.toml vision/tests -m 'not real and not browser'
SC_CV_REAL_TESTS=1 SC_CV_BROWSER_TESTS=1 \
  python -m pytest -q -c vision/pyproject.toml vision/tests
```

The last command requires real demo outputs, both weights, FFmpeg and Playwright Chromium. Backend boundary tests use real routes, SQLite, FFmpeg and the worker, with a **mocked** external service returning a copied WAV. They assert exact JSON forwarding, not separation quality. One optional test executes the inspected upstream `mask_videos` method if its source is locally downloaded; otherwise that test reports a skip. See [SAM.md](docs/SAM.md) for its pinned source command. Node enables a JS/Python hit-test parity check; without Node that check is explicitly skipped.

GitHub runs deterministic vision tests separately from backend CI. Manually dispatch **Vision tests** to download the demo, run real inference and Chromium, and collect temporary artifacts. Default tests do not download models or footage.

## Completed and pending

- [x] Real MP4 loading; YOLO classes, confidence, boxes and frame timestamps; annotated outputs.
- [x] Versioned JSON and selection schema; frontend/backend contract and tested pass-through.
- [x] Click selection, frozen export, letterboxing, source mismatch detection and interval bounds.
- [x] Persistent tracking with per-analysis scope; controlled loss/reappearance experiments.
- [x] Optional real YOLO instance masks and aligned CV tensors for SAM Audio preparation.
- [x] Official SAM 3 / SAM Audio feasibility review; explicit mock integration.
- [ ] SAM 3 inference and quantitative comparison: CUDA/access/checkpoint blockers.
- [ ] Full SAM Audio processor/model run and visual-guided audio isolation: no model/service locally; requires Person 1.
- [ ] Shared original-video resolver for the existing WAV+JSON audio HTTP boundary; cross-team approval of schema/model revision remains pending.
- [ ] Natural occlusion identity accuracy benchmark, automatic sound association and similar-speaker separation.

Only visible objects are detected. Track IDs can change, predicted classes can be wrong, and a detection does not establish a sound source. CFR timing is supported; normalize variable frame rate, edit lists, nonzero audio/video offsets and rotation before processing/upload so all workstreams share the same timeline. OpenCV's decoded orientation/dimensions are authoritative; no explicit VFR/PTS reconciliation is implemented. Masks are model estimates, not ground truth. License terms for Ultralytics and any future SAM deployment require the team's own review.

## Choosing a visual isolation interval

Generate tracking and segmentation metadata at the video’s native frame rate (for example, use `--sample-fps` at least as high as the source FPS). Open the original video and that `detections.json` in the viewer, select an object, and edit the start and end seconds if needed. The default covers approximately two seconds around the selected observation. Boundaries must be native-frame aligned, contain the observation, stay inside the processed footage and span 0.1–30 seconds. Export is disabled with an explanation if the interval is invalid, frames are missing, the selected object is untracked, or an observed object lacks an instance mask. A frame with no observation of the selected track is treated as absence only when that frame was actually analyzed. The range does not assert that the object makes a sound.

These limits match the audio service’s current defaults; a separately configured service may impose a lower maximum. The selection is only one part of the cross-team handoff: original video and matching detections still need to be supplied to the audio service as described in [the contract](docs/CONTRACT.md).
