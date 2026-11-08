# Part 2 validation evidence

Validated November 6, 2026 on Apple M2, macOS 26.3 arm64, Python 3.12.1, PyTorch 2.15.0, Ultralytics 8.4.204, OpenCV 4.14.0, ByteTrack via lap 0.5.14. All model experiments below used **CPU**. MPS was available but not benchmarked; CUDA was unavailable.

## Real media and inference

Input: 12 seconds of real moving pedestrians and parked vehicles, 768×576, 10 FPS, 120 decoded frames. See [download provenance and commands](../README.md). The added audio is silence. Missing `test.mp4` was an input-file problem; inference succeeds with this valid MP4.

| Run | Native frames / inference calls / JSON samples | Actual observations (repeated across frames) | Latest measured pipeline time |
| --- | --- | --- | --- |
| YOLO26n + ByteTrack, exports 5 FPS | 120 / 120 / 60 | 291 person, 60 car, 60 truck, 28 bird | 6.50 s |
| YOLO26n detection only, 5 FPS | 120 / 60 / 60 | 300 person, 60 car, 60 truck, 28 bird | 2.94 s |
| YOLO26n-seg + ByteTrack, exports native 10 FPS | 120 / 120 / 120 | 585 person, 120 car, 120 truck; **825 nonempty instance masks** | 9.73 s |

These are repeated observations, not numbers of unique physical objects. The first tracked run took 10.14s and the first segmentation run 12.15s; warm runs varied. Pipeline time includes decode, inference, tracking, annotations and mask encoding; excludes weight download/model construction and final JSON writing. Different inference/sample workloads are shown explicitly. There is no SAM 3 runtime measurement or accuracy ranking.

Both real model weights were used, with SHA256:

- `yolo26n.pt`: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- `yolo26n-seg.pt`: `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`

All detections were checked for valid confidence, nonempty pixel boxes within decoded dimensions, consistent normalized coordinates, unique per-frame detection IDs and correct CFR timestamps. Every annotated MP4 decoded back to 120 frames. Native masks were RLE-decoded and checked for dimensions/nonempty foreground. This establishes functioning inference and serialization, not model accuracy on labeled ground truth.

Visual inspection of the annotated first frame and a 0/3/6/9-second overview confirmed actual pedestrian and vehicle boxes. Inspection also found a low-confidence **false bird detection on light-colored material lying on the grass**, illustrating why class/confidence values need user review. YOLO box/track output should never be treated as proof of sound production.

## Tracking behavior

Multiple moving people are tracked simultaneously. The detection run emitted 12 person track IDs, and segmentation emitted 14; these are not verified person counts. Detection track #1 reaches the right edge and stops after 2.2s. Person track #9 has 54 exported observations from 1.2s through 11.8s. Other tracks have gaps; overlapping pedestrians can cause missed detections/fragmentation. No ground-truth ID-switch or natural-occlusion accuracy score was measured.

Controlled additional experiments ran **real YOLO inference** on a repeated real first frame, inserted completely black frames, then restored the image:

| Artificial missing frames | Person IDs before | Person IDs after | Result |
| --- | --- | --- | --- |
| 5 | 1, 2, 3 | 1, 2, 3 | All three IDs retained |
| 40 | 1, 2, 3 | 6, 7, 8 | IDs changed after tracker expiration |

This tests temporary loss and reappearance under controlled conditions, not realistic moving-person occlusion. IDs are scoped to `(video_id, analysis_id)`, reset for every processing call and can be wrong. Short missing detections do not create invented detections or masks.

## Automated and browser evidence

- **33 vision tests passed**, including opt-in actual detection/segmentation, real metadata through backend routes/worker, Chromium, and the actual upstream SAM Audio masking method. The upstream masking-method test is optional when its downloaded source is absent; it ran locally. Node parity checks also ran.
- **4 existing backend tests passed** using FFmpeg/ffprobe. One upstream Starlette/httpx deprecation warning remains; it did not fail tests.
- Sampling tests cover 29.97 FPS vs 7 FPS (source indices 0, 5, 9, 13, 18, 22, 26 over the first second), above-native sample requests, truncation, native-frame tracking and per-video reset.
- Invalid input, corrupt media, NaN/infinite/nonpositive rates, output/source collision, stale timestamps, bbox/mask errors, mismatched sources/analyses and tampered selections are covered.
- Chromium decoded the real H.264 video, clicked a real YOLO person box, matched the Python selection payload, downloaded it, and verified the original observation stayed frozen after seeking. Invalid JSON and wrong-source hashes disable selection. A screenshot is saved locally as `outputs/real_yolo/viewer.png`.
- Real routes, SQLite, FFmpeg extraction, queue submission and worker dispatch were exercised for stored visual metadata and explicit selection overrides. Exact JSON reached the external HTTP boundary. The external service was **mocked**, returning a copied WAV named as a mock; this is not sound isolation.

## SAM Audio preparation and remaining work

The segmentation demo selected its longest-observed person track (#7 in that run), explicitly requested the full 0–12s interval, and prepared RGB frames `[120,3,576,768]` plus masks `[120,1,576,768]`. It decoded actual source frames and actual YOLO instance masks; absent target observations are empty foreground. The local NPZ is a CV artifact. Mock handoff JSON states `inference_performed: false` and contains no stems.

Official SAM Audio mask polarity and shapes were verified against source, and the actual upstream `mask_videos` method ran with tensors. **Full SAM Audio processing/model inference, audible-source association, and isolated audio stems were not validated.** SAM 3 was not run. Local package/checkpoint absence, anonymous HTTP 401 and no CUDA were verified; see [SAM.md](SAM.md). The existing audio service receives only WAV+JSON; video resolution and timeline alignment require agreement with Persons 1/4.

## Reproduce and inspect

Use the exact commands in [README](../README.md). `python -m examples.validate_real` regenerates the runs, controlled tracking experiments, selection, masked preview, NPZ and machine-readable `outputs/validation_report.json`. All generated media/exports, downloaded research sources and model weights remain local and ignored, per CONTRIBUTING.md. Source/provenance and this measured report are committed as documentation. Manually dispatch the Vision tests workflow for temporary CI artifacts; no SAM access is required for YOLO tests.
