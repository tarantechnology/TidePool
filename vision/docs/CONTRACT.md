# CV integration contract 1.0 (proposed for teammate review)

This specializes the backend's existing `visual-metadata` / `visual_prompt` objects. Schema-1.0 selections now receive identity/mask validation and a server-owned video reference; legacy label/time-range objects retain pass-through behavior. Issues [#3](https://github.com/AkhilNam/SoundCanvas/issues/3) and [#4](https://github.com/AkhilNam/SoundCanvas/issues/4) track coordination; documentation is not evidence of approval by Person 1. See [current implementation/validation](ISSUE4.md).

## Identity, coordinates and time

`detections.json` follows [detections.schema.json](../schemas/detections.schema.json). Selection exports follow [selection.schema.json](../schemas/selection.schema.json). Existing `source`, `width`, `height`, `fps`, `frames`, `objects`, `label`, `object_id` and bounding-box keys are preserved. `class_name` aliases `label`; `frame_width/height` alias `width/height`. Legacy metadata lacking version/identity fields should be regenerated for this viewer.

| Field | Meaning |
| --- | --- |
| `schema_version` | `1.0` for both documents |
| `video_id` | Backend media ID passed via `--video-id`; default `sha256:<source digest>` |
| `source_sha256` | SHA256 of exact original input bytes; filenames alone are not identity |
| `analysis_id` | Unique run ID; scope track/detection identities by `(video_id, analysis_id)` |
| `frame_index` | Zero-based decoded source-frame index; skipped frames retain their source indices |
| `timestamp_seconds` | Observation time, seconds from first decoded frame, `frame_index / nominal fps` for CFR |
| `clicked_at_seconds` | Playhead time when clicked, possibly later than the observation |
| `bbox_xyxy` | Integer decoded pixels `[left, top, right, bottom]`; top-left origin, right/bottom exclusive; clamped to dimensions |
| `bbox_normalized` | Same pixel box divided by width/height, rounded to six decimals |
| `detection_id` | Unique within one analysis (`frame:<index>:det:<index>`) |
| `track_id` | Nullable integer from ByteTrack/selected tracker; can change after occlusion |
| `object_id` | Legacy `track:<id>` or frame-local `detection:<index>`; use `detection_id` for individual observations |
| `valid_until_seconds` | Next sample time or processed boundary; half-open hold interval `[timestamp, valid_until)`; no extrapolation past it |
| `mask` | Null, or native `H×W` binary instance mask encoded as below |

`sampling_fps` is the requested export rate capped at native FPS; noninteger frame/sample ratios yield alternating gaps rather than accumulated drift. Actual timestamps/indices are authoritative. Frames whose start time is below `max_seconds` are retained, so the processed end is quantized upward by less than one native frame. `analyzed_frames` is a legacy count of exported samples; `inference_frames` reports real detector calls. Tracking analyzes every native frame even when JSON is sampled. Preview video uses current tracker boxes each frame; sampled viewer boxes may lag between samples. An image has one observation at zero, null FPS/duration and null interval end; video-audio preparation rejects images.

The selection `interval` defaults to approximately two seconds around the observation, clipped to the processed video and aligned to native frame boundaries. The viewer lets the user edit this requested **visual processing interval** independently of the frozen observation; it is not evidence that the selected object produced audio throughout it. The viewer requires an interval of 0.1–30 seconds containing the observation, a tracked object and native-FPS segmentation metadata throughout. It refuses sparse or box-only data for visual export. The limits match the current audio service defaults and must be revisited if that service is configured with a different maximum. Do not bridge identity changes automatically. A missing selected-track observation yields empty foreground; it does not reuse the last mask or claim continuous identity.

## Masks

Example encoding, purely illustrative:

```json
{"encoding":"rle","order":"C","size":[2,3],"counts":[1,2,2,1],"foreground_value":1}
```

Flatten in row-major order; alternating runs of zero/one start with zeros, including a zero-length first run when needed. Counts sum to height × width. Here the rows are `[0,1,1]` and `[0,0,1]`. This is **not compressed COCO RLE** and not a SAM tensor. `encode_mask` / `decode_mask` are the reference codec. YOLO uses `retina_masks=True`, exports native-resolution masks, and refuses unexpected shapes rather than distorting letterboxing.

## Existing backend handoff

From the repository root, once `PROJECT_ID` and `PROJECT_TOKEN` identify an uploaded project:

```bash
curl --fail-with-body -X PUT "http://localhost:8000/api/projects/$PROJECT_ID/visual-metadata" \
  -H "Authorization: Bearer $PROJECT_TOKEN" -H 'Content-Type: application/json' \
  --data-binary @vision/outputs/real_seg/detections.json
```

A subsequent `/process` with no explicit `visual_prompt` forwards legacy metadata unchanged. Store native-FPS segmentation detections, then submit a specific frozen selection:

```python
import json, os, httpx
selection = json.load(open('vision/outputs/real_seg/selection.json'))
response = httpx.post(
    f"http://localhost:8000/api/projects/{os.environ['PROJECT_ID']}/process",
    headers={'Authorization': 'Bearer ' + os.environ['PROJECT_TOKEN']},
    json={'prompts': ['person speaking'], 'visual_prompt': selection},
)
response.raise_for_status()
print(response.json())  # job ID; poll existing /api/jobs/{id}
```

The worker POSTs `audio` (48kHz stereo WAV) plus JSON-string `metadata`. For a validated selection, `visual_prompt` is an envelope containing the unchanged selection and snapshotted stored detections; the server adds `source.original_ref` (a file URI under the project's upload directory), source SHA256, FPS/duration and project asset ID. No client media references are accepted. The audio service resolves the video under `AUDIO_MEDIA_ROOT`, uses its existing visual translator/frame-mask preparation, crops inference and reconstructs full-timeline target/residual stems. The backend checks zero offset and full extracted-audio duration/rate/channels before exposing stems. Both services require the same absolute storage mount. Jobs expose model/mock status and warnings under `inference`. [ISSUE4.md](ISSUE4.md) details validation, setup and the remaining real-model gate.

CV exports do not contain local absolute media paths or tokens. The `detector` field records model name, weight digest, thresholds, device and tracker; `runtime` records environment and pipeline seconds. Detection/selection IDs are machine references, not verified physical/person identities.

## Local SAM Audio preparation boundary

`prepare_visual_inputs(video_path, metadata, selection)` returns RGB `uint8` frames `N×3×H×W` and binary foreground masks `N×1×H×W`, FPS and explicit start/end seconds. It checks exact source hash, analysis identity, observed selection fields, dimensions, native frame alignment and complete frame metadata. No sparse-mask interpolation occurs. Full-resolution tensors for long clips can use substantial memory; this helper is a short-clip prototype.

`build_processor_batch(prepared, audio_48khz, processor)` accepts Person 1's already cropped/resampled mono/stereo tensor `C×T`, checks its length against the visual interval, inverts foreground masks for official `mask_videos`, then calls `processor(audios=[audio], descriptions=[''], masked_videos=...)`. Sampling rate cannot be inferred from a bare tensor; the caller owns actual resampling. This prepares a batch only; no audio model is loaded or called. `mocked_handoff` always sets `mock: true`, `inference_performed: false`, and returns zero stems.
