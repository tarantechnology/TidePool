# Person 2 (vision) artefacts — real, not synthetic

These three files were produced by **Person 2's own code**, running real YOLO
instance segmentation and ByteTrack tracking on real pedestrian footage. They are the
input of cross-team **Test 2 (Vision -> Audio)**, exercised by
`audio-service/tests/test_person2_artifacts.py`.

Nothing in this directory was hand-written, edited or synthesised, and **no SAM Audio
inference was run**: the test drives this service with `AUDIO_BACKEND=mock`, whose stems
are band-pass filtered copies of the mixture, not separation. The mock backend also
*ignores* the visual input it is handed, so these artefacts prove the **translation and
preparation** path (documents -> `VisualInput` -> frames/masks tensors -> HTTP contract),
not audio quality.

| File | Bytes | What it is |
| --- | --- | --- |
| `detections.json` | 1 391 807 | Person 2 *detections 1.0*: 60 native frames, per-frame RLE instance masks |
| `selection.json` | 3 343 | Person 2 *visual selection 1.0*: the clicked object (track 7, `person`) |
| `pedestrians.mp4` | 847 144 | The exact video bytes both documents were computed on |

Total 2.1 MB, so the clip is committed next to the JSON (the 3 MB fixture budget holds)
and `fixtures/generated/vision/` is unused.

## Provenance

| | |
| --- | --- |
| Person 2 branch | `origin/vision/validated-pipeline` |
| Commit used | `0f5654f` ("Use CPU PyTorch wheels for vision CI"); pipeline added in `b12bed0` |
| Package | `soundcanvas-cv` 1.0.0 (`vision/soundcanvas_cv`), installed with `uv pip install -r vision/requirements.txt -e vision` |
| Detector | `ultralytics` 8.4.204, `torch` 2.15.0, `opencv-python` 4.14.0, CPython 3.12.6, device `cpu` |
| Weights | `yolo26n-seg.pt` (the file `vision/README.md` names for the mask demo), SHA256 `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`, downloaded from `https://github.com/ultralytics/assets/releases/latest/download/yolo26n-seg.pt` |
| Tracker | `bytetrack.yaml` (their tested default), confidence 0.30, `retina_masks=True` |
| Analysis id | `42b310c52376427593e07ce4002dc54c` |
| `video_id` | `sha256:8e17935a332c7453937b5f18a7500c648a1f7853639f25b39e456ca2b8c8a94b` |
| `source_sha256` | `8e17935a332c7453937b5f18a7500c648a1f7853639f25b39e456ca2b8c8a94b` — verified against the committed `pedestrians.mp4` |

### Video

`pedestrians.mp4` is produced byte-for-byte by Person 2's `vision/examples/download_demo.py`:
OpenCV's pinned sample `vtest.avi` (revision `5728f796f6860246134066b4338c31c612adde77`,
AVI SHA256 `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`), first 12 s
transcoded to 768x576 H.264 at 10 FPS with an **added silent** 48 kHz stereo AAC track
(the original footage has no audio). No new media licence is asserted; the file is kept
here only because the masks are meaningless without the exact bytes they were computed on.

To regenerate it without their script (after downloading and digest-checking `vtest.avi`):

```bash
ffmpeg -v error -y -i vtest.avi -f lavfi -i anullsrc=r=48000:cl=stereo -t 12 \
  -map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 10 -c:a aac \
  -movflags +faststart pedestrians.mp4
```

### Commands that produced the JSON

From `vision/` on `origin/vision/validated-pipeline` (a read-only `git worktree`), with
`.venv` created per their README:

```bash
python examples/download_demo.py          # -> examples/media/pedestrians.mp4 (+ source.json)

# detections.json: their documented mask command, with a shorter analysis window
python -m soundcanvas_cv --input examples/media/pedestrians.mp4 --model yolo26n-seg.pt \
  --output-dir outputs/fixture_seg --sample-fps 10 --max-seconds 6 --device cpu
```

The only deviation from `vision/README.md` is `--max-seconds 6` instead of `12`:
per-frame native-resolution RLE for every detected object is dense, and the full 12 s run
yields a 2.97 MB `detections.json` (3.8 MB with the clip), over this repo's fixture
budget. That 12 s run was executed too and behaves identically — 120 samples / 120
frames / 120 inference calls, `{person: 585, car: 120, truck: 120}` observations. The
**video is still the full 12 s clip**, so its SHA256 is theirs unchanged; only the
analysed window is shorter (`processed_duration_seconds = 6.0`).

`selection.json` was created with their selection code, reproducing
`vision/examples/validate_real.py::prepare_demo` exactly (same track choice, same click
point, same explicit interval extension) against the detections above:

```python
from collections import Counter
import json
from soundcanvas_cv.selection import select_at_point
from soundcanvas_cv.audio_adapter import prepare_visual_inputs

data = json.load(open("outputs/fixture_seg/detections.json"))
counts = Counter(o["track_id"] for f in data["frames"] for o in f["objects"] if o["label"] == "person")
tid = counts.most_common(1)[0][0]                      # -> 7
frame = next(f for f in data["frames"] if any(o["track_id"] == tid for o in f["objects"]))
obj = next(o for o in frame["objects"] if o["track_id"] == tid)
x1, y1, x2, y2 = obj["bbox_normalized"]
selection = select_at_point(data, frame["timestamp_seconds"], (x1 + x2) / 2, (y1 + y2) / 2)
# vision/README.md: "an integrating editor may deliberately extend `interval` to native
# frame boundaries" - prepare_demo() does the same (0..12 for the 12 s clip).
selection["interval"] = {"start_seconds": 0.0, "end_seconds": data["processed_duration_seconds"]}
json.dump(selection, open("outputs/fixture_seg/selection.json", "w"), indent=2)
prepare_visual_inputs("examples/media/pedestrians.mp4", data, selection)  # their validator
```

A raw viewer click would hold only one sample interval (0.1 s here, since sampling runs at
native FPS). The extension above is Person 2's documented recipe for preparing longer
clips, and it is what makes the fixture a meaningful 60-frame test.

Person 2's own `prepare_visual_inputs` accepts these two documents and returns
`frames_rgb (60, 3, 576, 768)`, `foreground_masks (60, 1, 576, 768)`,
`mask_kind=instance_segmentation`, `fps=10.0`, `0.0 -> 6.0 s`, 49 non-empty mask frames —
so the artefacts are self-consistent on their side of the boundary before this service
sees them.

## What the documents contain

* 60 exported samples / 60 processed frames / 60 real detector calls, `sampling_fps`
  10.0 == native `fps` 10.0, so there is a mask at **every** native frame, which is what
  `prepare_masked_video` requires.
* 362 observations carry an instance mask; `{person: 242, car: 60, truck: 60}`
  observations overall; 13 distinct track ids, 11 of them `person`.
* Selected object: `track_id` 7, `object_id` `track:7`, label `person`, clicked at
  `frame_index` 11 / `timestamp_seconds` 1.1, `detection_id` `frame:11:det:5`,
  `interval` 0.0 -> 6.0 s, `audio_association: unverified`.
* Track 7 has masked observations in 49 of the interval's 60 native frames; it is simply
  not yet detected in frames 0-10. Per the contract those frames become **empty
  foreground** — no mask is reused and no continuous identity is claimed.

## Regenerating

```bash
git -C <repo> worktree add /tmp/vision-wt origin/vision/validated-pipeline
cd /tmp/vision-wt/vision
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
.venv/bin/python examples/download_demo.py                 # needs `requests` and ffmpeg
.venv/bin/python -m soundcanvas_cv --input examples/media/pedestrians.mp4 \
  --model yolo26n-seg.pt --output-dir outputs/fixture_seg \
  --sample-fps 10 --max-seconds 6 --device cpu
# then the selection snippet above, and copy
#   detections.json, selection.json, examples/media/pedestrians.mp4
# into audio-service/fixtures/vision/
```

`analysis_id` is a fresh UUID per run and track ids depend on tracker state, so a
regenerated `detections.json` will not be byte-identical; `source_sha256` and the
geometry will be.
