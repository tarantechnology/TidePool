# Runpod vision and object-to-sound experiments

This branch extends the existing detector, selection, audio service and export
contracts. Detection/selection schema 1.0 is unchanged. The optional evidence
sidecar is an offline research artifact, not a new accepted HTTP contract.
Related to Issue #4; real model execution alone does not close its quality gate.

## Environment and storage

Local source of truth, verified with `pwd`:
`/Users/shashwater/orca/SoundCanvas-AkhilNam`, branch
`vision/runpod-object-sound`, based on main `6e1f421`.
Remote source snapshot: `/workspace/SoundCanvas`. The initial private repository
clone was unavailable without credentials; a checksum-verified `git archive HEAD`
snapshot was transferred over the authorized SSH shell. No Mac credential or
untracked personal file was transferred. The remote Git snapshot has its own base
commit; local history remains authoritative.

The pod has an idle NVIDIA L4, 23,034 MiB reported VRAM (22,565 MiB CUDA-usable),
driver 595.91.07, Python 3.12.3, PyTorch 2.8.0+cu129, torchvision 0.23.0+cu129.
The driver advertises CUDA 13.2; the installed PyTorch CUDA runtime is 12.9.
The cgroup RAM limit is 70,999,998,464 bytes (66.1 GiB). Host-wide `free` reports
503 GiB and must not be mistaken for the pod allocation.

`findmnt -T /workspace` identifies a separate `fuse` mount backed by
`mfs#euro-2.runpod.net:9421[/podvolumes/.../ipo66fqm457b9b]`, not the container's
50-GB overlay. This establishes a Runpod pod volume: persistent across pod stops,
with survival after deletion **not established**. Runpod distinguishes
[pod disks](https://docs.runpod.io/pods/storage/types) and
[network volumes](https://docs.runpod.io/storage/network-volumes).
`df` reports 459 TiB free on the shared filesystem; account quota is unknown.
No pod stop/delete or volume migration was performed to test persistence.

Environments are `/workspace/venvs/soundcanvas-vision` and
`/workspace/venvs/soundcanvas-audio`, isolated with `--system-site-packages` to
reuse template Torch without modifying it. Model cache and user-provisioned
authentication remain at `/workspace/.cache/huggingface`; pip/Torch caches use
`/workspace/cache`. No authentication files are copied or changed.

## Model choices and compatibility

Official SAM 3.1 Object Multiplex supports text discovery and prompted multi-object
tracking. It supplies neither action recognition nor acoustic event detection.
See the [official release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
Use one concept per analysis, such as `knife`, `person`, `dog`, or `car`; each matching
instance receives its own upstream ID. `class_id=0` means the prompted concept,
not COCO person. IDs are model hypotheses, not verified physical identities.

- Official SAM 3 source: `eef6839bcfc3458dd52ea80535dc6bd62aea86b1`.
- SAM 3.1 checkpoint revision: `daa63191845a41281374e725f4c9e51c7a824460`.
- Official SAM Audio Small TV revision: `afc58d67609fee0086b9c032d91b7db8ebec1aa7`.
- SAM 2 source (explicit public alternative): `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- SAM 2.1 Large SHA256: `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
- YOLO26n-seg SHA256: `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`.
- YOLO26x-seg SHA256: `92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3`.

SAM 3.1's shared upstream session dispatcher passes unsupported
`offload_state_to_cpu` into multiplex `init_state`, even when false. Apply
`vision/patches/sam3-session-init.patch` to the pinned external source. It removes
the unsupported false argument and rejects true, preserving the public session API.
The adapter requests CPU **video** offload for 3.1. Default grounding batch 16
exceeded L4 VRAM (20.20 GiB allocated, 21.46 GiB total process memory, next 1.27-GiB
allocation failed). The adapter uses grounding batch 1 and postprocess batch 1,
with the same 1008-pixel model and checkpoint. FA3 and compilation are disabled.

SAM 2.1 Large is a separately reported alternative, never a silent SAM 3 substitute.
It requires user seed points and labels, and cannot discover knives from text.
Its optional CUDA hole-filling extension was disabled; upstream warns that this
postprocessing step is skipped. YOLO remains the default and needs no gated weights.

SAM Audio's official `-tv` checkpoints are designed for visual prompting and target
correctness ([official usage](https://github.com/facebookresearch/sam-audio)).
Small TV is the explicitly approved configuration for this L4 experiment; larger
TV variants have not been measured or selected. Rankers/span predictor are disabled,
one candidate is generated, seed 42, float32. The full visual encoder is retained.
`AUDIO_VISUAL_BATCH_SIZE=8` bounds its microbatch while retaining 336-pixel inputs.
This changes compute batching, not checkpoint size or mask resolution.
`bfloat16` is an explicit experimental option; fp16 remains rejected.

## Reproduction

Connect using:

```bash
ssh ipo66fqm457b9b-64410f3e@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/SoundCanvas
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/cache/torch PIP_CACHE_DIR=/workspace/cache/pip
export YOLO_CONFIG_DIR=/workspace/cache/ultralytics
V=/workspace/venvs/soundcanvas-vision/bin/python
A=/workspace/venvs/soundcanvas-audio/bin/python
```

The existing HF CLI at `/workspace/hf-cli-venv/bin/hf` uses the authenticated pod
account. Check access before download; never pass a credential on a command line:

```bash
/workspace/hf-cli-venv/bin/hf auth whoami
/workspace/hf-cli-venv/bin/hf download facebook/sam3.1 sam3.1_multiplex.pt
/workspace/hf-cli-venv/bin/hf download facebook/sam-audio-small-tv
$V scripts/runpod_preflight.py --output /workspace/experiments/preflight-new.json
```

Runtime installation uses existing template Torch with an explicit constraint file:

```bash
python -m venv --system-site-packages /workspace/venvs/soundcanvas-audio
/workspace/venvs/soundcanvas-audio/bin/python -c \
 'import importlib.metadata as m; print("\n".join(n+"=="+m.version(n) for n in ["torch","torchvision","torchaudio"]))' \
 > /workspace/audio-torch-constraints.txt
/workspace/venvs/soundcanvas-audio/bin/pip install --no-compile \
 -c /workspace/audio-torch-constraints.txt -e './audio-service[sam]' torchcodec==0.7.0
```

Vision uses numpy 1.26.4, opencv-python 4.11.0.86, contourpy 1.3.3, ultralytics
8.4.212, lap 0.5.14, timm 1.0.31, hydra-core 1.3.8 and decord 0.6.0. Install official
SAM source at the pinned revisions, editable into the vision environment, then:

```bash
git -C /workspace/vendor/sam3 apply --check "$PWD/vision/patches/sam3-session-init.patch"
git -C /workspace/vendor/sam3 apply "$PWD/vision/patches/sam3-session-init.patch"
/workspace/venvs/soundcanvas-vision/bin/pip install --no-deps -e ./vision
```

Use a fresh output directory for each run. GPU experiment scripts acquire an
exclusive file lease, refuse existing compute processes and never terminate them.
The lease coordinates these scripts; it cannot control unrelated applications.
The app services must also be operated sequentially until combined load is measured.

```bash
P=/workspace/experiments/cooking-new
$V vision/examples/download_cooking_demo.py --output "$P"
S=/workspace/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt
$V vision/scripts/gpu_vision.py --input "$P/cooking.mp4" --output "$P/sam3p1" \
 --adapter sam3.1 --model "$S" --concept knife
# Independent public comparisons, each after the previous process exits:
$V vision/scripts/gpu_vision.py --input "$P/cooking.mp4" --output "$P/yolo-n" \
 --model /workspace/checkpoints/yolo26n-seg.pt
$V vision/scripts/gpu_vision.py --input "$P/cooking.mp4" --output "$P/yolo-x" \
 --model /workspace/checkpoints/yolo26x-seg.pt --image-size 1280
$V vision/scripts/render_masks.py --video "$P/cooking.mp4" \
 --detections "$P/sam3p1/detections.json" --output "$P/sam3-review"
$V vision/scripts/analyze_events.py --video "$P/cooking.mp4" \
 --detections "$P/sam3p1/detections.json" --output "$P/sam3p1/evidence.json" \
 --hypotheses vision/config/knife-hypotheses.example.json
```

The sample is Obihoja's 2025 *Cutting Vegitable In A Market*, CC BY-SA 4.0, verified
against original SHA1 `fff39ddc436b0517615634087c0a2f13a8f9df52`. The downloader
retains attribution and records the 12-second/30-FPS/720p normalization. Its original
audio is retained; there are no reference stems and no guaranteed music layer.
It is a cooking-related stress test, not the requested chef/dialogue/music acceptance
fixture. Re-encoders can produce different bytes; always analyze and upload the same
normalized file. Generated media/weights are ignored and must not enter Git.

The existing picker remains `vision/viewer.html`. Load the same original and native
`detections.json`. Click and export a selection, with a short frame-aligned interval.
Then run the real audio comparison with that selection:

```bash
PYTHONPATH=vision:audio-service $A audio-service/scripts/compare_visual_modes.py \
 --video "$P/cooking.mp4" --detections "$P/sam3p1/detections.json" \
 --selection "$P/selection-4s.json" --prompt 'knife chopping' \
 --model-id facebook/sam-audio-small-tv --mode text --output "$P/audio-text"
# Use --mode visual or --mode text_visual for single comparison arms.
# --mode all additionally needs --anchors path/to/anchors.json.
```

The four arms are text, genuinely visual-only (blank description and no fallback
label), text+visual, and text+visual+temporal. Identical interval, seed, checkpoint
and numerical settings are used. Anchors have `{token,start,end}` source-timeline
coordinates, with provenance explicitly `unverified_candidate` unless reviewed.
They guide model inference; they do not assert a sound occurred or become mute ranges.
Object masks use the existing `keep_object` polarity. `--mask-mode notebook` is an
explicit A/B against upstream's unresolved opposite-polarity example.

Each successful arm writes full-timeline float32 target/residual WAVs, a residual
MP4, runtime/VRAM, sample counts, reconstruction error, near-silence flag and export
probes. Mixture residuals preserve original stereo. This standalone experiment
subtracts the target for comparison; the application should use its existing
candidate preview/apply/edit/export flow after review. It does not replace that API.

## Evidence and application integration

`events.py` exports model-observed IDs/native-mask references/timestamps, bbox
acceleration features, spectral-flux transients, optional independently annotated
audio events, supplied semantic hypotheses and temporal associations. Every
association carries its source IDs, timing delta, heuristic confidence and
`causality_verified=false`. Simultaneous candidate objects cause abstention.
Silence creates no audio association; missing observations break motion evidence.
Scores are uncalibrated and capped below .5. Camera motion, segmentation jitter,
stationary sources, sustained scraping and offscreen sounds remain limitations.
This is **not** a semantic action or audio event recognizer. Hypotheses such as
knife chopping/scraping/tapping are not observations. Only explicitly reviewed,
nonambiguous events can be promoted via `reviewed_anchors` for application use.

`VISION_ADAPTER=sam3.1`, `VISION_MODEL=<checkpoint>`, `VISION_CONCEPT=knife` select
the adapter through the existing backend detector subprocess. YOLO is still the
default. `VISION_IMAGE_SIZE=1280` applies only to YOLO. SAM 2 requires explicit
`VISION_SEED_OBJECTS`; the adapter never invents object seeds. No service is started
or reconfigured automatically. See `vision/config/runpod.env.example` and existing
`examples/visual_audio_flow.py` for trusted upload/selection/stem/edit/export.

## Recorded measurements

The first public-model runs used the same real 12-second/360-frame clip:

| Model | End-to-end seconds including load | Peak allocated / reserved GiB | Knife observations | Track hypotheses |
|---|---:|---:|---:|---|
| YOLO26n-seg, 640 | 23.77 | 0.107 / 0.236 | 0/360 | none |
| YOLO26x-seg, 1280 | 46.03 | 0.734 / 0.859 | 203/360 | 5 IDs; longest 172 observations |
| SAM 2.1 Large, manual points | 55.63 | 1.679 / 2.002 | 360/360 | 1 ID |
| SAM 3.1, text knife, batch 1 | 177.65 | 9.129 / 9.598 | 357 mask observations on 348/360 frames | dominant ID: 348 observations; 4 brief additional IDs |

Coverage and track lengths are **not accuracy or ID-switch metrics**. Six sampled
SAM 2 overlays visually follow the knife during chopping/lifting; no pixel ground
truth or annotated physical identities are available. Its mean adjacent unwarped
mask IoU is .453, which is affected by real motion and cannot establish accuracy.
The conservative sidecar found 90 visual-motion features, 30 audio transients and
57 candidate timing matches, none causally verified. These are not 57 knife sounds.

SAM 3.1 completed all 360 native frames with real CUDA inference. Six-frame visual review and pixel-ground-truth accuracy are separate from coverage. Its adjacent unwarped IoU is .441, also affected by real motion. The SAM 3.1 evidence sidecar has 88 motion features, 30 transients and 57 unverified associations. Real audio measurements are recorded after inference.

## Regression checks

```bash
python -m pytest -q
python -m pytest -q -c audio-service/pyproject.toml audio-service/tests
PYTHONPATH=vision python -m pytest -q -c vision/pyproject.toml vision/tests
```

Adapter tests inject predictors to validate identity, schema, absence and frame
ordering; they do not prove model inference. Real experiments are opt-in scripts.
Issue #4 stays open until representative visual-guided isolation, cross-team review
and listening/reference evaluation demonstrate the intended sound relationship.

## Real SAM Audio Small TV comparison

Official model code `bb4c6999d2677c7402360e426afc01ddfad6dce0`, Perception code
`e72b6810b1133e1c879f2cc965d276eb73803f1f`, DACVAE code
`414c20785fc3a28373073ea8ef7a1316eeeaca6e`. Small TV checkpoint size:
5,100,547,943 bytes; official Hub ETag/SHA256:
`1a9693b235efc3176986664dac349eb64022c7468cf8d4195954490a2457e6a9`.
Runtime import and `pip check` pass on the unchanged Torch 2.8.0+cu129 template,
with torchcodec 0.7.0, xformers 0.0.32.post2, transformers 4.57.6 and Hub 0.36.2.
Persistent-filesystem installation is slow; `--no-compile` avoids optional
installation-time bytecode writes. These do not disable the model or visual encoder.

A real run loaded 1,384,664,326 parameters, disabled only span predictor/rankers,
retained the 336-pixel visual encoder and used its official batch-size control (8).
Load including first imports: 92.67 s; model loader itself: 28.92 s. Vision had
already exited. Same native SAM 3.1 knife masks, interval `[0,4)`, float32, seed 42,
text `knife chopping`, one candidate, midpoint solver with step size .0625:

| Prompt | Inference seconds (CUDA synchronized) | Peak allocated / reserved GiB | Target RMS within processed interval |
|---|---:|---:|---:|
| Text | 1.504 | 6.564 / 7.238 | .015802 |
| Visual only, blank text | 7.398 | 6.595 / 7.242 | .015952 |
| Text + visual | 7.483 | 6.596 / 7.242 | .010551 |
| Text + visual + candidate temporal anchors | 7.438 | 6.598 / 7.242 | .004099 |

All four arms produced actual non-silent target/residual WAVs and edited MP4s.
Full-timeline stems have 576,512 samples at 48 kHz, offset zero; target is zero
outside `[0,4)`, and mixture-derived residual preserves the rest of the original.
Maximum float reconstruction error is `2.98e-8`. On the pod, the source has 360 video frames
and 12.000 s video; AAC decode includes a 512-sample padding tail (12.010667 s).
Export probes are saved for every arm. This is execution/alignment evidence,
not target-specific attenuation or speech/music preservation.

Three temporal anchors were selected from unverified audio/motion coincidences.
The combined arm's lower target energy could reflect missed target sound or reduced
leakage; neither interpretation is established. No isolated reference stems,
ground-truth masks or annotated physical identities exist for this sample. Listening
was not performed; **visual benefit, knife-specific attenuation, speech preservation
and music preservation remain unverified**. The clip is not an acceptance fixture
for chef narration plus background music. Keep Issue #4 open.

For the full existing backend workflow, the isolated harness starts two localhost
services, loads the real model, submits the existing trusted upload/selection
example, mutes the generated target, exports MP4, and always stops its own services:

```bash
PYTHONPATH=vision:audio-service $A scripts/run_gpu_e2e.py \
 --video "$P/cooking.mp4" --detections "$P/sam3p1/detections.json" \
 --selection "$P/sam3p1/selection.json" --prompt 'knife chopping and scraping' \
 --output "$P/backend-e2e"
```

The harness's `storage/` contains ordinary project credentials in its database;
never copy it into reports or artifact bundles. Only `result/` stems, edited media,
model metadata and export probes are shareable experiment outputs.

The real 12-second backend test completed with `model.is_mock=false`, CUDA, one
pass, visual guidance and text `knife chopping and scraping`. Service separation
took **22.403 s**, peak allocated VRAM **8.724 GiB**. It returned full stereo
stems with **576,512 samples at 48 kHz**, target muted, then exported H.264/AAC:
**360 video frames, 12.000 s video/audio, both stream origins 0.000 s**. The
service explicitly disabled chunking to keep the visual interval aligned. Both
isolated test services were stopped afterward and GPU memory returned to zero.
This proves the real-model upload → stored masks → separation → mute → MP4 path;
correct knife removal and preservation still require listening/reference evidence.

Local verification against the captured PCM16 `before.wav` (standard stereo upmix)
finds full-pair reconstruction error `1.08e-5`, within that reference's quantization.
AAC export versus the residual has RMSE `.0012175` over the meaningful 12 seconds.
The Mac FFmpeg decoder returns 575,552 samples for the original AAC MP4, while the
pod returned 576,512; cross-decoder sample parity is not established. Use the captured
source PCM for comparisons and retain the pod's FFmpeg version for reproduction.
Exported stream timing is verified; sample-level source/export identity across
different decoders is not claimed. Future reference fixtures should use lossless
source audio to remove this codec ambiguity.
