# Find people and objects, then choose their sound

Upload a video and automatic discovery starts without an object prompt. Open
**Pick a sound** to see people and common objects found by the cached YOLO26n
segmentation model. **Find people and objects** reruns the scan. Click a dog
outline or list row; the sound field starts as `dog barking`. Choose **Mute**,
compare the preview, then **Apply to timeline** and export.

For finer masks, press **Refine person**, **Refine dog**, or another discovered
category. This uses the existing SAM 3.1 path without typing the category. It
replaces the current observations with SAM's tracks; select an outline again.
**Find a specific object with SAM** retains the optional custom-description path.
These models run sequentially behind the shared GPU inference lock.

Person selections leave **Sound to edit** empty: describe the audible action,
such as `a man imitating a dog barking`. Clothing and person masks do not identify
a voice. Mute stays disabled until a sound description is supplied. The close
button hides the object actions. Automatic labels and track IDs can be wrong or
fragment across cuts; track counts are not verified counts of distinct people.
The automatic picker hides tracks seen fewer than three times to reduce brief
class flicker; this can miss objects visible only very briefly. SAM search keeps
its original observations.

SAM supplies tracked masks for selection. AudioSep receives the ordinary text
prompt and original audio, using the same suppression and inference profile as
the typed command. Masks and object visibility do not gate or condition AudioSep.
The sound is edited throughout the clip unless a separate range edit is requested.
Multiple dogs or similar sounds cannot be distinguished acoustically by a mask.
This does not fix the existing overlapping-speech separation artifacts.

## Runtime and storage

The existing private audio service exposes optional `/objects/capabilities` and
`/objects/analyze` endpoints. `VISION_SERVICE_URL` on the app backend points to
that service, e.g. `http://127.0.0.1:8001` through the existing SSH tunnel. The
remote launcher enables the prepared SAM runtime when these paths exist:

- `AUDIO_OBJECT_PYTHON`: `/workspace/venvs/soundcanvas-vision/bin/python`
- `AUDIO_OBJECT_CHECKPOINT`: the cached official SAM 3.1 multiplex checkpoint
  at revision `daa63191845a41281374e725f4c9e51c7a824460`
- `AUDIO_OBJECT_SOURCE_ROOT`: the TidePool source checkout
- `AUDIO_OBJECT_CACHE`: `/workspace/cache/soundcanvas-objects` by default
- `AUDIO_OBJECT_DISCOVERY_MODEL`: `/workspace/checkpoints/yolo26n-seg.pt`
  (explicit cached segmentation checkpoint for automatic discovery)

`AUDIO_OBJECT_ENABLE=0` disables launcher setup. No checkpoint is automatically
downloaded or substituted. Automatic discovery works with its YOLO weights and
runtime even when the gated SAM checkpoint is missing; in that configuration the
capabilities report `automatic: true` and `prompted: false`, and SAM refinement
is hidden. SAM refinement requires its own checkpoint and runtime. The remote local-app
launcher configures `VISION_SERVICE_URL` for the default AudioSep setup. Other
deployment modes may configure it explicitly. Keep application endpoints private.
The `automatic` capability is advertised only when discovery weights are present.
The app launches discovery once per new video upload; reopening an existing
project loads its stored observations. An unavailable automatic runtime leaves
the explicit SAM search available. Cache results are separated by source, model,
implementation and search concept. Automatic class changes on one native tracker
ID receive distinct selection IDs, so a dog row cannot turn into a person row.

SAM runs in its prepared interpreter in a subprocess, with grounding batch one,
compilation/FA3 disabled, and CPU video offload. Its subprocess holds the same
inference lock as AudioSep and exits before another inference can start. AudioSep
can remain resident; only one GPU-service instance should own these requests.
Cache keys include analysis-video bytes, concept, checkpoint identity, and adapter
implementation. A repeated analysis reads cached masks without loading SAM again.

The app makes a silent 10 FPS analysis proxy, preserving the extracted original
audio, project source and existing layer files. Masks use proxy-frame timestamps
and normalized coordinates for the displayed video. Nonaligned audio/video starts
are rejected explicitly. The original video stream is retained for export.
`OBJECT_ANALYSIS_FPS=15` on the app backend restores the earlier, finer tracking
cadence; values from 1 to 30 are accepted. The default updates outlines every
100 ms, so fast motion may be less precise between observations. Resolution,
checkpoint, confidence threshold, and the 30-second coverage limit are unchanged.
Audio edits still use the original audio across the requested scope.

Selections carry analysis/frame/detection/track IDs. The backend validates them
against stored observations and original media identity, retains context in
candidate provenance, and strips it before audio inference. Matching existing
target layers can be muted/lowered through a revision-checked preview without
running separation again; apply/undo preserves their audio files and other edits.
Explicit processing ranges, anchors, quality retries and extra passes use the
ordinary separation path instead of this reuse optimization.

## Verified movie demo on the L4

Automatic discovery on the user's `IMG_1758.MOV` found tracked people, including
the opening frame, in 29.95 seconds end to end (9.02 seconds in the detector
pipeline). The movie found both people and a dog in the opening frame, in 19.54
seconds end to end (7.79 seconds in the pipeline). Both scans preserved the
extracted original audio bytes. These are individual execution measurements,
not recognition-accuracy scores. Transient false labels and fragmented tracks
were observed. The full 80-class scan is a quick selection aid; SAM refinement
remains available for more precise masks and custom concepts.

Browser upload automatically discovered people without an object prompt. Person
selection required an audible description and successfully generated a real
AudioSep preview. No listening or speaker-isolation accuracy is claimed for it.
Artifacts: `~/Downloads/TidePool-Automatic-Objects/`.

At the earlier 15 FPS cadence, the user's 20.6-second movie clip produced one SAM
dog track, observed in 262 of 308 analysis frames. Cold analysis took 150.48
seconds; peak SAM allocated memory was 9,461,733,888 bytes. These are execution/coverage measurements, not mask
accuracy against labeled ground truth. Object analysis preserved the original
extracted WAV bytes exactly.

The 10 FPS speed trial processed 206 frames in 99.15 seconds inside the SAM
pipeline (105.37 seconds through the proxy/HTTP path), versus 140.80 seconds in
the earlier pipeline run: about 30% less model-pipeline time. Peak SAM allocated
memory fell to 8,607,673,344 bytes. One dog track was observed in 176 frames,
including the opening frame. At nearest timestamps (within 33.4 ms), there was
one presence disagreement out of 206 samples: an extra mask at the 5.6-second
scene cut. Mean mask IoU when both runs had a mask was 0.926; this measures output
agreement, not ground-truth accuracy. Outlines at fast motion/cuts can differ.

An 8 FPS trial was faster (79.37 seconds), but missed the opening half-second and
was not chosen. An additional outer bfloat16 context gave no meaningful gain;
the official SAM 3.1 predictor already enables it internally. That redundant
setting was removed. Trial reports and inspected mask comparisons are on the Mac
under `~/Downloads/TidePool-SAM-Speed/`. Timings are single runs on the L4 and vary
with startup, GPU load, transfers and queuing.

Real AudioSep previews from a typed `dog barking` mute and from the selected dog
produced sample-identical 48 kHz stereo edited WAVs (maximum difference 0.0).
API preview/apply/undo, existing-layer reuse without inference, and MP4 export
passed. Browser upload, cached detection, dog outline selection, prefilled sound,
Mute, preview, apply, and export also passed. Exports retain all 609 source video
frames. The existing short speech artifact remains possible because the audio
separator is unchanged.

Demo files, metadata and screenshots are on the Mac under
`~/Downloads/TidePool-SAM-Dog-Selection/`. Credentials and model weights are not
included. Interactive SAM correction clicks are available in the research adapter
but are not exposed by this initial editor workflow; reanalyze with a different
object description or use a text prompt if selection is unsuitable.
