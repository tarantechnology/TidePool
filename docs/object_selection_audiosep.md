# Click the dog, mute its barking

Upload a video, open **Pick a sound**, and press **Find dog in video**. The
object description defaults to `dog` and can be changed. SAM 3.1 analyzes the
first 30 seconds on the GPU. Click its dog outline or the dog in the object list;
the sound field starts as `dog barking`. Choose **Mute**, compare the preview,
then **Apply to timeline** and export. The close button hides the object actions.

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
- `AUDIO_OBJECT_SOURCE_ROOT`: the SoundCanvas source checkout
- `AUDIO_OBJECT_CACHE`: `/workspace/cache/soundcanvas-objects` by default

`AUDIO_OBJECT_ENABLE=0` disables launcher setup. Missing paths report unavailable;
no checkpoint is automatically downloaded or substituted. The remote local-app
launcher configures `VISION_SERVICE_URL` for the default AudioSep setup. Other
deployment modes may configure it explicitly. Keep application endpoints private.

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
