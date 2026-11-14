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

The app makes a silent 15 FPS analysis proxy, preserving the extracted original
audio, project source and existing layer files. Masks use proxy-frame timestamps
and normalized coordinates for the displayed video. Nonaligned audio/video starts
are rejected explicitly. The original video stream is retained for export.

Selections carry analysis/frame/detection/track IDs. The backend validates them
against stored observations and original media identity, retains context in
candidate provenance, and strips it before audio inference. Matching existing
target layers can be muted/lowered through a revision-checked preview without
running separation again; apply/undo preserves their audio files and other edits.
Explicit processing ranges, anchors, quality retries and extra passes use the
ordinary separation path instead of this reuse optimization.

## Verified movie demo on the L4

The user's 20.6-second movie clip produced one SAM dog track, observed in 262 of
308 analysis frames. Cold analysis took 150.48 seconds; peak SAM allocated memory
was 9,461,733,888 bytes. These are execution/coverage measurements, not mask
accuracy against labeled ground truth. Object analysis preserved the original
extracted WAV bytes exactly.

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
