# TidePool defaults to AudioSep

The normal upload → sound prompt → preview → apply/undo → edit → export flow uses
AudioSep base 4M. The UI displays the separator reported by the running service.
SAM remains an explicit alternative, and mock/fake operation must be explicitly
selected for development.

## Behavior

The default request is text-only, one candidate/pass, float32, 10-second windows,
2-second overlaps and batch one. Conservative suppression runs after the complete
target is reconstructed from all windows, preserving the original stereo phase
and limiting edits when target activity is weak. At strength .9 the residual
retains some target sound to reduce unintended changes to other sounds. This is
not a guarantee of unchanged voices where speech and the selected sound overlap.

`options.suppression` accepts `conservative` or `off`, and
`options.suppression_strength` accepts 0–1. Defaults are configured through
`AUDIO_SUPPRESSION` and `AUDIO_SUPPRESSION_STRENGTH`. The effective settings are
reported in `model.suppression`. Legacy `refine=mask` is not applied a second time
when conservative suppression is active. Returned target + residual reconstruct
the source, and the existing editor mutes/lowers that processed target. WAVs are
float32; server MP4 exports use copied video and AAC 192 kb/s.

AudioSep uses no SAM visual masks, time anchors, independent candidates or ranked
retries. Those controls follow the live capabilities rather than pretending they
work with AudioSep. Capability-driven requests use one candidate and batch one;
unsupported quality retries are rejected before starting a job.

## RunPod service

Use the authorized SSH connection and a source checkout on the persistent volume.
Existing dirty source checkouts and previous experiment outputs must be preserved.
The verified installation uses `/workspace/vendor/AudioSep`, revision
`944583f18b84589dc965de3ad77525c945334252` plus the existing compatibility patch.
The official public weights are pinned to Space revision
`0807a713283035d8b0d0628f1b8e3d6580459f7b`, reused from the checkout/cache when
present. Explicit missing checkpoint paths fail; no other model is substituted.

```sh
# On the Pod, from the source checkout:
bash scripts/runpod_audiosep.sh setup   # only if the runtime needs preparing
bash scripts/runpod_audiosep.sh start
bash scripts/runpod_audiosep.sh status
```

The script reuses the verified existing isolated environment when present,
otherwise prepares `/workspace/venvs/soundcanvas-audiosep`. It preserves template
Torch by constraining the installed Torch/torchaudio/torchvision versions. Model
access is serialized, and the service listens only at `127.0.0.1:8011` to avoid the
RunPod template's nginx listener on port 8001. The stop command signals only a
service with this checkout's matching PID and working directory:

```sh
bash scripts/runpod_audiosep.sh stop
```

`scripts/runpod_setup.sh` delegates to this AudioSep launcher by default. Set
`AUDIO_BACKEND=sam` explicitly to use the legacy SAM setup instead.

## Mac app

Forward local port 8001 to remote loopback port 8011 using the Pod's direct TCP SSH
host/port and your authorized identity. The gateway SSH address is useful for
interactive administration; the direct TCP endpoint supplies forwarding.

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8001:127.0.0.1:8011 -p SSH_PORT -i ~/.ssh/id_ed25519 root@SSH_HOST
```

In the source checkout, install the API/frontend dependencies and start:

```sh
scripts/run_local_real.sh remote  # also the default mode with no arguments
```

Open `http://localhost:5173/#/app`. The launcher checks that the expected separator
is ready before starting the backend. The backend defaults to HTTP inference at
`http://127.0.0.1:8001/separate`; an unavailable separator fails instead of silently
copying the recording. Project storage stays local to the checkout. No Mac
credentials or project database need to be transferred to the Pod for text
separation; the service receives uploaded audio bytes.

Keep the Pod and SSH tunnel running while uploading/separating clips. The browser
can play already downloaded/exported files after the Pod is stopped.

## Containers

`compose.cloud.yaml` and the GPU overlay select the AudioSep image target by
default. The standalone `audio-service/Dockerfile` also defaults to the AudioSep
stage. The image installs the pinned `audiosep` dependency extra and official
source patch, caching model files under `/models`. HF_TOKEN is optional for these
public weights. `sam` and `mock` image targets remain explicit choices.

For UI-only development, set `INFERENCE_ADAPTER=fake` or explicitly run
`AUDIO_TARGET=mock AUDIO_BACKEND=mock` with the audio-service overlay. Those modes
are labeled and are not real separation.
