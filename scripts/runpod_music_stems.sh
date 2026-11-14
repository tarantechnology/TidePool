#!/bin/bash
# Audio service with music-stem routing on a RunPod GPU pod (branch audio/music-stems).
#
#   bash scripts/runpod_music_stems.sh setup           build /workspace/venv-svc (torch 2.8 cu128 + splitters + AudioSep deps)
#   bash scripts/runpod_music_stems.sh start [quality] start on 127.0.0.1:8021 (default profile: fast)
#   bash scripts/runpod_music_stems.sh stop
#
# Profiles (A40, 4-min song, see audio-service/benchmarks/results/2026-11-14-music-stems/REPORT.md):
#   fast     htdemucs, shifts 0                                   ~20 s, vocals 8.5 dB uSDR
#   quality  BS-RoFormer ep_317 vocals (overlap 4) -> htdemucs    ~200-300 s today, vocals 11.7 dB uSDR
# From the Mac: ssh -fN -L 8001:127.0.0.1:8021 -p PORT -i ~/.ssh/id_ed25519 root@HOST
#               then scripts/run_local_real.sh remote
set -euo pipefail
ROOT=${MUSIC_STEMS_ROOT:-/workspace/music-stems}   # holds audio-service/ (rsync --no-owner --no-group from the repo)
V=${MUSIC_STEMS_VENV:-/workspace/venv-svc}
export HF_HOME=${HF_HOME:-/workspace/hf} TORCH_HOME=${TORCH_HOME:-/workspace/torch-cache}

case "${1:-}" in
setup)
  export UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
  [ -x "$V/bin/python" ] || uv venv -q --python 3.12 "$V"
  uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
  uv pip install --python "$V/bin/python" torch==2.8.0 torchaudio==2.8.0 demucs==4.0.1 julius==0.2.8 "audio-separator[gpu]==0.48.0" \
    audioread==3.1.0 bs-roformer==0.3.10 beartype==0.18.5 panns-inference==0.1.1 transformers pyyaml soundfile scipy librosa \
    lightning torchlibrosa h5py ftfy braceexpand webdataset wget==3.2 huggingface_hub pandas tqdm scikit-learn regex \
    "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "python-multipart>=0.0.9" "pydantic>=2.7,<3" "httpx>=0.27,<1"
  uv pip install --python "$V/bin/python" --no-deps -e "$ROOT/audio-service"
  # AudioSep (prompted model) checkout + checkpoints: audio-service/scripts/setup_music_stems_venv.sh prepares /workspace/AudioSep.
  "$V/bin/python" -c "import torch, demucs, audio_separator, sam_audio_service; print('ok', torch.__version__, torch.cuda.is_available())" ;;
start)
  pkill -f "$V/bin/python -m sam_audio_service" 2>/dev/null || true; sleep 1
  common=(AUDIO_BACKEND=audiosep AUDIO_AUDIOSEP_ROOT=/workspace/AudioSep AUDIO_DEVICE=cuda AUDIO_HOST=127.0.0.1 AUDIO_PORT=8021
          AUDIO_MEDIA_ROOT="$ROOT/media" AUDIO_ROUTING_DEFAULT=auto AUDIO_ROUTER_CLASSIFIER=ast AUDIO_STEM_SHIFTS=0
          AUDIO_STEM_MODEL=htdemucs AUDIO_STEM_MODEL_DIR="$TORCH_HOME/audio-separator")
  if [ "${2:-fast}" = quality ]; then
    profile=(AUDIO_STEM_SPLITTER=chain AUDIO_STEM_VOCAL_MODEL=model_bs_roformer_ep_317_sdr_12.9755.ckpt AUDIO_STEM_VOCAL_OVERLAP=4)
  else
    profile=(AUDIO_STEM_SPLITTER=demucs)
  fi
  mkdir -p "$ROOT/media"; cd "$ROOT/audio-service"
  env "${common[@]}" "${profile[@]}" setsid nohup "$V/bin/python" -m sam_audio_service > "$ROOT/svc.log" 2>&1 < /dev/null &
  echo "started (${2:-fast}); log $ROOT/svc.log" ;;
stop) pkill -f "$V/bin/python -m sam_audio_service" && echo stopped ;;
*) echo "usage: $0 setup|start [fast|quality]|stop"; exit 1 ;;
esac
