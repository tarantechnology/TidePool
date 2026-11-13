#!/bin/bash
# One-shot setup of the SoundCanvas audio service on a RunPod pod or any Ubuntu/Debian VM with an NVIDIA GPU (H100 etc).
# Run ON THE POD as root:  bash runpod_setup.sh [branch]      (HF_TOKEN must be in the environment)
# Afterwards:  bash /workspace/soundcanvas/scripts/runpod_setup.sh start|stop|status|bench
set -euo pipefail
# Work directory: /workspace on RunPod, else $HOME/soundcanvas-work; override with WS=... .
WS=${WS:-$([ -d /workspace ] && echo /workspace || echo "$HOME/soundcanvas-work")}; mkdir -p "$WS"
REPO=$WS/soundcanvas; VENV=$WS/venv; PY=$VENV/bin/python
BRANCH=${1:-main}; LOG=$WS/audio.log
cmd=${1:-setup}
model_for_gpu() {
  local mb; mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "${AUDIO_MODEL_ID:-}" ]; then echo "$AUDIO_MODEL_ID"; elif [ "$mb" -ge 40000 ]; then echo facebook/sam-audio-large; else echo facebook/sam-audio-small; fi
}
chunk_for_gpu() {
  local mb; mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "$mb" -ge 70000 ]; then echo 30; elif [ "$mb" -ge 40000 ]; then echo 20; else echo 10; fi   # H100 80 GB -> 30 s windows
}
case "$cmd" in
  start)
    MODEL=$(model_for_gpu); CHUNK=$(chunk_for_gpu)
    pkill -f sam_audio_service 2>/dev/null || true; sleep 1
    cd $REPO/audio-service
    AUDIO_BACKEND=sam AUDIO_MODEL_ID=$MODEL AUDIO_DEVICE=cuda AUDIO_CHUNK_SECONDS=$CHUNK AUDIO_CHUNK_BATCH=2 \
      AUDIO_LOAD_SPAN_PREDICTOR=${AUDIO_LOAD_SPAN_PREDICTOR:-false} AUDIO_LOAD_RANKERS=${AUDIO_LOAD_RANKERS:-false} \
      AUDIO_MEDIA_ROOT=$WS/media AUDIO_PORT=8001 nohup $PY -m sam_audio_service > $LOG 2>&1 < /dev/null & disown
    echo "audio service starting with $MODEL (chunk ${CHUNK}s); log: $LOG"
    t0=$(date +%s); until curl -sf http://127.0.0.1:8001/ready >/dev/null; do sleep 5; [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "not ready after 30 min; tail $LOG"; exit 1; }; done
    echo "ready in $(( $(date +%s)-t0 ))s"; curl -s http://127.0.0.1:8001/health | $PY -c "import sys,json;m=json.load(sys.stdin)['model'];print({k:m[k] for k in ('model_id','device','loaded','load_seconds','parameters')})"
    ;;
  stop) pkill -f sam_audio_service && echo stopped || echo "not running" ;;
  status) curl -s http://127.0.0.1:8001/health || echo "not running"; echo; nvidia-smi --query-gpu=memory.used,memory.total --format=csv ;;
  bench)  # full quality sweeps on the GPU (results in /workspace/bench)
    shift; cd $REPO/audio-service
    AUDIO_BACKEND=sam AUDIO_MODEL_ID=$(model_for_gpu) AUDIO_DEVICE=cuda AUDIO_LOAD_SPAN_PREDICTOR=false AUDIO_LOAD_RANKERS=false \
      $PY -m benchmarks.run_benchmark --backend sam --seconds 6 --latency-seconds 15 --long-seconds 24 --speech-mode synthetic --seed 1234 "$@"
    ;;
  setup|*)
    [ "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (needed for the gated facebook/sam-audio-* weights)"; exit 1; }
    export DEBIAN_FRONTEND=noninteractive
    command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg git > /dev/null; }
    command -v uv >/dev/null || pip install -q uv
    # The repo is private: either push it here from a machine with access
    #   git push ssh://root@HOST:PORT/workspace/soundcanvas BRANCH   (after: git init + receive.denyCurrentBranch=updateInstead)
    # or clone it with credentials of your own before running this script.
    if [ -d $REPO/.git ]; then
      git -C $REPO fetch -q origin "$BRANCH" 2>/dev/null && git -C $REPO checkout -q "$BRANCH" 2>/dev/null && git -C $REPO merge -q --ff-only FETCH_HEAD 2>/dev/null || git -C $REPO checkout -q "$BRANCH"
    else
      git clone -q https://github.com/AkhilNam/SoundCanvas.git $REPO && git -C $REPO checkout -q "$BRANCH"
    fi
    echo "repo at $(git -C $REPO rev-parse --short HEAD) ($(git -C $REPO branch --show-current))"
    [ -x $PY ] || uv venv --python 3.12 $VENV
    uv pip install --python $PY -e "$REPO/audio-service[sam,dev]"
    $PY -c "import torch, sam_audio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    mkdir -p ~/.cache/huggingface && printf '%s' "$HF_TOKEN" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
    MODEL=$(model_for_gpu); echo "prefetching $MODEL"
    $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
    df -h $WS | tail -1; echo "setup done: next  bash $REPO/scripts/runpod_setup.sh start"
    ;;
esac
