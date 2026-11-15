#!/usr/bin/env bash
# TidePool's default GPU separator. Uses a prepared persistent environment/cache.
set -euo pipefail
AUDIOSEP_PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUDIOSEP_WORKSPACE="${AUDIOSEP_WORKSPACE:-/workspace}"
AUDIOSEP_VENV="$AUDIOSEP_WORKSPACE/venvs/soundcanvas-audiosep"
if [ -z "${AUDIO_PYTHON:-}" ] && [ ! -x "$AUDIOSEP_VENV/bin/python" ] && [ -x "$AUDIOSEP_WORKSPACE/venvs/dog-audiosep/bin/python" ]; then
  AUDIO_PYTHON="$AUDIOSEP_WORKSPACE/venvs/dog-audiosep/bin/python"
fi
AUDIO_PYTHON="${AUDIO_PYTHON:-$AUDIOSEP_VENV/bin/python}"
export HF_HOME="${HF_HOME:-$AUDIOSEP_WORKSPACE/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$AUDIOSEP_WORKSPACE/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$AUDIOSEP_WORKSPACE/cache/pip}"
export AUDIO_AUDIOSEP_ROOT="${AUDIO_AUDIOSEP_ROOT:-$AUDIOSEP_WORKSPACE/vendor/AudioSep}"
if [ "${AUDIO_OBJECT_ENABLE:-1}" = 1 ]; then
  export AUDIO_OBJECT_PYTHON="${AUDIO_OBJECT_PYTHON:-$AUDIOSEP_WORKSPACE/venvs/soundcanvas-vision/bin/python}"
  export AUDIO_OBJECT_CHECKPOINT="${AUDIO_OBJECT_CHECKPOINT:-$HF_HOME/hub/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt}"
  export AUDIO_OBJECT_SOURCE_ROOT="${AUDIO_OBJECT_SOURCE_ROOT:-$AUDIOSEP_PROJECT_ROOT}"
  export AUDIO_OBJECT_DISCOVERY_MODEL="${AUDIO_OBJECT_DISCOVERY_MODEL:-$AUDIOSEP_WORKSPACE/checkpoints/yolo26n-seg.pt}"
fi
AUDIOSEP_PORT="${AUDIO_PORT:-8011}"
AUDIOSEP_PIDFILE="$AUDIOSEP_WORKSPACE/locks/tidepool-audiosep-$AUDIOSEP_PORT.pid"
AUDIOSEP_LOG="$AUDIOSEP_WORKSPACE/logs/tidepool-audiosep-$AUDIOSEP_PORT.log"
mkdir -p "$(dirname "$AUDIOSEP_PIDFILE")" "$(dirname "$AUDIOSEP_LOG")"
owned_pid() {
  [ -f "$AUDIOSEP_PIDFILE" ] || return 1
  python - "$AUDIOSEP_PIDFILE" "$AUDIOSEP_PROJECT_ROOT" <<'PY'
from pathlib import Path
import sys
try:
    pid=int(Path(sys.argv[1]).read_text())
    proc=Path('/proc')/str(pid)
    valid=proc.joinpath('cwd').resolve()==Path(sys.argv[2]) and b'sam_audio_service' in proc.joinpath('cmdline').read_bytes()
    if valid: print(pid)
    sys.exit(0 if valid else 1)
except (OSError,ValueError): sys.exit(1)
PY
}
case "${1:-start}" in
  setup)
    if [ ! -x "$AUDIO_PYTHON" ]; then python -m venv --system-site-packages "$AUDIOSEP_VENV"; fi
    if [ ! -d "$AUDIO_AUDIOSEP_ROOT" ]; then
      git clone https://github.com/Audio-AGI/AudioSep.git "$AUDIO_AUDIOSEP_ROOT"
      git -C "$AUDIO_AUDIOSEP_ROOT" checkout 944583f18b84589dc965de3ad77525c945334252
    fi
    if [ "$(git -C "$AUDIO_AUDIOSEP_ROOT" rev-parse HEAD)" != 944583f18b84589dc965de3ad77525c945334252 ]; then
      echo 'AudioSep checkout revision differs. Prepare the pinned checkout without overwriting existing work.' >&2; exit 1
    fi
    AUDIOSEP_PATCH="$AUDIOSEP_PROJECT_ROOT/audio-service/patches/audiosep-checkpoint-compat.patch"
    if git -C "$AUDIO_AUDIOSEP_ROOT" apply --check "$AUDIOSEP_PATCH" 2>/dev/null; then
      git -C "$AUDIO_AUDIOSEP_ROOT" apply "$AUDIOSEP_PATCH"
    else git -C "$AUDIO_AUDIOSEP_ROOT" apply --reverse --check "$AUDIOSEP_PATCH"; fi
    "$AUDIO_PYTHON" -c 'import importlib.metadata as m;print("\n".join(n+"=="+m.version(n) for n in ("torch","torchaudio","torchvision")))' > "$AUDIOSEP_WORKSPACE/audiosep-torch-constraints.txt"
    "$AUDIO_PYTHON" -m pip install --no-compile -c "$AUDIOSEP_WORKSPACE/audiosep-torch-constraints.txt" -e "$AUDIOSEP_PROJECT_ROOT/audio-service[audiosep]"
    echo 'AudioSep runtime prepared. Official checkpoints are cached/reused on startup.'
    ;;
  start)
    [ -x "$AUDIO_PYTHON" ] || { echo 'Run scripts/runpod_audiosep.sh setup first.' >&2; exit 1; }
    if pid=$(owned_pid); then echo "AudioSep is already running as $pid"; exit 0; fi
    python - "$AUDIOSEP_PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(('127.0.0.1',int(sys.argv[1])))
PY
    cd "$AUDIOSEP_PROJECT_ROOT"
    nohup env PYTHONPATH="$AUDIOSEP_PROJECT_ROOT/audio-service${PYTHONPATH:+:$PYTHONPATH}" \
      AUDIO_BACKEND=audiosep AUDIO_MODEL_ID=audiosep-base-4M AUDIO_DEVICE=cuda \
      AUDIO_HOST=127.0.0.1 AUDIO_PORT="$AUDIOSEP_PORT" AUDIO_CHUNK_BATCH=1 AUDIO_CHUNK_SECONDS=10 \
      AUDIO_SUPPRESSION="${AUDIO_SUPPRESSION:-conservative}" AUDIO_SUPPRESSION_STRENGTH="${AUDIO_SUPPRESSION_STRENGTH:-0.9}" \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      "$AUDIO_PYTHON" -m sam_audio_service > "$AUDIOSEP_LOG" 2>&1 < /dev/null &
    echo $! > "$AUDIOSEP_PIDFILE"
    echo "AudioSep starting on 127.0.0.1:$AUDIOSEP_PORT; log: $AUDIOSEP_LOG"
    ;;
  status)
    curl -fsS "http://127.0.0.1:$AUDIOSEP_PORT/ready"
    ;;
  stop)
    if pid=$(owned_pid); then kill "$pid"; rm -f "$AUDIOSEP_PIDFILE"; echo 'Stopped this workspace AudioSep service.'
    else echo 'No owned AudioSep service is running.'; fi
    ;;
  *) echo 'Usage: scripts/runpod_audiosep.sh setup|start|status|stop' >&2; exit 2 ;;
esac
