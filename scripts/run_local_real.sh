#!/bin/bash
# Start the whole SoundCanvas stack on this Mac with the REAL SAM Audio small model on Apple MPS.
#   scripts/run_local_real.sh          start audio service (8001), backend (8000), editor (5173)
#   scripts/run_local_real.sh stop     stop all three
#   scripts/run_local_real.sh mock     same, but mock audio backend (instant, no real separation)
#   scripts/run_local_real.sh remote   audio service on a GPU host reached through an SSH tunnel on 127.0.0.1:8001 (see scripts/runpod_setup.sh)
# Logs: logs/audio.log logs/backend.log logs/frontend.log
# Secrets/settings: ./.env (see .env.example), e.g. OPENAI_API_KEY=... enables the natural-language command bar.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs storage
# Optional settings (OPENAI_API_KEY for the command bar, VISION_DEVICE, ...): put them in ./.env (gitignored).
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export VISION_DEVICE="${VISION_DEVICE:-mps}"   # YOLO on the Apple GPU: ~30% faster than cpu on an M1 Max
if [ "${1:-}" = stop ]; then
  pkill -f "sam_audio_service" ; pkill -f "uvicorn app.main:app" ; pkill -f "vite" ; echo "stopped"; exit 0
fi
MODE=${1:-real}
if [ "$MODE" = real ]; then
  PY="$ROOT/audio-service/.venv-sam/bin/python"
  [ -x "$PY" ] || { echo "missing $PY; run the build first"; exit 1; }
  ( cd audio-service && PYTORCH_ENABLE_MPS_FALLBACK=1 AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-small AUDIO_DEVICE=mps \
    AUDIO_LOAD_SPAN_PREDICTOR=false AUDIO_LOAD_RANKERS=false AUDIO_CHUNK_SECONDS=15 AUDIO_MEDIA_ROOT="$ROOT/storage" AUDIO_PORT=8001 \
    nohup "$PY" -m sam_audio_service > "$ROOT/logs/audio.log" 2>&1 < /dev/null & disown )
elif [ "$MODE" = remote ]; then
  curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "nothing answers on 127.0.0.1:8001; open the tunnel first: ssh -N -L 8001:127.0.0.1:8001 -p PORT -i ~/.ssh/id_ed25519 root@HOST &"; exit 1; }
else
  ( cd audio-service && AUDIO_BACKEND=mock AUDIO_MEDIA_ROOT="$ROOT/storage" AUDIO_PORT=8001 \
    nohup .venv/bin/python -m sam_audio_service > "$ROOT/logs/audio.log" 2>&1 < /dev/null & disown )
fi
STORAGE_ROOT="$ROOT/storage" INFERENCE_ADAPTER=http SAM_AUDIO_SERVICE_URL=http://127.0.0.1:8001/separate INFERENCE_TIMEOUT_SECONDS=3600 \
  nohup audio-service/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > logs/backend.log 2>&1 < /dev/null & disown
( cd frontend && VITE_API_BASE_URL=/api nohup npm run dev -- --port 5173 > "$ROOT/logs/frontend.log" 2>&1 < /dev/null & disown )
echo "audio service ($MODE) -> http://127.0.0.1:8001   backend -> http://127.0.0.1:8000   editor -> http://localhost:5173/#/app"
if [ "$MODE" = real ]; then
  echo -n "loading model"; t0=$(date +%s)
  until curl -sf http://127.0.0.1:8001/ready >/dev/null; do sleep 5; echo -n .; [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo; echo "model not ready after 20 min; see logs/audio.log"; exit 1; }; done
  echo " ready in $(( $(date +%s) - t0 ))s"
fi
echo "open http://localhost:5173/#/app   (stop with: scripts/run_local_real.sh stop)"
