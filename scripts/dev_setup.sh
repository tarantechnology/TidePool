#!/usr/bin/env bash
# Local (no-Docker) backend setup: API requirements, the vision package that object detection
# runs as `python -m soundcanvas_cv`, and the default detector weights.
#
#   python3 -m venv .venv && source .venv/bin/activate   # recommended
#   scripts/dev_setup.sh
#
# Uses $PYTHON (default: python) for everything, so install into the interpreter that will run
# uvicorn: the worker launches the detector with that same interpreter. Weights go to
# $STORAGE_ROOT/vision (default ./storage/vision), where the backend looks for them.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; install FFmpeg first (macOS: brew install ffmpeg)." >&2; exit 1; }
done

"$PYTHON" -m pip install -r requirements.txt
if [ "$(uname -s)" = Linux ] && ! "$PYTHON" -c 'import torch' 2>/dev/null; then
  # Same CPU wheels as the Docker image; avoids pulling multi-GB CUDA builds by default.
  "$PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
"$PYTHON" -m pip install -e ./vision
"$PYTHON" -m app.detector --prefetch
echo "Done. Start the API with: $PYTHON -m uvicorn app.main:app --reload --port 8000"
