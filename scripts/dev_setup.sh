#!/usr/bin/env bash
# Local (no-Docker) backend setup: API requirements, the vision package that object detection
# runs as `python -m soundcanvas_cv`, and the default detector weights.
#
#   python3 -m venv .venv && source .venv/bin/activate   # recommended
#   scripts/dev_setup.sh
#
# Uses $PYTHON for everything, so install into the interpreter that will run uvicorn: the
# worker launches the detector with that same interpreter. Default: the project's
# audio-service/.venv if it exists (it already carries torch), else python3. Python >= 3.11 is
# required (the vision package refuses older interpreters), and that is checked before anything
# is installed so a stray conda/system Python is never modified. Weights go to
# $STORAGE_ROOT/vision (default ./storage/vision), where the backend looks for them.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PYTHON:-}" ]; then
  if [ -x audio-service/.venv/bin/python ]; then PYTHON=audio-service/.venv/bin/python; else PYTHON=python3; fi
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "$PYTHON is $("$PYTHON" --version 2>&1); SoundCanvas needs Python 3.11+." >&2
  echo "Create one (python3.12 -m venv .venv && source .venv/bin/activate) or set PYTHON=/path/to/python3.11+ and rerun." >&2
  exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"

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
