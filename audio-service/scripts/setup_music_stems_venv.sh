#!/bin/bash
# Phase 0 music-stems benchmark environment on the A40 pod (Linux x86_64, NVIDIA driver with
# CUDA >= 12.8, Python 3.12, uv, ffmpeg, git).
#
# Creates ONE venv with torch 2.8.0 cu128 (the AudioSep service's torch) plus Demucs v4,
# python-audio-separator (RoFormer models), the AudioSet taggers (AST via transformers, PANNs)
# and AudioSep's own runtime deps, i.e. it is also the co-install test for loading the splitter
# in the same process as AudioSep. Optionally prepares the AudioSep checkout + checkpoints.
#
# Usage: bash setup_music_stems_venv.sh [venv_path] [audiosep_root]
#   defaults: /workspace/venv-stems  /workspace/AudioSep
# Activate: source /workspace/venv-stems/bin/activate
#           export TORCH_HOME=/workspace/torch-cache HF_HOME=/workspace/hf
# The exact versions that resolved are written to $V/freeze.txt (see the Phase 0 REPORT.md pins).
set -euo pipefail
V=${1:-/workspace/venv-stems}
A=${2:-/workspace/AudioSep}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=${AUDIOSEP_PATCH:-$HERE/../patches/audiosep-checkpoint-compat.patch}
export UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
# uv cache stays on the local overlay disk, not the quota'd network volume.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}

[ -x "$V/bin/python" ] || uv venv -q --python 3.12 "$V"
PY="$V/bin/python"

# 1. torch from the cu128 index (explicit; PyPI's linux wheel is also cu128).
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0" "torchaudio==2.8.0"

# 2. Everything else; torch pinned again so no resolver step can move it.
#    PINS (set after Phase 0) can override the unpinned names via a constraints file.
uv pip install --python "$PY" ${CONSTRAINTS:+-c "$CONSTRAINTS"} \
  "torch==2.8.0" "torchaudio==2.8.0" \
  "demucs==4.0.1" "audio-separator[gpu]" transformers panns_inference "bs-roformer==0.3.10" \
  musdb soundfile scipy julius librosa audioread pyarrow \
  lightning torchlibrosa h5py ftfy braceexpand webdataset wget huggingface_hub pyyaml pandas tqdm scikit-learn regex

"$PY" - <<'PYEOF'
import torch, importlib.metadata as md
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
for p in ['torchaudio', 'demucs', 'audio-separator', 'onnxruntime-gpu', 'transformers', 'panns-inference', 'lightning', 'numpy', 'julius']:
    try: print(p, md.version(p))
    except Exception as e: print(p, 'MISSING', e)
PYEOF
uv pip freeze --python "$PY" > "$V/freeze.txt"

# 3. AudioSep checkout (pinned commit + repo patch) and checkpoints, for the text-prompt baseline.
if [ "${SKIP_AUDIOSEP:-0}" != 1 ]; then
  [ -d "$A/.git" ] || git clone -q https://github.com/Audio-AGI/AudioSep.git "$A"
  git -C "$A" checkout -q 944583f18b84589dc965de3ad77525c945334252
  if git -C "$A" apply --check "$PATCH" 2>/dev/null; then git -C "$A" apply "$PATCH"; echo patched; else echo "patch not applied (already applied?)"; fi
  mkdir -p "$A/checkpoint"
  for f in audiosep_base_4M_steps.ckpt music_speech_audioset_epoch_15_esc_89.98.pt; do
    [ -s "$A/checkpoint/$f" ] || curl -fsSL --retry 3 -o "$A/checkpoint/$f" \
      "https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/$f"
  done
  ls -la "$A/checkpoint"
fi
echo SETUP_DONE
