#!/bin/bash
# Build audio-service/.venv-sam: the real SAM Audio model runnable on Apple Silicon (MPS).
# macOS has no wheels for decord/xformers, so sam-audio and perception_models are installed
# without deps and the xformers imports are guarded (upstream facebookresearch/sam-audio#29).
# Development only; the Linux CUDA container stays the deployment target. ~8 GB venv + 5 GB checkpoint.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
V=.venv-sam; P=$V/bin/python
step(){ echo; echo "=== $(date +%T) $*"; }
step "create venv"; uv venv --python 3.12 $V || exit 1
step "install sam-audio + perception_models (no deps)"
uv pip install --python $P --no-deps "git+https://github.com/facebookresearch/sam-audio.git" "git+https://github.com/facebookresearch/perception_models@unpin-deps" || exit 1
step "install runtime deps"
uv pip install --python $P torch torchaudio torchvision torchcodec torchdiffeq einops numpy pydub "transformers>=4.54,<5" "huggingface_hub>=0.25,<1" audiobox_aesthetics "git+https://github.com/facebookresearch/dacvae.git" "git+https://github.com/facebookresearch/ImageBind.git" "git+https://github.com/lematt1991/CLAP.git" timm ftfy regex iopath -e . || exit 1
step "patch xformers imports (macOS has no xformers wheel; upstream sam-audio#29)"
SP=$($P -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
$P - "$SP" <<'PY'
import re, sys
from pathlib import Path
sp = Path(sys.argv[1])
for name in ["probe.py", "profiling.py", "distributed.py", "transformer.py"]:
    p = sp / "core" / name
    if not p.exists(): print("missing", p); continue
    out, changed = [], False
    for line in p.read_text().splitlines(keepends=True):
        m = re.match(r"^(from xformers[^\n]*|import xformers[^\n]*)\n$", line)
        if m:
            out.append("try:\n    " + m.group(1) + "\nexcept ImportError:  # local macOS patch (upstream sam-audio#29)\n    AttentionBias = fmha = None\n"); changed = True
        else:
            out.append(line)
    if changed: p.write_text("".join(out)); print("patched", name)
PY
step "import check"
$P -c "import sam_audio, torch; from sam_audio import SAMAudio, SAMAudioProcessor; import sam_audio_service; print('IMPORT OK torch', torch.__version__, 'mps', torch.backends.mps.is_available())" || exit 1
step "download facebook/sam-audio-small checkpoint (~5 GB)"
$P -c "from huggingface_hub import snapshot_download; print(snapshot_download('facebook/sam-audio-small'))" || exit 1
step "DONE"; df -h / | tail -1
