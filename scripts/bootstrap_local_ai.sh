#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${PERSONAGEMIA_AI_ROOT:-$HOME/.personagemia-ai}"
PY311="${PYTHON311:-python3.11}"
PY310="${PYTHON310:-python3.10}"
ECHO_REPO="$AI_ROOT/engines/echomimic_v3"
MODEL_ROOT="$AI_ROOT/models/echomimic_v3/flash"
LATENT_ROOT="$AI_ROOT/engines/LatentSync"

mkdir -p "$AI_ROOT/engines" "$AI_ROOT/venvs" "$MODEL_ROOT/transformer"

echo "== PersonagemIA local bootstrap =="
echo "AI root: $AI_ROOT"

for binary in git ffmpeg "$PY311"; do
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "Missing required command: $binary" >&2
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi was not found. TTS can still be installed, but avatar generation needs an NVIDIA GPU."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

create_venv() {
  local name="$1"
  local python_bin="${2:-$PY311}"
  if [ ! -x "$AI_ROOT/venvs/$name/bin/python" ]; then
    "$python_bin" -m venv "$AI_ROOT/venvs/$name"
  fi
  "$AI_ROOT/venvs/$name/bin/python" -m pip install --upgrade pip wheel setuptools
}

echo "[1/6] Installing Chatterbox TTS runtime..."
create_venv chatterbox
"$AI_ROOT/venvs/chatterbox/bin/pip" install chatterbox-tts

echo "[2/6] Installing Kokoro fallback runtime..."
create_venv kokoro
"$AI_ROOT/venvs/kokoro/bin/pip" install 'kokoro>=0.9.4' soundfile 'misaki[en]'

echo "[3/6] Installing EchoMimicV3-Flash runtime..."
if [ ! -d "$ECHO_REPO/.git" ]; then
  git clone https://github.com/antgroup/echomimic_v3.git "$ECHO_REPO"
else
  git -C "$ECHO_REPO" pull --ff-only
fi
create_venv echomimic
"$AI_ROOT/venvs/echomimic/bin/pip" install -r "$ECHO_REPO/requirements.txt"
"$AI_ROOT/venvs/echomimic/bin/pip" install 'huggingface_hub>=0.27'

echo "[4/6] Downloading EchoMimicV3-Flash weights (large one-time download)..."
AI_ROOT="$AI_ROOT" "$AI_ROOT/venvs/echomimic/bin/python" - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(os.environ["AI_ROOT"])
model_root = root / "models/echomimic_v3/flash"
model_root.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    local_dir=model_root / "Wan2.1-Fun-V1.1-1.3B-InP",
)
snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    local_dir=model_root / "chinese-wav2vec2-base",
)
source = Path(
    hf_hub_download(
        repo_id="BadToBest/EchoMimicV3",
        filename="echomimicv3-flash-pro/diffusion_pytorch_model.safetensors",
    )
)
target = model_root / "transformer/diffusion_pytorch_model.safetensors"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists() or target.stat().st_size != source.stat().st_size:
    shutil.copy2(source, target)
print(f"EchoMimic models ready at {model_root}")
PY

echo "[5/6] Installing LatentSync 1.5 temporal mouth/teeth refinement..."
if [ ! -d "$LATENT_ROOT/.git" ]; then
  git clone https://github.com/bytedance/LatentSync.git "$LATENT_ROOT"
else
  git -C "$LATENT_ROOT" pull --ff-only
fi

LATENT_PY="$PY310"
if ! command -v "$LATENT_PY" >/dev/null 2>&1; then
  echo "python3.10 not found; using $PY311 for LatentSync." >&2
  LATENT_PY="$PY311"
fi
create_venv latentsync "$LATENT_PY"
"$AI_ROOT/venvs/latentsync/bin/pip" install -r "$LATENT_ROOT/requirements.txt"
LATENT_ROOT="$LATENT_ROOT" "$AI_ROOT/venvs/latentsync/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(os.environ["LATENT_ROOT"])
checkpoints = root / "checkpoints"
checkpoints.mkdir(parents=True, exist_ok=True)
for filename in ("latentsync_unet.pt", "whisper/tiny.pt"):
    hf_hub_download(
        repo_id="ByteDance/LatentSync-1.5",
        filename=filename,
        local_dir=checkpoints,
    )
print(f"LatentSync 1.5 models ready at {checkpoints}")
PY

echo "[6/6] Optional MuseTalk 1.5 comparison setup..."
if [ "${INSTALL_MUSETALK:-0}" = "1" ]; then
  MUSE_ROOT="$AI_ROOT/engines/MuseTalk"
  if [ ! -d "$MUSE_ROOT/.git" ]; then
    git clone https://github.com/TMElyralab/MuseTalk.git "$MUSE_ROOT"
  else
    git -C "$MUSE_ROOT" pull --ff-only
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    if [ ! -x "$AI_ROOT/venvs/musetalk/bin/python" ]; then
      python3.10 -m venv "$AI_ROOT/venvs/musetalk"
    fi
    "$AI_ROOT/venvs/musetalk/bin/pip" install --upgrade pip wheel setuptools
    "$AI_ROOT/venvs/musetalk/bin/pip" install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    "$AI_ROOT/venvs/musetalk/bin/pip" install -r "$MUSE_ROOT/requirements.txt"
    echo "MuseTalk code/runtime installed. Run the official weight downloader before enabling it."
  else
    echo "MuseTalk requested but python3.10 is missing; skipping its venv." >&2
  fi
else
  echo "MuseTalk skipped; LatentSync 1.5 is the default Quality V2 refiner."
fi

echo
echo "Bootstrap complete."
echo "Run: personagemia local-doctor"
echo "Then: personagemia local-test --script \"I'm Mr. Uncut. Here's what I really think.\""
