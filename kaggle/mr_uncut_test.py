"""Kaggle GPU proof-of-concept for Mr. Uncut.

This file is used as a template by the GitHub Actions workflow. The workflow replaces
SCRIPT_B64 and VOICE before pushing the kernel to Kaggle.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

SCRIPT_B64 = "__SCRIPT_B64__"
VOICE = "__VOICE__"

WORK = Path("/kaggle/working")
REPO = WORK / "personagemia"
AI_ROOT = WORK / "personagemia-ai"
ECHO_ROOT = AI_ROOT / "engines/echomimic_v3"
MODEL_ROOT = AI_ROOT / "models/echomimic_v3/flash"
OUTPUT = WORK / "mr_uncut_kaggle_test.mp4"
AUDIO = WORK / "mr_uncut_voice.wav"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sh(command: str, *, cwd: Path | None = None) -> None:
    print("+", command, flush=True)
    subprocess.run(["bash", "-lc", command], cwd=cwd, check=True)


def install_runtime() -> None:
    run(["nvidia-smi"])
    sh("apt-get -qq update && apt-get -qq -y install espeak-ng ffmpeg git")

    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", "https://github.com/thebusinessflowtv/personagemia.git", str(REPO)])

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "kokoro>=0.9.4",
            "soundfile",
            "misaki[en]",
            "huggingface_hub>=0.27",
            "pyloudnorm",
        ]
    )

    ECHO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not ECHO_ROOT.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/antgroup/echomimic_v3.git", str(ECHO_ROOT)])

    # Reuse Kaggle's CUDA-enabled PyTorch instead of downloading another giant torch wheel.
    filtered = WORK / "echomimic-requirements.txt"
    requirements = (ECHO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    skipped_prefixes = ("torch", "tensorflow", "gradio", "tensorboard", "retina-face")
    filtered.write_text(
        "\n".join(
            line
            for line in requirements
            if line.strip() and not line.strip().lower().startswith(skipped_prefixes)
        )
        + "\n",
        encoding="utf-8",
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(filtered)])


def download_models() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
        local_dir=MODEL_ROOT / "Wan2.1-Fun-V1.1-1.3B-InP",
    )
    snapshot_download(
        repo_id="TencentGameMate/chinese-wav2vec2-base",
        local_dir=MODEL_ROOT / "chinese-wav2vec2-base",
    )
    source = Path(
        hf_hub_download(
            repo_id="BadToBest/EchoMimicV3",
            filename="echomimicv3-flash-pro/diffusion_pytorch_model.safetensors",
        )
    )
    target = MODEL_ROOT / "transformer/diffusion_pytorch_model.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)


def make_audio(script: str) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    voices = {
        "puck": "am_puck",
        "adam": "am_adam",
        "liam": "am_liam",
    }
    voice = voices.get(VOICE, "am_puck")
    pipeline = KPipeline(lang_code="a")
    chunks = [audio for _, _, audio in pipeline(script, voice=voice, speed=1.13)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    sf.write(AUDIO, np.concatenate(chunks), 24000)


def render() -> None:
    os.environ.update(
        {
            "PERSONAGEMIA_AI_ROOT": str(AI_ROOT),
            "PERSONAGEMIA_ECHO_ROOT": str(ECHO_ROOT),
            "PERSONAGEMIA_ECHO_PYTHON": sys.executable,
            "PERSONAGEMIA_ECHO_MODEL_ROOT": str(MODEL_ROOT),
            "PERSONAGEMIA_ECHO_OUTPUT_DIR": str(WORK / "echo-output"),
            # T4/Turing: use FP16 rather than BF16 for maximum compatibility.
            "PERSONAGEMIA_ECHO_WEIGHT_DTYPE": "float16",
            # Start conservatively. Once the proof-of-concept passes we can benchmark 640/768.
            "PERSONAGEMIA_ECHO_SIZE": "512",
            "PERSONAGEMIA_ECHO_STEPS": "8",
            "PERSONAGEMIA_ECHO_AUDIO_GUIDANCE_SCALE": "2.0",
        }
    )
    sys.path.insert(0, str(REPO))
    from app.local_avatar import EchoMimicFlashEngine

    image = REPO / "assets/mr_uncut_master.jpg"
    EchoMimicFlashEngine().render(image, AUDIO, OUTPUT)


def main() -> None:
    script = base64.b64decode(SCRIPT_B64).decode("utf-8")
    if not script.strip():
        raise RuntimeError("Test script is empty")

    print("=== Mr. Uncut Kaggle GPU test ===", flush=True)
    print(f"Voice preset: {VOICE}", flush=True)
    print(f"Script: {script}", flush=True)

    install_runtime()
    download_models()
    make_audio(script)
    render()

    info = {
        "status": "completed",
        "video": OUTPUT.name,
        "audio": AUDIO.name,
        "voice": VOICE,
        "echo_size": 512,
        "weight_dtype": "float16",
    }
    (WORK / "mr_uncut_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"DONE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        (WORK / "mr_uncut_error.txt").write_text(error, encoding="utf-8")
        raise
