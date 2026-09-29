"""Kaggle GPU quality calibration for Mr. Uncut using full EchoMimicV2.

This calibration deliberately favors identity and temporal consistency over
speed. Heavy dependencies live under /kaggle/temp so Kaggle does not publish
them as kernel outputs. Only the final audio/video/status files are written to
/kaggle/working.
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
RENDER_SIZE = 768
TEST_FRAMES = 72
FPS = 24
STEPS = 30
CFG = 2.5
CONTEXT_FRAMES = 12
CONTEXT_OVERLAP = 8
SEED = 3407

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/mr-uncut-v2")
REPO = TEMP / "personagemia"
ECHO_ROOT = TEMP / "echomimic_v2"
HF_CACHE = TEMP / "hf-cache"
OUTPUT = WORK / "mr_uncut_kaggle_test.mp4"
AUDIO = WORK / "mr_uncut_voice.wav"
INFO = WORK / "mr_uncut_run_info.json"
ERROR = WORK / "mr_uncut_error.txt"

os.environ.update(
    {
        "PIP_NO_CACHE_DIR": "1",
        "HF_HOME": str(HF_CACHE),
        "HUGGINGFACE_HUB_CACHE": str(HF_CACHE / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sh(command: str, *, cwd: Path | None = None) -> None:
    print("+", command, flush=True)
    subprocess.run(["bash", "-lc", command], cwd=cwd, check=True)


def disk_report(label: str) -> None:
    usage = shutil.disk_usage("/kaggle")
    gib = 1024**3
    print(
        f"DISK {label}: total={usage.total/gib:.1f}GiB "
        f"used={usage.used/gib:.1f}GiB free={usage.free/gib:.1f}GiB",
        flush=True,
    )


def install_runtime() -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    disk_report("start")
    run(["nvidia-smi"])
    sh("apt-get -qq update && apt-get -qq -y install espeak-ng ffmpeg git")

    if REPO.exists():
        shutil.rmtree(REPO)
    if ECHO_ROOT.exists():
        shutil.rmtree(ECHO_ROOT)

    run(["git", "clone", "--depth", "1", "https://github.com/thebusinessflowtv/personagemia.git", str(REPO)])
    run(["git", "clone", "--depth", "1", "https://github.com/antgroup/echomimic_v2.git", str(ECHO_ROOT)])

    packages = [
        "numpy==1.26.4",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "torchmetrics",
        "torchtyping",
        "einops==0.8.0",
        "omegaconf==2.3.0",
        "opencv-python-headless==4.10.0.84",
        "av==13.1.0",
        "decord==0.6.0",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "scipy==1.14.1",
        "ffmpeg-python",
        "soundfile",
        "moviepy==1.0.3",
        "huggingface_hub>=0.27",
        "matplotlib",
        "kokoro>=0.9.4",
        "misaki[en]",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *packages])

    sh(
        f"{sys.executable} - <<'PY'\n"
        "from transformers import AlbertModel\n"
        "from accelerate import __version__ as accelerate_version\n"
        "import transformers\n"
        "print('transformers', transformers.__version__)\n"
        "print('accelerate', accelerate_version)\n"
        "print('dependency probe OK:', AlbertModel.__name__)\n"
        "PY"
    )
    disk_report("after runtime")


def download_models() -> dict[str, str]:
    from huggingface_hub import hf_hub_download, snapshot_download

    ckpt_repo = "BadToBest/EchoMimicV2"
    checkpoints = {}
    for filename in (
        "denoising_unet.pth",
        "reference_unet.pth",
        "pose_encoder.pth",
        "motion_module.pth",
    ):
        checkpoints[filename] = hf_hub_download(
            repo_id=ckpt_repo,
            filename=filename,
            cache_dir=HF_CACHE,
        )
        disk_report(f"after {filename}")

    base_model = snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers",
        allow_patterns=["unet/*"],
        cache_dir=HF_CACHE,
    )
    disk_report("after base UNet")

    vae_model = snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        allow_patterns=["config.json", "diffusion_pytorch_model.*"],
        cache_dir=HF_CACHE,
    )
    disk_report("after VAE")

    audio_dir = TEMP / "audio_processor"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tiny = audio_dir / "tiny.pt"
    if not tiny.exists():
        run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "-o",
                str(tiny),
                "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
            ]
        )
    disk_report("models ready")

    return {
        "base": str(base_model),
        "vae": str(vae_model),
        "denoising": checkpoints["denoising_unet.pth"],
        "reference": checkpoints["reference_unet.pth"],
        "pose": checkpoints["pose_encoder.pth"],
        "motion": checkpoints["motion_module.pth"],
        "audio": str(tiny),
    }


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
    print(f"Audio ready: {AUDIO} ({AUDIO.stat().st_size/1024/1024:.1f} MB)", flush=True)


def prepare_reference() -> Path:
    from PIL import Image

    source = REPO / "assets/mr_uncut_master.png"
    if not source.exists():
        raise FileNotFoundError(f"Missing Mr. Uncut master image: {source}")

    image = Image.open(source).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = max(0, (w - side) // 2)
    top = 0 if h > w else max(0, (h - side) // 2)
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((RENDER_SIZE, RENDER_SIZE), Image.Resampling.LANCZOS)

    target = TEMP / "mr_uncut_reference.png"
    image.save(target, optimize=True)
    print(
        f"Reference ready: {target} source={w}x{h} crop=({left},{top},{left + side},{top + side}) "
        f"output={RENDER_SIZE}x{RENDER_SIZE}",
        flush=True,
    )
    return target


def prepare_neutral_pose() -> Path:
    """Keep EchoMimic's official pose schema but remove all hand motion."""
    import copy
    import numpy as np

    pose_dir = TEMP / "mr_uncut_neutral_pose"
    if pose_dir.exists():
        shutil.rmtree(pose_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)

    template_path = ECHO_ROOT / "assets/halfbody_demo/pose/01/0.npy"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing official EchoMimic pose template: {template_path}")

    template = np.load(template_path, allow_pickle=True).tolist()
    if "bodies" not in template or "hands" not in template or "hands_score" not in template:
        raise RuntimeError("Official EchoMimic pose template has an unexpected schema")

    for index in range(TEST_FRAMES):
        neutral_pose = copy.deepcopy(template)
        neutral_pose["hands"] = np.zeros_like(neutral_pose["hands"], dtype=np.float32)
        neutral_pose["hands_score"] = np.zeros_like(neutral_pose["hands_score"], dtype=np.float32)
        np.save(pose_dir / f"{index}.npy", neutral_pose, allow_pickle=True)

    print(
        f"Neutral hand conditioning ready from official schema: {pose_dir} "
        f"({TEST_FRAMES} frames / {TEST_FRAMES / FPS:.1f}s)",
        flush=True,
    )
    return pose_dir


def render(models: dict[str, str]) -> None:
    reference = prepare_reference()
    pose_dir = prepare_neutral_pose()

    config = TEMP / "mr_uncut_v2_quality.yaml"
    inference_config = ECHO_ROOT / "configs/inference/inference_v2.yaml"
    config.write_text(
        "\n".join(
            [
                f'pretrained_base_model_path: "{models["base"]}"',
                f'pretrained_vae_path: "{models["vae"]}"',
                f'denoising_unet_path: "{models["denoising"]}"',
                f'reference_unet_path: "{models["reference"]}"',
                f'pose_encoder_path: "{models["pose"]}"',
                f'motion_module_path: "{models["motion"]}"',
                f'audio_model_path: "{models["audio"]}"',
                f'inference_config: "{inference_config}"',
                "weight_dtype: 'fp16'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    disk_report("before quality render")
    os.environ["FFMPEG_PATH"] = "/usr/bin"
    run(
        [
            sys.executable,
            "infer.py",
            "--config",
            str(config),
            "-W",
            str(RENDER_SIZE),
            "-H",
            str(RENDER_SIZE),
            "-L",
            str(TEST_FRAMES),
            "--steps",
            str(STEPS),
            "--cfg",
            str(CFG),
            "--fps",
            str(FPS),
            "--seed",
            str(SEED),
            "--context_frames",
            str(CONTEXT_FRAMES),
            "--context_overlap",
            str(CONTEXT_OVERLAP),
            "--ref_images_dir",
            str(TEMP),
            "--refimg_name",
            reference.name,
            "--audio_dir",
            str(WORK),
            "--audio_name",
            AUDIO.name,
            "--pose_dir",
            str(TEMP),
            "--pose_name",
            pose_dir.name,
        ],
        cwd=ECHO_ROOT,
    )

    candidates = sorted(
        (ECHO_ROOT / "outputs").rglob("*_sig.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Full EchoMimicV2 completed without producing *_sig.mp4")
    shutil.copy2(candidates[0], OUTPUT)
    print(f"Video ready: {OUTPUT} ({OUTPUT.stat().st_size/1024/1024:.1f} MB)", flush=True)


def main() -> None:
    script = base64.b64decode(SCRIPT_B64).decode("utf-8")
    if not script.strip():
        raise RuntimeError("Test script is empty")

    print("=== Mr. Uncut Kaggle quality calibration / full EchoMimicV2 ===", flush=True)
    print(f"Voice preset: {VOICE}", flush=True)
    print(f"Script: {script}", flush=True)
    print(f"Render size: {RENDER_SIZE}x{RENDER_SIZE}", flush=True)
    print(f"Calibration length: {TEST_FRAMES} frames ({TEST_FRAMES / FPS:.1f}s)", flush=True)
    print(
        f"Quality settings: steps={STEPS} cfg={CFG} context={CONTEXT_FRAMES} overlap={CONTEXT_OVERLAP} seed={SEED}",
        flush=True,
    )

    install_runtime()
    make_audio(script)
    models = download_models()
    render(models)

    INFO.write_text(
        json.dumps(
            {
                "status": "completed",
                "engine": "EchoMimicV2-full",
                "mode": "identity-temporal-consistency-calibration",
                "video": OUTPUT.name,
                "audio": AUDIO.name,
                "voice": VOICE,
                "resolution": f"{RENDER_SIZE}x{RENDER_SIZE}",
                "fps": FPS,
                "frames": TEST_FRAMES,
                "duration_seconds": TEST_FRAMES / FPS,
                "steps": STEPS,
                "cfg": CFG,
                "context_frames": CONTEXT_FRAMES,
                "context_overlap": CONTEXT_OVERLAP,
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    disk_report("done")
    print(f"DONE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        ERROR.write_text(error, encoding="utf-8")
        disk_report("failure")
        raise
