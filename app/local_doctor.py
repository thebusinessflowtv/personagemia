import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(settings: Settings | None = None) -> list[Check]:
    settings = settings or Settings()
    checks: list[Check] = []

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            result = subprocess.run(
                [nvidia, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            detail = result.stdout.strip() or "NVIDIA GPU detected"
            checks.append(Check("NVIDIA GPU", True, detail))
        except Exception as exc:  # pragma: no cover - depends on host hardware
            checks.append(Check("NVIDIA GPU", False, str(exc)))
    else:
        checks.append(Check("NVIDIA GPU", False, "nvidia-smi not found"))

    checks.append(_binary_check("FFmpeg", "ffmpeg"))
    checks.append(_path_check("Chatterbox runtime", settings.chatterbox_python))
    checks.append(_path_check("Kokoro runtime", settings.kokoro_python))
    checks.append(_path_check("EchoMimic runtime", settings.echo_python))
    checks.append(_path_check("EchoMimic code", settings.echo_root / "infer_flash.py"))
    checks.append(
        _path_check(
            "EchoMimic base model",
            settings.echo_model_root / "Wan2.1-Fun-V1.1-1.3B-InP",
        )
    )
    checks.append(
        _path_check(
            "EchoMimic audio model",
            settings.echo_model_root / "chinese-wav2vec2-base",
        )
    )
    checks.append(
        _path_check(
            "EchoMimic Flash weights",
            settings.echo_model_root / "transformer/diffusion_pytorch_model.safetensors",
        )
    )

    if settings.musetalk_enabled:
        checks.append(_path_check("MuseTalk runtime", settings.musetalk_python))
        checks.append(_path_check("MuseTalk code", settings.musetalk_root / "scripts/inference.py"))

    return checks


def _binary_check(name: str, binary: str) -> Check:
    path = shutil.which(binary)
    return Check(name, bool(path), path or f"{binary} not found")


def _path_check(name: str, path: Path) -> Check:
    return Check(name, path.exists(), str(path))
