import shlex
import subprocess
from pathlib import Path

from app.engines.base import AvatarEngine
from app.models import PerformancePlan


class CommandAvatarEngine(AvatarEngine):
    """Adapter for any local/open-source avatar renderer installed on the GPU machine."""

    name = "command"

    def __init__(self, command: str):
        if not command:
            raise ValueError("PERSONAGEMIA_AVATAR_COMMAND is required for command engine")
        self.command = command

    def render(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        plan: PerformancePlan,
        plan_path: Path,
        output_path: Path,
    ) -> Path:
        args = shlex.split(self.command) + [
            "--image", str(image_path),
            "--audio", str(audio_path),
            "--plan", str(plan_path),
            "--output", str(output_path),
        ]
        subprocess.run(args, check=True)
        if not output_path.exists():
            raise RuntimeError(f"Avatar command finished but did not create {output_path}")
        return output_path
