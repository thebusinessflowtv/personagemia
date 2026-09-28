from pathlib import Path

from app.engines.base import AvatarEngine
from app.models import PerformancePlan


class MockAvatarEngine(AvatarEngine):
    name = "mock"

    def render(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        plan: PerformancePlan,
        plan_path: Path,
        output_path: Path,
    ) -> Path | None:
        # The mock validates orchestration without pretending a video was rendered.
        return None
