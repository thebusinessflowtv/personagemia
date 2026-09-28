from abc import ABC, abstractmethod
from pathlib import Path

from app.models import PerformancePlan


class AvatarEngine(ABC):
    name: str

    @abstractmethod
    def render(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        plan: PerformancePlan,
        plan_path: Path,
        output_path: Path,
    ) -> Path | None:
        raise NotImplementedError
