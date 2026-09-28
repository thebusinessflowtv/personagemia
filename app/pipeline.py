import json
import uuid
from pathlib import Path

from app.acting import ActingDirector
from app.config import Settings
from app.engines import CommandAvatarEngine, MockAvatarEngine
from app.engines.base import AvatarEngine
from app.models import RenderRequest, RenderResult


class AvatarPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.director = ActingDirector()
        self.engine = self._build_engine()

    def _build_engine(self) -> AvatarEngine:
        if self.settings.avatar_engine == "mock":
            return MockAvatarEngine()
        if self.settings.avatar_engine == "command":
            return CommandAvatarEngine(self.settings.avatar_command or "")
        raise ValueError(f"Unsupported avatar engine: {self.settings.avatar_engine}")

    def render(self, request: RenderRequest) -> RenderResult:
        self._validate_input(request.audio_path, "audio")
        self._validate_input(request.character_image_path, "character image")

        job_id = uuid.uuid4().hex[:12]
        job_dir = self.settings.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        plan = self.director.plan(request.script)
        plan_path = job_dir / "performance_plan.json"
        plan_path.write_text(json.dumps(plan.model_dump(mode="json"), indent=2), encoding="utf-8")

        output_path = job_dir / request.output_name
        rendered = self.engine.render(
            image_path=request.character_image_path,
            audio_path=request.audio_path,
            plan=plan,
            plan_path=plan_path,
            output_path=output_path,
        )

        if rendered is None:
            return RenderResult(
                job_id=job_id,
                engine=self.engine.name,
                status="planned",
                performance_plan_path=plan_path,
                message="Pipeline validated. Configure a production avatar engine to render video.",
            )

        return RenderResult(
            job_id=job_id,
            engine=self.engine.name,
            status="rendered",
            output_path=rendered,
            performance_plan_path=plan_path,
        )

    @staticmethod
    def _validate_input(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
