from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Emotion(str, Enum):
    neutral = "neutral"
    confident = "confident"
    excited = "excited"
    skeptical = "skeptical"
    disbelief = "disbelief"
    serious = "serious"
    amused = "amused"
    frustrated = "frustrated"


class Gesture(str, Enum):
    none = "none"
    open_palms = "open_palms"
    explain = "explain"
    point = "point"
    count = "count"
    hands_together = "hands_together"
    lean_forward = "lean_forward"
    shrug = "shrug"


class PerformanceBeat(BaseModel):
    index: int
    text: str
    emotion: Emotion = Emotion.neutral
    gesture: Gesture = Gesture.none
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    eye_contact: float = Field(default=0.9, ge=0.0, le=1.0)
    head_motion: float = Field(default=0.35, ge=0.0, le=1.0)
    brow_motion: float = Field(default=0.35, ge=0.0, le=1.0)


class PerformancePlan(BaseModel):
    language: str = "en-US"
    style: str = "outspoken"
    beats: list[PerformanceBeat]


class RenderRequest(BaseModel):
    script: str = Field(min_length=1)
    audio_path: Path
    character_image_path: Path
    output_name: str = "presenter.mp4"


class RenderResult(BaseModel):
    job_id: str
    engine: str
    status: str
    output_path: Path | None = None
    performance_plan_path: Path
    message: str | None = None
