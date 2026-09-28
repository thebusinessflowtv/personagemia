from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PERSONAGEMIA_", env_file=".env", extra="ignore")

    env: str = "development"
    output_dir: Path = Path("outputs")
    character_config: Path = Path("config/character.example.yaml")
    studio_config: Path = Path("config/studio.example.yaml")
    avatar_engine: str = "mock"
    avatar_command: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data
