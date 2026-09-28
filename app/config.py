from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PERSONAGEMIA_", env_file=".env", extra="ignore")

    env: str = "development"
    output_dir: Path = Path("outputs")
    character_config: Path = Path("config/mr_uncut.yaml")
    studio_config: Path = Path("config/studio.example.yaml")
    avatar_engine: str = "mock"
    avatar_command: str | None = None

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    sync_api_key: str | None = None
    sync_base_url: str = "https://api.sync.so"
    sync_model: str = "sync-3"
    sync_voice_id: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data
