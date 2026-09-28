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

    # Script generation. Anthropic stays optional and is the only paid API in the intended stack.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    # Local AI root. Heavy engines live outside this repository on the GPU runner.
    ai_root: Path = Path.home() / ".personagemia-ai"

    # Local TTS. Chatterbox is the expressive default; Kokoro is the lightweight fallback.
    tts_engine: str = "chatterbox"
    chatterbox_python: Path = Path.home() / ".personagemia-ai/venvs/chatterbox/bin/python"
    chatterbox_reference_audio: Path | None = None
    chatterbox_exaggeration: float = 0.85
    chatterbox_cfg_weight: float = 0.30
    kokoro_python: Path = Path.home() / ".personagemia-ai/venvs/kokoro/bin/python"
    kokoro_voice: str = "am_puck"
    kokoro_speed: float = 1.12

    # EchoMimicV3-Flash: main body/face animation engine.
    echo_root: Path = Path.home() / ".personagemia-ai/engines/echomimic_v3"
    echo_python: Path = Path.home() / ".personagemia-ai/venvs/echomimic/bin/python"
    echo_model_root: Path = Path.home() / ".personagemia-ai/models/echomimic_v3/flash"
    echo_output_dir: Path = Path("outputs/echomimic")
    echo_fps: int = 25
    echo_size: int = 768
    echo_steps: int = 8
    echo_guidance_scale: float = 4.5
    echo_audio_guidance_scale: float = 2.0
    echo_seed: int = 43

    # MuseTalk 1.5 is an optional lip-sync refinement stage after EchoMimic.
    musetalk_enabled: bool = False
    musetalk_root: Path = Path.home() / ".personagemia-ai/engines/MuseTalk"
    musetalk_python: Path = Path.home() / ".personagemia-ai/venvs/musetalk/bin/python"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data
