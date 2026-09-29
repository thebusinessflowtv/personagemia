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

    # EchoMimicV3-Flash remains available as the ultra motion engine.
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
    echo_weight_dtype: str = "bfloat16"

    # Avatar Quality V2: short deterministic motion chunks + temporal lip refinement.
    quality_profile: str = "v2"
    avatar_chunk_seconds: float = 3.0
    avatar_chunk_overlap_seconds: float = 0.12
    quality_seed: int = 3407

    # LatentSync 1.5 is the preferred mouth/teeth refiner for a 16 GB-class GPU.
    latentsync_enabled: bool = True
    latentsync_root: Path = Path.home() / ".personagemia-ai/engines/LatentSync"
    latentsync_python: Path = Path.home() / ".personagemia-ai/venvs/latentsync/bin/python"
    latentsync_checkpoint: Path = Path.home() / ".personagemia-ai/models/latentsync/latentsync_unet.pt"
    latentsync_steps: int = 30
    latentsync_guidance_scale: float = 1.35

    # Fixed presentation layer. The studio is composited, never regenerated per frame.
    studio_background: Path = Path("assets/studio/mr_uncut_garage_bg_800x450.jpg")
    final_width: int = 1280
    final_height: int = 720
    subject_height_ratio: float = 0.93
    alpha_temporal_history: float = 0.22

    # MuseTalk remains an optional comparison/refinement engine.
    musetalk_enabled: bool = False
    musetalk_root: Path = Path.home() / ".personagemia-ai/engines/MuseTalk"
    musetalk_python: Path = Path.home() / ".personagemia-ai/venvs/musetalk/bin/python"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data
