import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


@dataclass(frozen=True)
class VoicePreset:
    name: str
    engine: str
    exaggeration: float = 0.85
    cfg_weight: float = 0.30
    kokoro_voice: str = "am_puck"
    speed: float = 1.12


VOICE_PRESETS: dict[str, VoicePreset] = {
    # Chatterbox style presets. Once Mr. Uncut has an approved reference voice,
    # all three preserve the same identity while changing performance intensity.
    "electric": VoicePreset("electric", "chatterbox", exaggeration=0.90, cfg_weight=0.28),
    "intense": VoicePreset("intense", "chatterbox", exaggeration=1.05, cfg_weight=0.25),
    "controlled": VoicePreset("controlled", "chatterbox", exaggeration=0.72, cfg_weight=0.35),
    # Built-in Kokoro candidates are useful for zero-cost voice auditions.
    "puck": VoicePreset("puck", "kokoro", kokoro_voice="am_puck", speed=1.13),
    "adam": VoicePreset("adam", "kokoro", kokoro_voice="am_adam", speed=1.09),
    "liam": VoicePreset("liam", "kokoro", kokoro_voice="am_liam", speed=1.12),
}


class LocalTTS:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        preset_name: str = "electric",
        engine: str | None = None,
    ) -> Path:
        if not text.strip():
            raise ValueError("TTS text cannot be empty")
        if preset_name not in VOICE_PRESETS:
            raise ValueError(f"Unknown voice preset: {preset_name}")

        preset = VOICE_PRESETS[preset_name]
        selected_engine = engine or preset.engine or self.settings.tts_engine
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if selected_engine == "chatterbox":
            return self._chatterbox(text, output_path, preset)
        if selected_engine == "kokoro":
            return self._kokoro(text, output_path, preset)
        raise ValueError(f"Unsupported local TTS engine: {selected_engine}")

    def _chatterbox(self, text: str, output_path: Path, preset: VoicePreset) -> Path:
        self._require_file(self.settings.chatterbox_python, "Chatterbox Python")
        helper = Path("scripts/chatterbox_tts.py")
        self._require_file(helper, "Chatterbox helper")

        command = [
            str(self.settings.chatterbox_python),
            str(helper),
            "--text",
            text,
            "--output",
            str(output_path),
            "--exaggeration",
            str(preset.exaggeration),
            "--cfg-weight",
            str(preset.cfg_weight),
        ]
        reference = self.settings.chatterbox_reference_audio
        if reference:
            self._require_file(reference, "Mr. Uncut voice reference")
            command.extend(["--reference", str(reference)])

        subprocess.run(command, check=True)
        self._require_file(output_path, "generated Chatterbox audio")
        return output_path

    def _kokoro(self, text: str, output_path: Path, preset: VoicePreset) -> Path:
        self._require_file(self.settings.kokoro_python, "Kokoro Python")
        helper = Path("scripts/kokoro_tts.py")
        self._require_file(helper, "Kokoro helper")
        voice = preset.kokoro_voice or self.settings.kokoro_voice
        speed = preset.speed or self.settings.kokoro_speed

        subprocess.run(
            [
                str(self.settings.kokoro_python),
                str(helper),
                "--text",
                text,
                "--output",
                str(output_path),
                "--voice",
                voice,
                "--speed",
                str(speed),
            ],
            check=True,
        )
        self._require_file(output_path, "generated Kokoro audio")
        return output_path

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
