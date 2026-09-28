from app.local_avatar import compatible_video_length
from app.local_tts import VOICE_PRESETS


def test_video_length_is_wan_compatible_and_covers_audio() -> None:
    frames = compatible_video_length(10.0, 25)
    assert frames >= 250
    assert (frames - 1) % 4 == 0


def test_short_video_uses_safe_minimum() -> None:
    assert compatible_video_length(1.0, 25) == 81


def test_mr_uncut_has_local_voice_options() -> None:
    assert VOICE_PRESETS["electric"].engine == "chatterbox"
    assert VOICE_PRESETS["puck"].engine == "kokoro"
    assert {"puck", "adam", "liam"}.issubset(VOICE_PRESETS)
