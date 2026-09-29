import json
import uuid
from pathlib import Path

import typer

from app.acting import ActingDirector
from app.config import Settings
from app.local_avatar import AvatarQualityV2Engine, EchoMimicFlashEngine, LatentSyncRefiner
from app.local_doctor import run_doctor
from app.local_tts import VOICE_PRESETS, LocalTTS
from app.models import RenderRequest
from app.pipeline import AvatarPipeline
from app.script_writer import ScriptWriter

app = typer.Typer(help="PersonagemIA virtual creator toolkit")


@app.command()
def plan(
    script: str = typer.Option(..., "--script", help="English script to direct"),
    output: Path = typer.Option(Path("performance_plan.json"), "--output"),
) -> None:
    """Create an acting/performance plan from an English script."""
    result = ActingDirector().plan(script)
    output.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    typer.echo(str(output))


@app.command("write-script")
def write_script(
    opinion: str = typer.Option(..., "--opinion", help="User opinion/transcript, any language"),
    topic: str | None = typer.Option(None, "--topic"),
    context: str | None = typer.Option(None, "--context"),
    seconds: int | None = typer.Option(None, "--seconds", min=3, max=1800),
) -> None:
    """Turn the user's opinion into a strong first-person Mr. Uncut script."""
    script = ScriptWriter().rewrite_opinion(
        opinion,
        topic=topic,
        context=context,
        target_seconds=seconds,
    )
    typer.echo(script)


@app.command("local-doctor")
def local_doctor() -> None:
    """Validate the local zero-cost TTS/GPU/avatar runtime."""
    checks = run_doctor()
    failures = 0
    for check in checks:
        icon = "OK" if check.ok else "MISSING"
        typer.echo(f"[{icon}] {check.name}: {check.detail}")
        failures += int(not check.ok)
    if failures:
        raise typer.Exit(code=2)


@app.command("voice-samples")
def voice_samples(
    text: str = typer.Option(
        "Everybody keeps pretending this is normal. I don't buy it. I'm Mr. Uncut, and I'm going to say exactly what I think.",
        "--text",
    ),
    presets: str = typer.Option("puck,adam,liam", "--presets"),
    output_dir: Path = typer.Option(Path("outputs/voice_samples"), "--output-dir"),
) -> None:
    """Generate several local voice candidates without any paid API."""
    tts = LocalTTS()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in presets.split(",") if item.strip()]
    for preset in requested:
        if preset not in VOICE_PRESETS:
            raise typer.BadParameter(f"Unknown preset {preset}. Available: {', '.join(VOICE_PRESETS)}")
        path = output_dir / f"mr_uncut_{preset}.wav"
        tts.synthesize(text, path, preset_name=preset)
        typer.echo(f"{preset}: {path}")


@app.command("local-test")
def local_test(
    script: str = typer.Option(
        "Everybody keeps pretending this is normal. I don't buy it. I'm Mr. Uncut, and I'm going to say exactly what I think.",
        "--script",
    ),
    image: Path = typer.Option(Path("assets/mr_uncut_master.png"), "--image"),
    voice: str = typer.Option("electric", "--voice"),
    tts_engine: str | None = typer.Option(None, "--tts-engine"),
    quality_v2: bool = typer.Option(True, "--quality-v2/--legacy-motion"),
    refine_lips: bool = typer.Option(True, "--refine-lips/--no-refine-lips"),
) -> None:
    """Generate Mr. Uncut with short-chunk motion and temporal lip refinement by default."""
    if not image.exists():
        raise FileNotFoundError(image)

    settings = Settings()
    job_id = uuid.uuid4().hex[:10]
    job_dir = settings.output_dir / "local-tests" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = job_dir / "mr_uncut.wav"
    body_video = job_dir / "mr_uncut_echomimic.mp4"
    final_video = job_dir / "mr_uncut_final.mp4"

    typer.echo(f"Job: {job_id}")
    typer.echo("1/3 Generating local voice...")
    LocalTTS(settings).synthesize(
        script,
        audio_path,
        preset_name=voice,
        engine=tts_engine,
    )

    if quality_v2:
        typer.echo("2/3 Running Avatar Quality V2 in short deterministic chunks...")
        typer.echo("3/3 Refining mouth/teeth temporal consistency with LatentSync 1.5...")
        AvatarQualityV2Engine(settings).render(image, audio_path, final_video)
        body_video = final_video
        lip_refiner = "latentsync-1.5"
    else:
        typer.echo("2/3 Animating Mr. Uncut with EchoMimicV3-Flash...")
        EchoMimicFlashEngine(settings).render(image, audio_path, body_video)
        if refine_lips:
            typer.echo("3/3 Refining lip sync with LatentSync 1.5...")
            LatentSyncRefiner(settings).refine(body_video, audio_path, final_video)
            lip_refiner = "latentsync-1.5"
        else:
            typer.echo("3/3 Lip refinement disabled; using EchoMimic output.")
            final_video.write_bytes(body_video.read_bytes())
            lip_refiner = None

    result = {
        "job_id": job_id,
        "audio": str(audio_path),
        "body_video": str(body_video),
        "final_video": str(final_video),
        "tts_engine": tts_engine or VOICE_PRESETS[voice].engine,
        "voice_preset": voice,
        "quality_profile": "v2" if quality_v2 else "legacy",
        "chunk_seconds": settings.avatar_chunk_seconds if quality_v2 else None,
        "lip_refiner": lip_refiner,
    }
    typer.echo(json.dumps(result, indent=2))


@app.command()
def render(
    script: str = typer.Option(..., "--script", help="English presenter script"),
    audio: Path = typer.Option(..., "--audio", exists=True, dir_okay=False),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    output_name: str = typer.Option("presenter.mp4", "--output-name"),
) -> None:
    """Run the generic pluggable avatar pipeline."""
    result = AvatarPipeline().render(
        RenderRequest(
            script=script,
            audio_path=audio,
            character_image_path=image,
            output_name=output_name,
        )
    )
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
