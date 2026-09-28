import json
from pathlib import Path

import typer

from app.acting import ActingDirector
from app.models import RenderRequest
from app.pipeline import AvatarPipeline
from app.script_writer import ScriptWriter
from app.sync_labs import SyncLabsClient

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


@app.command("list-voices")
def list_voices() -> None:
    """List ElevenLabs voices exposed to this Sync Labs organization."""
    voices = SyncLabsClient().list_voices()
    typer.echo(json.dumps(voices, indent=2))


@app.command("test-shot")
def test_shot(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    script: str = typer.Option(..., "--script", help="English line for the test"),
    voice_id: str = typer.Option(..., "--voice-id"),
    output_name: str = typer.Option("mr_uncut_test", "--output-name"),
    stability: float = typer.Option(0.28, "--stability", min=0.0, max=1.0),
) -> None:
    """Generate a short Mr. Uncut image-to-talking-video test using Sync Labs sync-3."""
    sync = SyncLabsClient()
    audio = sync.synthesize(
        script,
        voice_id=voice_id,
        stability=stability,
        similarity_boost=0.80,
    )
    generation = sync.generate_from_image(
        image,
        audio_url=audio["url"],
        output_name=output_name,
    )
    result = sync.wait_for_generation(generation["id"])
    typer.echo(json.dumps(result, indent=2))


@app.command()
def render(
    script: str = typer.Option(..., "--script", help="English presenter script"),
    audio: Path = typer.Option(..., "--audio", exists=True, dir_okay=False),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    output_name: str = typer.Option("presenter.mp4", "--output-name"),
) -> None:
    """Run the local/open-source avatar pipeline."""
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
