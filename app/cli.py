import json
from pathlib import Path

import typer

from app.acting import ActingDirector
from app.models import RenderRequest
from app.pipeline import AvatarPipeline

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


@app.command()
def render(
    script: str = typer.Option(..., "--script", help="English presenter script"),
    audio: Path = typer.Option(..., "--audio", exists=True, dir_okay=False),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    output_name: str = typer.Option("presenter.mp4", "--output-name"),
) -> None:
    """Run the avatar pipeline."""
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
