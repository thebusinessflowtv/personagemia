from fastapi import FastAPI, HTTPException

from app import __version__
from app.models import RenderRequest, RenderResult
from app.pipeline import AvatarPipeline

app = FastAPI(title="PersonagemIA", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/render", response_model=RenderResult)
def render(request: RenderRequest) -> RenderResult:
    try:
        return AvatarPipeline().render(request)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
