"""Web API and a small chat UI.

    uvicorn rag.api:app --reload          then open http://127.0.0.1:8000

Endpoints:
    GET  /          the web page
    GET  /health    is the index loaded?
    GET  /sources   which documents are indexed
    POST /ask       {"question": "...", "mode": "hybrid", "top_k": 4, "rerank": false}

Environment: RAG_INDEX_DIR, RAG_LLM (anthropic | fake), ANTHROPIC_API_KEY. See rag/config.py.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from rag import config
from rag.generation import LLMConfigError, LLMError, make_generator
from rag.pipeline import RAGPipeline
from rag.rerank import RerankerUnavailable
from rag.store import IndexNotFoundError

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., max_length=2000, examples=["How long is the warranty on tents?"])
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    top_k: int = Field(4, ge=1, le=10)
    rerank: bool = False

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty or whitespace only")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once at startup, not on every request."""
    app.state.pipeline = None
    app.state.startup_error = None
    try:
        app.state.pipeline = RAGPipeline.load(config.INDEX_DIR, generator=make_generator(config.LLM_BACKEND))
    except IndexNotFoundError as exc:
        app.state.startup_error = str(exc)
    yield


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail=request.app.state.startup_error or "Index is not loaded")
    return pipeline


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ask the docs",
        description="Retrieval-augmented question answering with citations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request):
        pipeline = request.app.state.pipeline
        if pipeline is None:
            return {"status": "no_index", "detail": request.app.state.startup_error}
        return {
            "status": "ok",
            "documents": len(pipeline.index.meta["documents"]),
            "chunks": len(pipeline.index.chunks),
            "embedder": pipeline.index.meta["embedder"]["name"],
            "llm": pipeline.generator.model,
        }

    @app.get("/sources")
    def sources(request: Request):
        return {"documents": get_pipeline(request).index.document_summary()}

    @app.post("/ask")
    def ask(payload: AskRequest, request: Request):
        pipeline = get_pipeline(request)
        try:
            return pipeline.ask(payload.question, mode=payload.mode, top_k=payload.top_k, rerank=payload.rerank).to_dict()
        except LLMConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except RerankerUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


app = create_app()
