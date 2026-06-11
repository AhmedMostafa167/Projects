"""FastAPI service. Provides programmatic access to the pipeline.

Run: `uvicorn api:app --reload`
Swagger UI: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.pipeline import ResearchPipeline
from src.utils import get_logger

log = get_logger(__name__)


class IngestRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["retrieval-augmented generation"])
    arxiv_n: int = Field(5, ge=0, le=20)
    web_n: int = Field(5, ge=0, le=20)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["What is RAG and why does it help LLMs?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


pipeline: ResearchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info("starting", provider=settings.llm_provider)
    pipeline = ResearchPipeline()
    yield
    log.info("shutting_down")


app = FastAPI(
    title="Research Assistant API",
    description="RAG over ArXiv + open web, powered by LangGraph.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": settings.llm_provider}


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, int]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    count = pipeline.ingest(req.query, arxiv_n=req.arxiv_n, web_n=req.web_n)
    return {"chunks_indexed": count}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    result = pipeline.ask(req.question)
    return AskResponse(answer=result.answer, sources=result.sources)
