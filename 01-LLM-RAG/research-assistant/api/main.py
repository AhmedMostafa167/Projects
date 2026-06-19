"""FastAPI service. Provides programmatic access to the pipeline.

Run: `uvicorn api.main:app --reload`
Swagger UI: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from src.pipeline import ResearchPipeline

from src.config import settings
from src.utils import get_logger
from api.routers import health, ingest, ask

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info("starting", provider=settings.llm_provider)
    app.state.pipeline = ResearchPipeline()
    yield
    log.info("shutting_down")


app = FastAPI(
    title="Research Assistant API",
    description="RAG over ArXiv + open web, powered by LangGraph.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.health_router)

app.include_router(ingest.ingest_router)

app.include_router(ask.ask_router)

