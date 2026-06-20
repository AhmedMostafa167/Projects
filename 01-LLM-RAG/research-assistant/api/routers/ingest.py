from fastapi import APIRouter, Depends

from api.dependencies import get_pipeline
from api.schemas import IngestRequest
from src.pipeline import ResearchPipeline

ingest_router = APIRouter()


@ingest_router.post("/ingest")
async def ingest(
    req: IngestRequest,
    pipeline: ResearchPipeline = Depends(get_pipeline),
) -> dict[str, int]:
    count = pipeline.ingest(req.query, arxiv_n=req.arxiv_n, web_n=req.web_n)
    return {"chunks_indexed": count}