from fastapi import APIRouter, HTTPException, Depends
from api.dependencies import get_pipeline
from api.schemas import IngestRequest

ingest_router = APIRouter()

@ingest_router.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, int]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline = Depends(get_pipeline)
    count = pipeline.ingest(req.query, arxiv_n=req.arxiv_n, web_n=req.web_n)
    return {"chunks_indexed": count}
