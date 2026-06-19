from fastapi import APIRouter, HTTPException, Depends
from api.dependencies import get_pipeline
from api.schemas import AskRequest, AskResponse

ask_router = APIRouter()

@ask_router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, pipeline: ResearchPipeline = Depends(get_pipeline)) -> AskResponse:
    result = pipeline.ask(req.question)
    return AskResponse(
        answer=result.answer, 
        sources=result.sources
        )