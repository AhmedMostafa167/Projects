from fastapi import APIRouter, HTTPException, Depends
from api.dependencies import get_pipeline
from api.schemas import AskRequest, AskResponse

ask_router = APIRouter()

@ask_router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline = Depends(get_pipeline)
    result = pipeline.ask(req.question)
    return AskResponse(
        answer=result.answer, 
        sources=result.sources
        )