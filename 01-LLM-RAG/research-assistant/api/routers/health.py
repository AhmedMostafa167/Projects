from fastapi import APIRouter
from src.config import settings

health_router = APIRouter(

)
@health_router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": settings.llm_provider}