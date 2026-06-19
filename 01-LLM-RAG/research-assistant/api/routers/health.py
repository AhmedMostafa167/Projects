from fastapi import APIRouter
from src.config import settings

health_router = APIRouter(
    prefix='api/health',
    tags='api_health'
)
@health_router.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": settings.llm_provider}