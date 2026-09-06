"""
Health-check route.
GET /api/health — confirms the backend is running.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Simple liveness probe — useful for Phase 1 verification and later CI checks."""
    return HealthResponse(
        status="ok",
        message="LearnSimplify AI backend is running.",
        version="1.0.0",
    )
