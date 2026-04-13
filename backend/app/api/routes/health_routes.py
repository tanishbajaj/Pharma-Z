"""Health check route."""

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health")
async def check_health():
    """Return application health status."""
    return {
        "status": "healthy",
        "service": "Pharma-Z",
        "message": "All systems operational",
    }
