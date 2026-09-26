"""Health endpoints.

/health        -> liveness  (process is up; never touches dependencies)
/health/ready  -> readiness (dependencies reachable); 503 if degraded
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.config import get_settings
from app.database.health import check_database, check_extensions
from app.schemas.health import LivenessResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    settings = get_settings()
    return LivenessResponse(app=settings.app_name, environment=settings.app_env)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    engine = request.app.state.engine
    db = await check_database(engine)
    components = [db]
    # Only probe extensions if the connection works.
    if db.ok:
        components.append(await check_extensions(engine))
    all_ok = all(c.ok for c in components)
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if all_ok else "degraded", components=components)