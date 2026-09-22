"""Liveness and readiness probes.

Split deliberately:

* `/health` answers "is this process running?" - it touches no dependency, so a
  database blip never causes the orchestrator to restart a healthy container.
* `/health/ready` answers "can this process serve traffic?" - it checks the
  datastores and is what a load balancer or `depends_on: service_healthy`
  should gate on.

The readiness check opens its own connection rather than taking the
request-scoped session dependency: a probe must be able to *report* that the
database is unreachable, which it cannot do if acquiring the session is what
fails.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app import __version__
from app.api.deps import SettingsDep
from app.core.db import get_engine
from app.core.logging import get_logger
from app.schemas.common import DependencyHealth, HealthResponse, ReadinessResponse

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.app_env.value,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "One or more dependencies are unavailable."}},
)
async def readiness(
    request: Request, response: Response, settings: SettingsDep
) -> ReadinessResponse:
    dependencies = {
        "database": await _check_database(),
        "vector_store": await _check_vector_store(request),
    }

    overall = "ok" if all(d.status == "ok" for d in dependencies.values()) else "error"
    if overall != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall,
        service=settings.app_name,
        version=__version__,
        environment=settings.app_env.value,
        dependencies=dependencies,
    )


async def _check_vector_store(request: Request) -> DependencyHealth:
    start = time.perf_counter()
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        return DependencyHealth(status="error", detail="not initialised")
    if not await store.health():
        return DependencyHealth(status="error", detail="unreachable")
    return DependencyHealth(status="ok", latency_ms=round((time.perf_counter() - start) * 1000, 2))


async def _check_database() -> DependencyHealth:
    start = time.perf_counter()
    try:
        async with get_engine().connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, never raises
        logger.warning("readiness_check_failed", dependency="database", error=str(exc))
        return DependencyHealth(status="error", detail=type(exc).__name__)
    return DependencyHealth(status="ok", latency_ms=round((time.perf_counter() - start) * 1000, 2))
