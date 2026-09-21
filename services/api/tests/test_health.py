"""Health and readiness probes."""

from __future__ import annotations

from httpx import AsyncClient

from app import __version__
from app.core import db as db_module


async def test_liveness_reports_version_and_environment(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["environment"] == "ci"


async def test_liveness_does_not_touch_the_database(client: AsyncClient) -> None:
    """A database outage must not make the orchestrator kill a healthy process."""
    await db_module.dispose_engine()

    response = await client.get("/health")

    assert response.status_code == 200


async def test_readiness_checks_the_database(client: AsyncClient) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 200
    database = response.json()["dependencies"]["database"]
    assert database["status"] == "ok"
    assert database["latency_ms"] is not None


async def test_readiness_fails_when_the_database_is_gone(client: AsyncClient) -> None:
    await db_module.dispose_engine()

    response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["dependencies"]["database"]["status"] == "error"


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.headers["x-request-id"]


async def test_inbound_request_id_is_preserved(client: AsyncClient) -> None:
    """A trace started upstream must continue, not restart, at the API."""
    response = await client.get("/health", headers={"X-Request-ID": "upstream-trace-1"})

    assert response.headers["x-request-id"] == "upstream-trace-1"
