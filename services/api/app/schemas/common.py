"""Response envelopes shared by every endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(examples=["not_found"])
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """The single error shape returned by the API."""

    error: ErrorDetail
    request_id: str | None = None


class DependencyHealth(BaseModel):
    status: Literal["ok", "degraded", "error"]
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    service: str
    version: str
    environment: str


class ReadinessResponse(HealthResponse):
    dependencies: dict[str, DependencyHealth]
