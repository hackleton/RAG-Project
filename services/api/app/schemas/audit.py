"""Audit trail payloads.

Read-only by design. There is no write schema and no update endpoint, because
an audit row that the application can edit is a log, not a trail.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.enums import AuditAction


class AuditEntryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    action: AuditAction
    actor_id: uuid.UUID | None
    actor_email: str | None
    resource_type: str | None
    resource_id: str | None
    allowed: bool | None
    # Correlates a row with the request that produced it, and with the
    # structured log line for the same request.
    request_id: str | None
    ip_address: str | None
    details: dict[str, Any]


class AuditListResponse(BaseModel):
    items: list[AuditEntryRead]
    total: int
    limit: int
    offset: int


class AuditActionCount(BaseModel):
    action: AuditAction
    total: int
    denied: int


class AuditSummaryResponse(BaseModel):
    """Counts per action over a window - what a reviewer reads first.

    `denied` is broken out because the interesting rows are the refusals: a
    trail where nothing was ever denied usually means the controls are not
    being exercised rather than that everything was permitted.
    """

    since: datetime | None
    counts: list[AuditActionCount]
    total: int
