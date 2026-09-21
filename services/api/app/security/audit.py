"""Audit trail writes.

The audit row is added to the caller's session, so it commits with the action
it describes: an action cannot succeed while its audit record is lost, and a
rolled-back action leaves no misleading trail.

Only identifiers, outcomes and counts go into `details` - never document text,
prompts or credentials (project plan section 15).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_sessionmaker
from app.core.logging import get_logger, request_id_ctx
from app.models.audit import AuditLog
from app.models.enums import AuditAction
from app.security.principal import Principal

logger = get_logger(__name__)


async def record_audit(
    session: AsyncSession,
    *,
    action: AuditAction,
    principal: Principal | None = None,
    actor_email: str | None = None,
    resource_type: str | None = None,
    resource_id: str | uuid.UUID | None = None,
    allowed: bool | None = None,
    ip_address: str | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    entry = AuditLog(
        action=action,
        actor_id=principal.id if principal else None,
        actor_email=principal.email if principal else actor_email,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        allowed=allowed,
        request_id=request_id_ctx.get(),
        ip_address=ip_address,
        details=details or {},
    )
    session.add(entry)
    logger.info(
        "audit",
        action=action.value,
        resource_type=resource_type,
        resource_id=entry.resource_id,
        allowed=allowed,
    )
    return entry


async def record_audit_independent(
    *,
    action: AuditAction,
    principal: Principal | None = None,
    actor_email: str | None = None,
    resource_type: str | None = None,
    resource_id: str | uuid.UUID | None = None,
    allowed: bool | None = None,
    ip_address: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Write an audit row in its own transaction.

    For events that accompany a *rejected* request. The request's transaction
    is rolled back when the handler raises, which would otherwise erase the
    record of the rejection - precisely the event worth keeping. A failure to
    write the audit row must not mask the original rejection, so it is logged
    rather than raised.
    """
    try:
        async with get_sessionmaker()() as session:
            await record_audit(
                session,
                action=action,
                principal=principal,
                actor_email=actor_email,
                resource_type=resource_type,
                resource_id=resource_id,
                allowed=allowed,
                ip_address=ip_address,
                details=details,
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - never mask the original failure
        logger.error("audit_write_failed", action=action.value, error=str(exc))
