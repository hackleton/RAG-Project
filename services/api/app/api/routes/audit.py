"""The audit trail, read back.

Project plan section 11 asks for audit logs over document access, tool calls
and generated reports. Sprints 1 to 5 wrote them; nothing could read them, and
a trail nobody can query is a half-built control - it satisfies a checklist and
answers no question.

Two things are deliberate.

**Admin only, and no filter on another user's behalf.** Rows name who did what
to which patient's document, so the endpoint is as sensitive as the documents
it describes. `require_roles(Role.ADMIN)` is the gate; there is no "my audit
trail" variant, because a per-user view is a different product decision and
guessing at it would put patient identifiers in front of an analyst.

**Reading the trail is itself audited.** `/audit` writes an `AUDIT_ACCESSED`
row before returning. Without it the one action that reveals everything the
trail holds is the one action it does not record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import ColumnElement, case, false, func, select

from app.api.deps import DbSession, client_ip, require_roles
from app.models.audit import AuditLog
from app.models.enums import AuditAction, Role
from app.schemas.audit import (
    AuditActionCount,
    AuditEntryRead,
    AuditListResponse,
    AuditSummaryResponse,
)
from app.security.audit import record_audit
from app.security.principal import Principal

router = APIRouter(prefix="/audit", tags=["audit"])

AdminUser = Annotated[Principal, Depends(require_roles(Role.ADMIN))]


def _predicates(
    *,
    action: AuditAction | None,
    actor_email: str | None,
    resource_type: str | None,
    allowed: bool | None,
    since: datetime | None,
    request_id: str | None,
) -> list[ColumnElement[bool]]:
    """The filters, built once and applied to both the page and the count.

    Returned as conditions rather than as a patched statement so the two
    queries provably share them: a total computed over different predicates
    than the page it labels is worse than no total at all.

    `== allowed` rather than `.is_(allowed)`. SQLAlchemy renders `.is_()` as
    `IS`, and SQL Server accepts `IS` only against NULL - `allowed IS 1` is a
    syntax error there, and SQLite would have accepted it happily.
    """
    conditions: list[ColumnElement[bool]] = []
    if action is not None:
        conditions.append(AuditLog.action == action)
    if actor_email:
        conditions.append(AuditLog.actor_email == actor_email.strip().lower())
    if resource_type:
        conditions.append(AuditLog.resource_type == resource_type)
    if allowed is not None:
        conditions.append(AuditLog.allowed == allowed)
    if since is not None:
        conditions.append(AuditLog.created_at >= since)
    if request_id:
        conditions.append(AuditLog.request_id == request_id)
    return conditions


@router.get(
    "",
    response_model=AuditListResponse,
    summary="Read the audit trail (admin only)",
    responses={403: {"description": "The caller is not an administrator."}},
)
async def list_audit_entries(
    request: Request,
    session: DbSession,
    principal: AdminUser,
    action: Annotated[AuditAction | None, Query()] = None,
    actor_email: Annotated[str | None, Query(max_length=320)] = None,
    resource_type: Annotated[str | None, Query(max_length=64)] = None,
    allowed: Annotated[bool | None, Query()] = None,
    since: Annotated[datetime | None, Query()] = None,
    request_id: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditListResponse:
    conditions = _predicates(
        action=action,
        actor_email=actor_email,
        resource_type=resource_type,
        allowed=allowed,
        since=since,
        request_id=request_id,
    )
    applied: dict[str, str] = {
        name: str(value)
        for name, value in (
            ("action", action),
            ("actor_email", actor_email),
            ("resource_type", resource_type),
            ("allowed", allowed),
            ("since", since),
            ("request_id", request_id),
        )
        if value is not None
    }

    total = (
        await session.scalar(select(func.count()).select_from(AuditLog).where(*conditions))
    ) or 0
    rows = list(
        await session.scalars(
            select(AuditLog)
            .where(*conditions)
            .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )

    await record_audit(
        session,
        action=AuditAction.AUDIT_ACCESSED,
        principal=principal,
        resource_type="audit_log",
        allowed=True,
        ip_address=client_ip(request),
        # The filters, not the rows: what was looked for is the reviewable
        # fact, and copying returned rows into a new row would double the trail
        # on every read.
        details={"filters": applied, "returned": len(rows)},
    )
    return AuditListResponse(
        items=[AuditEntryRead.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/summary",
    response_model=AuditSummaryResponse,
    summary="Counts per action, with denials broken out (admin only)",
)
async def audit_summary(
    session: DbSession,
    principal: AdminUser,
    since: Annotated[datetime | None, Query()] = None,
) -> AuditSummaryResponse:
    statement = select(
        AuditLog.action,
        func.count().label("total"),
        # `allowed` is nullable - an event that is neither a grant nor a
        # refusal - so denials are counted explicitly as false rather than as
        # "not allowed". `case` rather than `IIF`, which SQLite does not have,
        # and `== false()` rather than `.is_(False)`, which SQL Server rejects.
        func.sum(case((AuditLog.allowed == false(), 1), else_=0)).label("denied"),
    ).group_by(AuditLog.action)
    if since is not None:
        statement = statement.where(AuditLog.created_at >= since)

    rows = (await session.execute(statement)).all()
    counts = [
        AuditActionCount(action=row.action, total=row.total, denied=int(row.denied or 0))
        for row in sorted(rows, key=lambda row: (-row.total, str(row.action)))
    ]
    return AuditSummaryResponse(
        since=since, counts=counts, total=sum(count.total for count in counts)
    )
