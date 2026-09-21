"""Append-only audit trail.

Project plan section 11 requires audit logs for document access, tool calls
and generated reports. Rows are written on the same transaction as the action
they describe, so an action cannot succeed without its audit record.

`details` holds identifiers, outcomes and counts - never document text,
prompts or credentials (section 15).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from app.models.base import Base, JSONType, UUIDPrimaryKeyMixin
from app.models.enums import AuditAction


class AuditLog(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_action_created", "action", "created_at"),
        Index("ix_audit_logs_actor_created", "actor_id", "created_at"),
    )

    # No updated_at: audit rows are never modified.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    action: Mapped[AuditAction] = mapped_column(
        Enum(AuditAction, native_enum=False, length=48, validate_strings=True), nullable=False
    )
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # Kept alongside actor_id so the trail survives user deletion.
    actor_email: Mapped[str | None] = mapped_column(String(320))
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(128))
    allowed: Mapped[bool | None] = mapped_column()
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
