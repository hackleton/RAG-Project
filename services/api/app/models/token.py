"""Revoked access tokens.

`create_access_token` has issued a `jti` since Sprint 1 specifically so that
revocation would have something to key on. This is that key being used.

A revocation list rather than a session table: tokens stay stateless and the
common path - a token that was never revoked - costs one indexed miss. The
table only ever holds tokens that were explicitly given up, and only until they
would have expired anyway, which is what `expires_at` is for.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from app.models.base import Base, UUIDPrimaryKeyMixin


class RevokedToken(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "revoked_tokens"

    # The JWT's `jti`. Unique, so revoking twice is idempotent rather than an
    # integrity error - a client retrying a logout is normal.
    jti: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    # When the token would have expired on its own. Past this point the row is
    # dead weight and `purge_expired` can drop it.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
