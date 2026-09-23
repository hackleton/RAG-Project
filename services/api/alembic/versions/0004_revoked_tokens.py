"""Access tokens can be revoked before they expire.

Sprint 6. `create_access_token` has issued a `jti` since Sprint 1 so that
revocation would have a handle; `/auth/logout` is what finally uses it.

A deny-list rather than a session table. The common path is a token nobody
revoked, which costs one indexed miss, and the table only holds tokens that
were explicitly given up - until `expires_at`, after which they are dead weight
and `purge_expired` drops them.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "revoked_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("jti", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "revoked_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # Unique so revoking the same token twice is a no-op rather than an
    # integrity error - a client retrying a logout is ordinary.
    op.create_index("ix_revoked_tokens_jti", "revoked_tokens", ["jti"], unique=True)
    # The purge scans on this; without it every logout table-scans.
    op.create_index("ix_revoked_tokens_expires_at", "revoked_tokens", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_revoked_tokens_expires_at", table_name="revoked_tokens")
    op.drop_index("ix_revoked_tokens_jti", table_name="revoked_tokens")
    op.drop_table("revoked_tokens")
