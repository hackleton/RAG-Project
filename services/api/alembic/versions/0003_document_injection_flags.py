"""Documents carry a count of chunks with instruction-like text.

Sprint 6. The injection filter detects planted instructions at ingestion and
neutralises them at retrieval; this column is the part an operator can query.
"Which documents in this corpus carry planted instructions?" is the first
question asked after an incident, and answering it by scanning a JSON field
across every chunk is the wrong shape for it.

Existing rows are backfilled to 0 rather than rescanned. Rescanning would mean
running the Sprint 6 pattern set inside a migration, which pins the schema to a
version of the detector that will change; the honest alternative is
`scripts/rescan_documents.py`, which re-runs the current detector whenever the
patterns move and is a normal operation rather than a schema change.

Additive and reversible: the column is created nullable, defaulted, then
tightened, so it behaves the same on an empty database and a populated one.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("injection_flag_count", sa.Integer(), nullable=True))
    op.execute("UPDATE documents SET injection_flag_count = 0 WHERE injection_flag_count IS NULL")
    op.alter_column(
        "documents",
        "injection_flag_count",
        existing_type=sa.Integer(),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("documents", "injection_flag_count")
