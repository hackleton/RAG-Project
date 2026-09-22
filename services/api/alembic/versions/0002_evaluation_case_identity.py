"""Evaluation cases gain a stable key, a persona and a patient scope.

Sprint 5. The dataset file is the source of truth and this table materialises
it, so a case needs an identity that survives a re-sync: `case_key` is that
identity, unique within a dataset version, and it is what per-case results are
correlated on across runs.

`persona` and `patient_external_id` move two things out of the runner and into
the case. An access-control case is only meaningful when the caller's scope
excludes the material, so who asks is part of what is being tested.

Additive and reversible. The columns are created nullable and then tightened,
which costs one extra statement and means the migration behaves the same
whether or not the table already holds rows.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("evaluation_cases", sa.Column("case_key", sa.String(length=64), nullable=True))
    op.add_column("evaluation_cases", sa.Column("persona", sa.String(length=16), nullable=True))
    op.add_column(
        "evaluation_cases", sa.Column("patient_external_id", sa.String(length=64), nullable=True)
    )

    # Any case that predates this migration was loaded without a key. Falling
    # back to the primary key keeps the uniqueness constraint satisfiable
    # without inventing a name that looks meaningful.
    op.execute(
        "UPDATE evaluation_cases SET case_key = CONVERT(nvarchar(64), id) WHERE case_key IS NULL"
    )
    op.execute("UPDATE evaluation_cases SET persona = 'clinician' WHERE persona IS NULL")

    op.alter_column(
        "evaluation_cases", "case_key", existing_type=sa.String(length=64), nullable=False
    )
    op.alter_column(
        "evaluation_cases", "persona", existing_type=sa.String(length=16), nullable=False
    )
    op.create_unique_constraint(
        "uq_evaluation_cases_version_key", "evaluation_cases", ["dataset_version", "case_key"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_evaluation_cases_version_key", "evaluation_cases", type_="unique")
    op.drop_column("evaluation_cases", "patient_external_id")
    op.drop_column("evaluation_cases", "persona")
    op.drop_column("evaluation_cases", "case_key")
