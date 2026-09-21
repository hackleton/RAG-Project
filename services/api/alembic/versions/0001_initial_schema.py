"""Initial schema: users, patients, documents, chunks, labs, chat, evaluation, audit.

Every constraint is named explicitly. The op.* helpers do not inherit the
metadata naming convention, so relying on it here would produce constraints
that a downgrade cannot drop by name on Postgres.

Revision ID: 0001
Revises:
Create Date: 2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB(astext_type=sa.Text())
NOW = sa.text("now()")


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("hashed_password", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "patients",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("birth_year", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(length=16), nullable=True),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_patients"),
        # Project plan section 2: synthetic data only, enforced by the database
        # rather than by convention.
        sa.CheckConstraint("is_synthetic = true", name="ck_patients_must_be_synthetic"),
    )
    op.create_index("ix_patients_external_id", "patients", ["external_id"], unique=True)

    op.create_table(
        "patient_access_grants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_patient_access_grants"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_patient_access_grants_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_patient_access_grants_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "user_id", "patient_id", name="uq_patient_access_grants_user_id_patient_id"
        ),
    )
    op.create_index("ix_patient_access_grants_user_id", "patient_access_grants", ["user_id"])
    op.create_index("ix_patient_access_grants_patient_id", "patient_access_grants", ["patient_id"])

    op.create_table(
        "lab_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("test_name", sa.String(length=128), nullable=False),
        sa.Column("test_code", sa.String(length=32), nullable=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("reference_low", sa.Float(), nullable=True),
        sa.Column("reference_high", sa.Float(), nullable=True),
        sa.Column("abnormal_flag", sa.String(length=8), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_lab_results"),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_lab_results_patient_id_patients",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_lab_results_patient_id", "lab_results", ["patient_id"])
    op.create_index("ix_lab_results_test_name", "lab_results", ["test_name"])
    # The SQL tool's dominant access pattern: one patient, one test, over time.
    op.create_index(
        "ix_lab_results_patient_test_time",
        "lab_results",
        ["patient_id", "test_name", "collected_at"],
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("document_type", sa.String(length=64), nullable=False),
        sa.Column("blob_uri", sa.String(length=1024), nullable=True),
        sa.Column("access_level", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("status_detail", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("uploaded_by_id", sa.Uuid(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_documents_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by_id"],
            ["users.id"],
            name="fk_documents_uploaded_by_id_users",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_documents_patient_id", "documents", ["patient_id"])
    op.create_index("ix_documents_checksum_sha256", "documents", ["checksum_sha256"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_patient_status", "documents", ["patient_id", "status"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column("access_level", sa.String(length=32), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(length=512), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding_reference", sa.String(length=256), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("chunk_metadata", JSONB, nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_id_chunk_index"
        ),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_patient_id", "document_chunks", ["patient_id"])
    op.create_index(
        "ix_document_chunks_embedding_reference", "document_chunks", ["embedding_reference"]
    )
    # Supports the pre-retrieval permission filter (project plan section 11).
    op.create_index(
        "ix_document_chunks_patient_access", "document_chunks", ["patient_id", "access_level"]
    )

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=True),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_sessions"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_chat_sessions_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_chat_sessions_patient_id_patients",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_user_created", "chat_sessions", ["user_id", "created_at"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", JSONB, nullable=False),
        sa.Column("retrieval_trace", JSONB, nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_chat_messages"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["chat_sessions.id"],
            name="fk_chat_messages_session_id_chat_sessions",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_index(
        "ix_chat_messages_session_created", "chat_messages", ["session_id", "created_at"]
    )

    op.create_table(
        "evaluation_cases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version", sa.String(length=32), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("expected_facts", JSONB, nullable=False),
        sa.Column("expected_sources", JSONB, nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("difficulty", sa.String(length=16), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_cases"),
    )
    op.create_index("ix_evaluation_cases_dataset_version", "evaluation_cases", ["dataset_version"])
    op.create_index("ix_evaluation_cases_category", "evaluation_cases", ["category"])
    op.create_index(
        "ix_evaluation_cases_dataset_category",
        "evaluation_cases",
        ["dataset_version", "category"],
    )

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version", sa.String(length=32), nullable=False),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics", JSONB, nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_by_id", sa.Uuid(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_runs"),
        sa.ForeignKeyConstraint(
            ["triggered_by_id"],
            ["users.id"],
            name="fk_evaluation_runs_triggered_by_id_users",
            ondelete="SET NULL",
        ),
    )

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("retrieved_sources", JSONB, nullable=False),
        sa.Column("scores", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_results"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_runs.id"],
            name="fk_evaluation_results_run_id_evaluation_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["evaluation_cases.id"],
            name="fk_evaluation_results_case_id_evaluation_cases",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_evaluation_results_run_id", "evaluation_results", ["run_id"])
    op.create_index("ix_evaluation_results_run_case", "evaluation_results", ["run_id", "case_id"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        # No updated_at: audit rows are append-only.
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("action", sa.String(length=48), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=True),
        sa.Column("actor_email", sa.String(length=320), nullable=True),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=128), nullable=True),
        sa.Column("allowed", sa.Boolean(), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("details", JSONB, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
        sa.ForeignKeyConstraint(
            ["actor_id"], ["users.id"], name="fk_audit_logs_actor_id_users", ondelete="SET NULL"
        ),
    )
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])
    op.create_index("ix_audit_logs_action_created", "audit_logs", ["action", "created_at"])
    op.create_index("ix_audit_logs_actor_created", "audit_logs", ["actor_id", "created_at"])


def downgrade() -> None:
    # Reverse dependency order; indexes and constraints drop with their table.
    for table in (
        "audit_logs",
        "evaluation_results",
        "evaluation_runs",
        "evaluation_cases",
        "chat_messages",
        "chat_sessions",
        "document_chunks",
        "documents",
        "lab_results",
        "patient_access_grants",
        "patients",
        "users",
    ):
        op.drop_table(table)
