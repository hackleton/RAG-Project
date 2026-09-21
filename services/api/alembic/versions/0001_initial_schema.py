"""Initial schema: users, patients, documents, chunks, labs, chat, evaluation, audit.

Generated against SQL Server 2022 from app.models, so the column types are the
ones SQL Server actually uses: NVARCHAR for human text, NVARCHAR(max) for JSON
(2022 has no native JSON type), UNIQUEIDENTIFIER for the UUID keys.

`alembic check` in CI fails if this drifts from the ORM metadata.

Revision ID: 0001
Revises:
Create Date: 2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evaluation_cases",
        sa.Column("dataset_version", sa.String(length=32), nullable=False),
        sa.Column("question", sa.UnicodeText(), nullable=False),
        sa.Column("expected_answer", sa.UnicodeText(), nullable=True),
        sa.Column("expected_facts", sa.JSON(), nullable=False),
        sa.Column("expected_sources", sa.JSON(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("difficulty", sa.String(length=16), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_cases")),
    )
    op.create_index(
        op.f("ix_evaluation_cases_category"), "evaluation_cases", ["category"], unique=False
    )
    op.create_index(
        "ix_evaluation_cases_dataset_category",
        "evaluation_cases",
        ["dataset_version", "category"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_cases_dataset_version"),
        "evaluation_cases",
        ["dataset_version"],
        unique=False,
    )
    op.create_table(
        "patients",
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.Unicode(length=200), nullable=False),
        sa.Column("birth_year", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(length=16), nullable=True),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("is_synthetic = 1", name=op.f("ck_patients_must_be_synthetic")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patients")),
    )
    op.create_index(op.f("ix_patients_external_id"), "patients", ["external_id"], unique=True)
    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("name", sa.Unicode(length=200), nullable=False),
        sa.Column(
            "role",
            sa.Enum("ANALYST", "CLINICIAN", "ADMIN", name="role", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("hashed_password", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_table(
        "audit_logs",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "action",
            sa.Enum(
                "LOGIN_SUCCEEDED",
                "LOGIN_FAILED",
                "DOCUMENT_UPLOADED",
                "DOCUMENT_ACCESSED",
                "TOOL_CALLED",
                "TOOL_DENIED",
                "SQL_EXECUTED",
                "SQL_REJECTED",
                "REPORT_GENERATED",
                "INJECTION_BLOCKED",
                name="auditaction",
                native_enum=False,
                length=48,
            ),
            nullable=False,
        ),
        sa.Column("actor_id", sa.Uuid(), nullable=True),
        sa.Column("actor_email", sa.String(length=320), nullable=True),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=128), nullable=True),
        sa.Column("allowed", sa.Boolean(), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_id"],
            ["users.id"],
            name=op.f("fk_audit_logs_actor_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
    )
    op.create_index(
        "ix_audit_logs_action_created", "audit_logs", ["action", "created_at"], unique=False
    )
    op.create_index(
        "ix_audit_logs_actor_created", "audit_logs", ["actor_id", "created_at"], unique=False
    )
    op.create_index(op.f("ix_audit_logs_request_id"), "audit_logs", ["request_id"], unique=False)
    op.create_table(
        "chat_sessions",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Unicode(length=300), nullable=True),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_chat_sessions_patient_id_patients"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_chat_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_sessions")),
    )
    op.create_index(
        "ix_chat_sessions_user_created", "chat_sessions", ["user_id", "created_at"], unique=False
    )
    op.create_index(op.f("ix_chat_sessions_user_id"), "chat_sessions", ["user_id"], unique=False)
    op.create_table(
        "documents",
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column("filename", sa.Unicode(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("document_type", sa.Unicode(length=64), nullable=False),
        sa.Column("blob_uri", sa.String(length=1024), nullable=True),
        sa.Column(
            "access_level",
            sa.Enum(
                "PUBLIC", "INTERNAL", "RESTRICTED", name="accesslevel", native_enum=False, length=32
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "UPLOADED",
                "EXTRACTING",
                "CHUNKING",
                "INDEXING",
                "READY",
                "FAILED",
                name="documentstatus",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("status_detail", sa.UnicodeText(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("uploaded_by_id", sa.Uuid(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_documents_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by_id"],
            ["users.id"],
            name=op.f("fk_documents_uploaded_by_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
    )
    op.create_index(
        op.f("ix_documents_checksum_sha256"), "documents", ["checksum_sha256"], unique=False
    )
    op.create_index(op.f("ix_documents_patient_id"), "documents", ["patient_id"], unique=False)
    op.create_index(
        "ix_documents_patient_status", "documents", ["patient_id", "status"], unique=False
    )
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)
    op.create_table(
        "evaluation_runs",
        sa.Column("dataset_version", sa.String(length=32), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["triggered_by_id"],
            ["users.id"],
            name=op.f("fk_evaluation_runs_triggered_by_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_runs")),
    )
    op.create_table(
        "lab_results",
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("test_name", sa.Unicode(length=128), nullable=False),
        sa.Column("test_code", sa.String(length=32), nullable=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.Unicode(length=32), nullable=False),
        sa.Column("reference_low", sa.Float(), nullable=True),
        sa.Column("reference_high", sa.Float(), nullable=True),
        sa.Column("abnormal_flag", sa.String(length=8), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_lab_results_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lab_results")),
    )
    op.create_index(op.f("ix_lab_results_patient_id"), "lab_results", ["patient_id"], unique=False)
    op.create_index(
        "ix_lab_results_patient_test_time",
        "lab_results",
        ["patient_id", "test_name", "collected_at"],
        unique=False,
    )
    op.create_index(op.f("ix_lab_results_test_name"), "lab_results", ["test_name"], unique=False)
    op.create_table(
        "patient_access_grants",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_patient_access_grants_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_patient_access_grants_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patient_access_grants")),
        sa.UniqueConstraint(
            "user_id", "patient_id", name=op.f("uq_patient_access_grants_user_id_patient_id")
        ),
    )
    op.create_index(
        op.f("ix_patient_access_grants_patient_id"),
        "patient_access_grants",
        ["patient_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_patient_access_grants_user_id"), "patient_access_grants", ["user_id"], unique=False
    )
    op.create_table(
        "chat_messages",
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("USER", "ASSISTANT", name="chatrole", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("content", sa.UnicodeText(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("retrieval_trace", sa.JSON(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["chat_sessions.id"],
            name=op.f("fk_chat_messages_session_id_chat_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_messages")),
    )
    op.create_index(
        "ix_chat_messages_session_created",
        "chat_messages",
        ["session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chat_messages_session_id"), "chat_messages", ["session_id"], unique=False
    )
    op.create_table(
        "document_chunks",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=True),
        sa.Column(
            "access_level",
            sa.Enum(
                "PUBLIC", "INTERNAL", "RESTRICTED", name="accesslevel", native_enum=False, length=32
            ),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.Unicode(length=512), nullable=True),
        sa.Column("text", sa.UnicodeText(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding_reference", sa.String(length=256), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("chunk_metadata", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name=op.f("uq_document_chunks_document_id_chunk_index")
        ),
    )
    op.create_index(
        op.f("ix_document_chunks_document_id"), "document_chunks", ["document_id"], unique=False
    )
    op.create_index(
        op.f("ix_document_chunks_embedding_reference"),
        "document_chunks",
        ["embedding_reference"],
        unique=False,
    )
    op.create_index(
        "ix_document_chunks_patient_access",
        "document_chunks",
        ["patient_id", "access_level"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_patient_id"), "document_chunks", ["patient_id"], unique=False
    )
    op.create_table(
        "evaluation_results",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("answer", sa.UnicodeText(), nullable=True),
        sa.Column("retrieved_sources", sa.JSON(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("error", sa.UnicodeText(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["evaluation_cases.id"],
            name=op.f("fk_evaluation_results_case_id_evaluation_cases"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_runs.id"],
            name=op.f("fk_evaluation_results_run_id_evaluation_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_results")),
    )
    op.create_index(
        "ix_evaluation_results_run_case", "evaluation_results", ["run_id", "case_id"], unique=False
    )
    op.create_index(
        op.f("ix_evaluation_results_run_id"), "evaluation_results", ["run_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_evaluation_results_run_id"), table_name="evaluation_results")
    op.drop_index("ix_evaluation_results_run_case", table_name="evaluation_results")
    op.drop_table("evaluation_results")
    op.drop_index(op.f("ix_document_chunks_patient_id"), table_name="document_chunks")
    op.drop_index("ix_document_chunks_patient_access", table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_embedding_reference"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index(op.f("ix_chat_messages_session_id"), table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index(op.f("ix_patient_access_grants_user_id"), table_name="patient_access_grants")
    op.drop_index(op.f("ix_patient_access_grants_patient_id"), table_name="patient_access_grants")
    op.drop_table("patient_access_grants")
    op.drop_index(op.f("ix_lab_results_test_name"), table_name="lab_results")
    op.drop_index("ix_lab_results_patient_test_time", table_name="lab_results")
    op.drop_index(op.f("ix_lab_results_patient_id"), table_name="lab_results")
    op.drop_table("lab_results")
    op.drop_table("evaluation_runs")
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_index("ix_documents_patient_status", table_name="documents")
    op.drop_index(op.f("ix_documents_patient_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_checksum_sha256"), table_name="documents")
    op.drop_table("documents")
    op.drop_index(op.f("ix_chat_sessions_user_id"), table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_user_created", table_name="chat_sessions")
    op.drop_table("chat_sessions")
    op.drop_index(op.f("ix_audit_logs_request_id"), table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_created", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action_created", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_patients_external_id"), table_name="patients")
    op.drop_table("patients")
    op.drop_index(op.f("ix_evaluation_cases_dataset_version"), table_name="evaluation_cases")
    op.drop_index("ix_evaluation_cases_dataset_category", table_name="evaluation_cases")
    op.drop_index(op.f("ix_evaluation_cases_category"), table_name="evaluation_cases")
    op.drop_table("evaluation_cases")
