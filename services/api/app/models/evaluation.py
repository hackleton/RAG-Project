"""Evaluation dataset, runs and per-case results (filled in from Sprint 5).

The schema lands in Sprint 1 so that migrations are additive later and so the
evaluation harness has a stable contract to target.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UnicodeText
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin


class EvaluationCase(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_cases"
    __table_args__ = (Index("ix_evaluation_cases_dataset_category", "dataset_version", "category"),)

    dataset_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    question: Mapped[str] = mapped_column(UnicodeText, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(UnicodeText)
    # Key facts an answer must contain - graded independently of wording.
    expected_facts: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    expected_sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONType, default=list, nullable=False
    )
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)


class EvaluationRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_runs"

    dataset_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # The retrieval/generation configuration under test - the experiment record
    # required by project plan section 14.
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    triggered_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (Index("ix_evaluation_results_run_case", "run_id", "case_id"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("evaluation_cases.id", ondelete="CASCADE")
    )
    answer: Mapped[str | None] = mapped_column(UnicodeText)
    retrieved_sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONType, default=list, nullable=False
    )
    scores: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(UnicodeText)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")
