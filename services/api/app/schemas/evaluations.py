"""Evaluation run payloads."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.evaluation import EvaluationCase, EvaluationResult, EvaluationRun
from app.schemas.rag import RetrievalOverrides


class RunEvaluationRequest(BaseModel):
    """What to run, and with what retrieval settings.

    `overrides` is the same patch object `/chat` and `/search` accept, so an
    experiment is expressed identically wherever it is launched from.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(default="v1", max_length=32, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    # A bound on how much work one request can queue. The full dataset takes a
    # couple of minutes; this exists so a smoke run is one field away.
    limit: int | None = Field(default=None, ge=1, le=1000)
    overrides: RetrievalOverrides | None = None


class EvaluationRunRead(BaseModel):
    id: uuid.UUID
    dataset_version: str
    status: str
    case_count: int
    config: dict[str, Any]
    metrics: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    @classmethod
    def from_run(cls, run: EvaluationRun) -> EvaluationRunRead:
        return cls(
            id=run.id,
            dataset_version=run.dataset_version,
            status=run.status,
            case_count=run.case_count,
            config=run.config,
            metrics=run.metrics,
            started_at=run.started_at,
            completed_at=run.completed_at,
            created_at=run.created_at,
        )


class EvaluationResultRead(BaseModel):
    """One case's outcome.

    Carries the answer and the scores but not the retrieved chunk text - the
    result rows keep provenance only, for the same reason the retrieval trace
    does (project plan section 15: no raw document content in telemetry).
    """

    case_key: str
    question: str
    category: str
    difficulty: str
    persona: str
    passed: bool
    answer: str | None
    scores: dict[str, Any]
    retrieved_sources: list[dict[str, Any]]
    latency_ms: int | None
    error: str | None

    @classmethod
    def from_result(cls, result: EvaluationResult, case: EvaluationCase) -> EvaluationResultRead:
        return cls(
            case_key=case.case_key,
            question=case.question,
            category=case.category,
            difficulty=case.difficulty,
            persona=case.persona,
            passed=bool(result.scores.get("passed")),
            answer=result.answer,
            scores=result.scores,
            retrieved_sources=result.retrieved_sources,
            latency_ms=result.latency_ms,
            error=result.error,
        )


class EvaluationRunDetail(BaseModel):
    run: EvaluationRunRead
    results: list[EvaluationResultRead]


class EvaluationRunList(BaseModel):
    runs: list[EvaluationRunRead]
