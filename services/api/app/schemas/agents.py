"""Agent workflow payloads."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.service import AgentCitation, AgentRun, AgentStep
from app.schemas.rag import ChatUsage


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    session_id: uuid.UUID | None = None
    # Narrows the request to one patient. Never widens: the caller's grants
    # are still the ceiling, and a patient outside them is refused.
    patient_external_id: str | None = Field(default=None, max_length=64)
    # Lets a caller ask for a cheaper run. The deployed ceiling still applies.
    max_tool_calls: int | None = Field(default=None, ge=1, le=8)


class AgentStepRead(BaseModel):
    """One tool call, including the ones that were refused.

    Denied steps are returned rather than hidden. A caller who asked a
    question the system would not fully answer deserves to know which part was
    withheld, and an empty answer with no explanation is the failure mode that
    makes an agent feel broken.
    """

    tool: str
    arguments: dict[str, Any]
    reason: str | None
    allowed: bool
    summary: str
    denial_code: str | None
    duration_ms: int
    evidence: int
    data: dict[str, Any]
    trace: dict[str, Any]

    @classmethod
    def from_step(cls, step: AgentStep) -> AgentStepRead:
        return cls(
            tool=step.tool,
            arguments=step.arguments,
            reason=step.reason,
            allowed=step.allowed,
            summary=step.summary,
            denial_code=step.denial_code,
            duration_ms=step.duration_ms,
            evidence=step.evidence,
            data=step.data,
            trace=step.trace,
        )


class AgentCitationRead(BaseModel):
    number: int
    source: str
    # "document" or "database" - a citation can be either, and the frontend
    # renders them differently.
    kind: str
    excerpt: str
    chunk_id: uuid.UUID | None
    document_id: uuid.UUID | None
    section: str | None
    page_number: int | None
    score: float | None

    @classmethod
    def from_citation(cls, citation: AgentCitation) -> AgentCitationRead:
        return cls(
            number=citation.number,
            source=citation.source,
            kind=citation.kind,
            excerpt=citation.excerpt,
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            section=citation.section,
            page_number=citation.page_number,
            score=citation.score,
        )


class AgentRunResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    steps: list[AgentStepRead]
    citations: list[AgentCitationRead]
    trace: dict[str, Any]
    usage: ChatUsage

    @classmethod
    def from_run(cls, run: AgentRun) -> AgentRunResponse:
        return cls(
            session_id=run.session_id,
            message_id=run.message_id,
            answer=run.answer,
            steps=[AgentStepRead.from_step(step) for step in run.steps],
            citations=[AgentCitationRead.from_citation(item) for item in run.citations],
            trace=run.trace,
            usage=ChatUsage(
                prompt_tokens=run.prompt_tokens,
                completion_tokens=run.completion_tokens,
                latency_ms=run.latency_ms,
                estimated_cost_usd=run.estimated_cost_usd,
            ),
        )


class ToolCatalogueResponse(BaseModel):
    """The tools this caller may use.

    Exposed so the UI can show what the agent is able to do for *this* user,
    and so the role model is inspectable rather than folded into prose in a
    prompt somewhere.
    """

    planner: str
    tools: list[dict[str, Any]]
