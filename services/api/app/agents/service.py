"""The agent workflow.

    question
      -> planner proposes tool calls          (untrusted)
      -> registry authorises and runs each    (the control)
      -> evidence is pooled and de-duplicated
      -> synthesis composes one cited answer
      -> the run is persisted with its full trace

Sequential, not parallel. The tools share one database session and one
transaction, and the plan's section 9 fan-out is about which tools run, not
about running them at once. Concurrency here would buy tens of milliseconds
and cost a session-per-tool design; when a tool becomes slow enough to matter
that is the moment to pay for it, not before.

The run is persisted on the same `chat_sessions` / `chat_messages` tables the
RAG chat uses. An agent answer and a retrieval answer are the same thing to a
reader - a question, a grounded reply, citations, a trace - and giving them
one history means the evaluation harness and the cost dashboard do not each
need a second code path.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.planner import Planner
from app.agents.registry import DispatchOutcome, ToolRegistry
from app.agents.synthesis import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_prompt,
    collect_evidence,
    parse_citations,
)
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.chat import ChatMessage, ChatSession
from app.models.enums import ChatRole
from app.models.patient import Patient
from app.providers.llm import LLMProvider
from app.security.principal import Principal
from app.security.rbac import assert_patient_access
from app.tools.base import Evidence

logger = get_logger(__name__)

EXCERPT_CHARS = 320


@dataclass(frozen=True, slots=True)
class AgentCitation:
    number: int
    source: str
    kind: str
    excerpt: str
    chunk_id: uuid.UUID | None
    document_id: uuid.UUID | None
    section: str | None
    page_number: int | None
    score: float | None


@dataclass(slots=True)
class AgentStep:
    """One tool call and what became of it - allowed or not."""

    tool: str
    arguments: dict[str, Any]
    reason: str | None
    allowed: bool
    summary: str
    denial_code: str | None
    duration_ms: int
    evidence: int
    data: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRun:
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    steps: list[AgentStep]
    citations: list[AgentCitation]
    trace: dict[str, Any]
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None


class AgentService:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        planner: Planner,
        llm: LLMProvider,
        settings: Settings,
        planner_name: str,
    ) -> None:
        self._registry = registry
        self._planner = planner
        self._llm = llm
        self._settings = settings
        self._planner_name = planner_name

    async def run(
        self,
        session: AsyncSession,
        principal: Principal,
        question: str,
        *,
        session_id: uuid.UUID | None = None,
        patient_external_id: str | None = None,
        max_tool_calls: int | None = None,
    ) -> AgentRun:
        started = time.perf_counter()

        # An explicitly requested patient is authorised here so a caller who
        # names one they cannot see gets a straight refusal, the same as
        # /chat, rather than an answer that is empty for reasons they cannot
        # see. The dispatcher re-checks it per call regardless.
        if patient_external_id:
            await self._authorise_patient(session, principal, patient_external_id)

        available = self._registry.available_to(principal)
        budget = min(max_tool_calls or self._settings.agent_max_tool_calls, len(available) or 1)

        planning_started = time.perf_counter()
        calls = await self._planner.plan(
            question,
            principal,
            patient_external_id=patient_external_id,
            available=available,
            max_calls=budget,
        )
        planning_ms = round((time.perf_counter() - planning_started) * 1000, 2)

        outcomes: list[DispatchOutcome] = []
        for call in calls[:budget]:
            outcomes.append(
                await self._registry.dispatch(
                    session,
                    principal,
                    self._settings,
                    call,
                    default_patient_external_id=patient_external_id,
                )
            )

        evidence = collect_evidence(outcomes)
        synthesis_started = time.perf_counter()
        completion = await self._llm.complete(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user=build_synthesis_prompt(question, evidence, outcomes),
        )
        synthesis_ms = round((time.perf_counter() - synthesis_started) * 1000, 2)

        citations = _citations(completion.text, evidence)
        steps = [_step(outcome) for outcome in outcomes]
        latency_ms = int((time.perf_counter() - started) * 1000)

        trace = {
            "planner": self._planner_name,
            "available_tools": available,
            "planned": [call.name for call in calls],
            "executed": [step.tool for step in steps if step.allowed],
            "denied": [
                {"tool": step.tool, "reason": step.denial_code}
                for step in steps
                if not step.allowed
            ],
            "evidence": len(evidence),
            "cited": len(citations),
            "model": completion.model,
            "timings_ms": {
                "planning": planning_ms,
                "synthesis": synthesis_ms,
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
        }

        chat_session = await self._session_for(session, principal, session_id, question)
        session.add(ChatMessage(session_id=chat_session.id, role=ChatRole.USER, content=question))
        message = ChatMessage(
            session_id=chat_session.id,
            role=ChatRole.ASSISTANT,
            content=completion.text,
            citations=[_citation_payload(citation) for citation in citations],
            retrieval_trace={**trace, "steps": [_step_payload(step) for step in steps]},
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=completion.estimated_cost_usd,
        )
        session.add(message)
        await session.flush()

        logger.info("agent_run", **{key: trace[key] for key in ("planner", "executed", "denied")})
        return AgentRun(
            session_id=chat_session.id,
            message_id=message.id,
            answer=completion.text,
            steps=steps,
            citations=citations,
            trace=trace,
            latency_ms=latency_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            estimated_cost_usd=completion.estimated_cost_usd,
        )

    # --- internals --------------------------------------------------------

    async def _authorise_patient(
        self, session: AsyncSession, principal: Principal, external_id: str
    ) -> None:
        patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
        if patient is None:
            raise NotFoundError("Unknown patient.", details={"patient_external_id": external_id})
        assert_patient_access(principal, patient.id)

    async def _session_for(
        self,
        session: AsyncSession,
        principal: Principal,
        session_id: uuid.UUID | None,
        question: str,
    ) -> ChatSession:
        if session_id is not None:
            existing = await session.scalar(
                select(ChatSession).where(
                    ChatSession.id == session_id, ChatSession.user_id == principal.id
                )
            )
            if existing is None:
                raise NotFoundError("Chat session not found.")
            return existing
        created = ChatSession(user_id=principal.id, title=question[:120])
        session.add(created)
        await session.flush()
        return created


def _step(outcome: DispatchOutcome) -> AgentStep:
    result = outcome.result
    return AgentStep(
        tool=outcome.call.name,
        arguments=dict(outcome.call.arguments),
        reason=outcome.call.reason,
        allowed=outcome.allowed,
        summary=result.summary if result else (outcome.denial or "Denied."),
        denial_code=outcome.denial_code,
        duration_ms=outcome.duration_ms,
        evidence=len(result.evidence) if result else 0,
        data=result.data if result else {},
        trace=result.trace if result else {},
    )


def _citations(answer: str, evidence: list[Evidence]) -> list[AgentCitation]:
    return [
        AgentCitation(
            number=number,
            source=evidence[number - 1].source,
            kind=evidence[number - 1].kind,
            excerpt=_excerpt(evidence[number - 1].text),
            chunk_id=evidence[number - 1].chunk_id,
            document_id=evidence[number - 1].document_id,
            section=evidence[number - 1].section,
            page_number=evidence[number - 1].page_number,
            score=evidence[number - 1].score,
        )
        for number in parse_citations(answer, evidence)
    ]


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= EXCERPT_CHARS:
        return collapsed
    return collapsed[:EXCERPT_CHARS].rsplit(" ", maxsplit=1)[0] + "..."


def _citation_payload(citation: AgentCitation) -> dict[str, Any]:
    return {
        "number": citation.number,
        "source": citation.source,
        "kind": citation.kind,
        "excerpt": citation.excerpt,
        "chunk_id": str(citation.chunk_id) if citation.chunk_id else None,
        "document_id": str(citation.document_id) if citation.document_id else None,
        "section": citation.section,
        "page_number": citation.page_number,
        "score": citation.score,
    }


def _step_payload(step: AgentStep) -> dict[str, Any]:
    """What goes in the persisted trace.

    The step's `data` is left out on purpose: it holds result rows and chunk
    text, which is patient data, and this row is telemetry (plan section 15).
    The caller gets it in the response; the trace keeps counts.
    """
    return {
        "tool": step.tool,
        "allowed": step.allowed,
        "reason": step.reason,
        "denial_code": step.denial_code,
        "summary": step.summary,
        "evidence": step.evidence,
        "duration_ms": step.duration_ms,
        "trace": step.trace,
    }
