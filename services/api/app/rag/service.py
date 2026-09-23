"""Grounded chat and retrieval inspection.

    question -> retrieve -> build prompt -> generate -> parse and verify
                citations -> persist the turn with its trace

The turn is persisted with its citations, retrieval trace, token counts and
latency on the same row the user saw. Project plan section 15 wants that
telemetry; keeping it on the message rather than in a separate log means the
evaluation harness and the cost dashboard read exactly what was served, and
cannot drift from it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.chat import ChatMessage, ChatSession
from app.models.enums import AuditAction, ChatRole
from app.models.patient import Patient
from app.providers.llm import LLMProvider
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt, parse_citations
from app.rag.retriever import HybridRetriever, RetrievalConfig, RetrievedChunk
from app.security import injection
from app.security.audit import record_audit
from app.security.principal import Principal
from app.security.rbac import assert_patient_access

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Citation:
    """One cited source, traceable back to an exact chunk."""

    number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    section: str | None
    page_number: int | None
    excerpt: str
    score: float | None


@dataclass(slots=True)
class ChatAnswer:
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    sources: list[RetrievedChunk]
    trace: dict[str, Any]
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None


EXCERPT_CHARS = 320


class RagService:
    def __init__(self, *, retriever: HybridRetriever, llm: LLMProvider) -> None:
        self._retriever = retriever
        self._llm = llm

    async def search(
        self,
        session: AsyncSession,
        principal: Principal,
        query: str,
        *,
        patient_external_id: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """Retrieval only - no generation. The experiment and debug path."""
        patient_id = await self._resolve_patient(session, principal, patient_external_id)
        result = await self._retriever.retrieve(
            session, principal, query, patient_id=patient_id, config=config
        )
        return result.chunks, result.trace

    async def answer(
        self,
        session: AsyncSession,
        principal: Principal,
        question: str,
        *,
        session_id: uuid.UUID | None = None,
        patient_external_id: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> ChatAnswer:
        started = time.perf_counter()
        patient_id = await self._resolve_patient(session, principal, patient_external_id)

        retrieval = await self._retriever.retrieve(
            session, principal, question, patient_id=patient_id, config=config
        )

        completion = await self._llm.complete(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(question, retrieval.chunks),
        )
        answer_text, leaked = injection.redact_system_prompt(completion.text, SYSTEM_PROMPT)
        if leaked:
            # Section 12's first scenario. Nothing upstream can prevent it: the
            # system prompt is not data the caller lacks permission for, it is
            # text the model is holding. Checking the output is the only
            # control that survives swapping the model.
            retrieval.trace["injection"] = dict(retrieval.trace.get("injection", {})) | {
                "system_prompt_redacted": True
            }
            await record_audit(
                session,
                action=AuditAction.INJECTION_BLOCKED,
                principal=principal,
                resource_type="answer",
                allowed=True,
                details={"reason": "system_prompt_disclosure"},
            )
            logger.warning("system_prompt_redacted", question_chars=len(question))

        citations = self._citations(answer_text, retrieval.chunks)
        latency_ms = int((time.perf_counter() - started) * 1000)

        chat_session = await self._session_for(session, principal, session_id, question, patient_id)
        session.add(ChatMessage(session_id=chat_session.id, role=ChatRole.USER, content=question))
        message = ChatMessage(
            session_id=chat_session.id,
            role=ChatRole.ASSISTANT,
            content=answer_text,
            citations=[_citation_payload(citation) for citation in citations],
            retrieval_trace={**retrieval.trace, "model": completion.model},
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=completion.estimated_cost_usd,
        )
        session.add(message)
        await session.flush()

        logger.info(
            "chat_answered",
            session_id=str(chat_session.id),
            sources=len(retrieval.chunks),
            cited=len(citations),
            latency_ms=latency_ms,
            model=completion.model,
        )
        return ChatAnswer(
            session_id=chat_session.id,
            message_id=message.id,
            answer=answer_text,
            citations=citations,
            sources=retrieval.chunks,
            trace=retrieval.trace,
            latency_ms=latency_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            estimated_cost_usd=completion.estimated_cost_usd,
        )

    # --- internals --------------------------------------------------------

    def _citations(self, answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        """Map the answer's bracket numbers back to real chunks.

        `parse_citations` has already dropped anything out of range, so every
        citation returned here points at a chunk that was genuinely retrieved.
        """
        return [
            Citation(
                number=number,
                chunk_id=chunks[number - 1].chunk_id,
                document_id=chunks[number - 1].document_id,
                filename=chunks[number - 1].document_filename,
                section=chunks[number - 1].section,
                page_number=chunks[number - 1].page_number,
                excerpt=_excerpt(chunks[number - 1].text),
                score=chunks[number - 1].rerank_score or chunks[number - 1].fused_score,
            )
            for number in parse_citations(answer, chunks)
        ]

    async def _resolve_patient(
        self, session: AsyncSession, principal: Principal, external_id: str | None
    ) -> uuid.UUID | None:
        if not external_id:
            return None
        patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
        if patient is None:
            raise NotFoundError("Unknown patient.", details={"patient_external_id": external_id})
        assert_patient_access(principal, patient.id)
        return patient.id

    async def _session_for(
        self,
        session: AsyncSession,
        principal: Principal,
        session_id: uuid.UUID | None,
        question: str,
        patient_id: uuid.UUID | None,
    ) -> ChatSession:
        if session_id is not None:
            existing = await session.scalar(
                select(ChatSession).where(
                    ChatSession.id == session_id, ChatSession.user_id == principal.id
                )
            )
            if existing is None:
                # Scoped to the caller, so another user's session id is simply
                # not found rather than forbidden.
                raise NotFoundError("Chat session not found.")
            return existing

        created = ChatSession(user_id=principal.id, title=question[:120], patient_id=patient_id)
        session.add(created)
        await session.flush()
        return created


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= EXCERPT_CHARS:
        return collapsed
    return collapsed[:EXCERPT_CHARS].rsplit(" ", maxsplit=1)[0] + "..."


def _citation_payload(citation: Citation) -> dict[str, Any]:
    return {
        "number": citation.number,
        "chunk_id": str(citation.chunk_id),
        "document_id": str(citation.document_id),
        "filename": citation.filename,
        "section": citation.section,
        "page_number": citation.page_number,
        "excerpt": citation.excerpt,
        "score": citation.score,
    }
