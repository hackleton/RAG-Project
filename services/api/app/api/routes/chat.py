"""Grounded chat.

Retrieval, generation, citation verification and persistence of the turn with
its trace. Citations point at chunk ids the application already holds, so every
one is checkable rather than merely plausible.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession, RagServiceDep, RetrievalConfigDep
from app.schemas.rag import ChatRequest, ChatResponse, ChatUsage, CitationRead, SourceRead

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question and get a grounded, cited answer",
    responses={
        404: {"description": "Unknown patient, or a chat session the caller does not own."},
        503: {"description": "The language model is unreachable."},
    },
)
async def chat(
    payload: ChatRequest,
    session: DbSession,
    service: RagServiceDep,
    principal: CurrentUser,
    base_config: RetrievalConfigDep,
) -> ChatResponse:
    answer = await service.answer(
        session,
        principal,
        payload.question,
        session_id=payload.session_id,
        patient_external_id=payload.patient_external_id,
        config=base_config.with_overrides(payload.overrides),
    )
    return ChatResponse(
        session_id=answer.session_id,
        message_id=answer.message_id,
        answer=answer.answer,
        citations=[CitationRead.from_citation(citation) for citation in answer.citations],
        sources=[SourceRead.from_chunk(chunk) for chunk in answer.sources],
        trace=answer.trace,
        usage=ChatUsage(
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            latency_ms=answer.latency_ms,
            estimated_cost_usd=answer.estimated_cost_usd,
        ),
    )
