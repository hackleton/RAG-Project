"""Retrieval inspection.

Runs retrieval and returns the candidates with every score behind their
ranking - per-strategy rank, raw score, fused score, reranker score - but no
generation. That is what the project plan's section 14 experiments compare, and
what makes a bad answer diagnosable: it separates "retrieval did not find it"
from "the model had it and did not use it".
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession, RagServiceDep, RetrievalConfigDep
from app.schemas.rag import SearchRequest, SearchResponse, SourceRead

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Inspect retrieval results without generating an answer",
)
async def search(
    payload: SearchRequest,
    session: DbSession,
    service: RagServiceDep,
    principal: CurrentUser,
    base_config: RetrievalConfigDep,
) -> SearchResponse:
    chunks, trace = await service.search(
        session,
        principal,
        payload.query,
        patient_external_id=payload.patient_external_id,
        config=base_config.with_overrides(payload.overrides),
    )
    return SearchResponse(
        query=payload.query,
        results=[SourceRead.from_chunk(chunk) for chunk in chunks],
        trace=trace,
    )
