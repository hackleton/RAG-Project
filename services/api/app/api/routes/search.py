"""Retrieval debugging endpoint.

Returns the raw candidate set, fusion scores and reranker ordering without
generation, which is what the retrieval experiments in section 14 measure.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.core.errors import NotImplementedYetError

router = APIRouter(tags=["search"])


@router.post("/search", summary="Inspect retrieval results without generating an answer")
async def search(session: DbSession, principal: CurrentUser) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 3 - retrieval and generation")
