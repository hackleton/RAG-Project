"""Grounded chat endpoint.

Sprint 3 wires this to hybrid retrieval + semantic ranking and returns answers
with citations; Sprint 4 routes qualifying questions through the agent.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.core.errors import NotImplementedYetError

router = APIRouter(tags=["chat"])


@router.post("/chat", summary="Ask a question and get a grounded, cited answer")
async def chat(session: DbSession, principal: CurrentUser) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 3 - retrieval and generation")
