"""Agent workflow endpoint (manager/router, document tool, SQL tool, synthesis)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.core.errors import NotImplementedYetError

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run", summary="Run the multi-tool agent workflow")
async def run_agent(session: DbSession, principal: CurrentUser) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 4 - agentic workflow")
