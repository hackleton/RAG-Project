"""The agent workflow: manager/router, tools, synthesis (plan section 9)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import (
    AgentServiceDep,
    CurrentUser,
    DbSession,
    SettingsDep,
    ToolRegistryDep,
)
from app.schemas.agents import AgentRunRequest, AgentRunResponse, ToolCatalogueResponse

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Route a question to tools and synthesise a grounded answer",
    responses={
        404: {"description": "Unknown patient, or a chat session the caller does not own."},
        503: {"description": "The language model is unreachable."},
    },
)
async def run_agent(
    payload: AgentRunRequest,
    session: DbSession,
    service: AgentServiceDep,
    principal: CurrentUser,
) -> AgentRunResponse:
    run = await service.run(
        session,
        principal,
        payload.question,
        session_id=payload.session_id,
        patient_external_id=payload.patient_external_id,
        max_tool_calls=payload.max_tool_calls,
    )
    return AgentRunResponse.from_run(run)


@router.get(
    "/tools",
    response_model=ToolCatalogueResponse,
    summary="The tools available to the calling role",
)
async def list_tools(
    registry: ToolRegistryDep,
    principal: CurrentUser,
    settings: SettingsDep,
) -> ToolCatalogueResponse:
    return ToolCatalogueResponse(
        planner=settings.effective_planner.value,
        tools=registry.catalogue(principal),
    )
