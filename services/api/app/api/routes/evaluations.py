"""Evaluation run control and results."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUser, DbSession, require_roles
from app.core.errors import NotImplementedYetError
from app.models.enums import Role
from app.security.principal import Principal

router = APIRouter(prefix="/evaluations", tags=["evaluation"])


@router.post("/run", summary="Run the evaluation dataset against a configuration")
async def run_evaluation(
    session: DbSession,
    principal: Principal = Depends(require_roles(Role.ADMIN)),
) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 5 - evaluation harness")


@router.get("/{run_id}", summary="Metrics and per-case results for a run")
async def get_evaluation(
    run_id: uuid.UUID, session: DbSession, principal: CurrentUser
) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 5 - evaluation harness")
