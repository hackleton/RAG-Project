"""Evaluation run control and results.

Running the dataset takes minutes, not milliseconds, so `POST /evaluations/run`
starts the work and returns immediately with a run id. The row is created and
committed inside the request's own transaction, before the response goes out,
so the id the caller gets always resolves - and a process that dies mid-run
leaves a record saying so rather than leaving nothing.

Launching a run is admin-only: it drives the whole corpus through retrieval and
generation as three different users, which is both expensive and privileged.
Reading results is not, because a run's numbers are about the system rather
than about any patient, and a result row carries source provenance only - never
chunk text.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession, RetrievalConfigDep, require_roles
from app.core.config import Settings
from app.core.db import get_sessionmaker
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.evaluation.dataset import DatasetError, load_dataset, sync_cases
from app.evaluation.harness import FAILED, EvaluationHarness, create_run
from app.evaluation.paths import dataset_path
from app.models.enums import Role
from app.models.evaluation import EvaluationCase, EvaluationRun
from app.rag.retriever import HybridRetriever, RetrievalConfig
from app.rag.service import RagService
from app.schemas.evaluations import (
    EvaluationResultRead,
    EvaluationRunDetail,
    EvaluationRunList,
    EvaluationRunRead,
    RunEvaluationRequest,
)
from app.security.principal import Principal

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluation"])


@router.post(
    "/run",
    response_model=EvaluationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run the evaluation dataset against a configuration",
    responses={
        403: {"description": "Only an administrator may launch a run."},
        422: {"description": "No such dataset version, or the file is malformed."},
    },
)
async def run_evaluation(
    payload: RunEvaluationRequest,
    request: Request,
    background: BackgroundTasks,
    session: DbSession,
    base_config: RetrievalConfigDep,
    principal: Principal = Depends(require_roles(Role.ADMIN)),
) -> EvaluationRunRead:
    try:
        dataset = load_dataset(dataset_path(payload.dataset_version))
    except (DatasetError, ValueError) as exc:
        # Loaded and synced on the request's own transaction, so a malformed
        # dataset fails the call rather than a background task nobody is
        # watching.
        raise ValidationError(str(exc)) from exc

    await sync_cases(session, dataset)

    config = base_config.with_overrides(payload.overrides)
    run = create_run(
        session,
        dataset_version=dataset.version,
        config=config,
        triggered_by=principal.id,
    )
    await session.commit()

    background.add_task(
        _execute,
        request.app,
        run_id=run.id,
        dataset_version=dataset.version,
        config=config,
        limit=payload.limit,
        triggered_by=principal.id,
    )
    logger.info(
        "evaluation_run_queued",
        run_id=str(run.id),
        dataset_version=dataset.version,
        cases=len(dataset),
        limit=payload.limit,
    )
    return EvaluationRunRead.from_run(run)


@router.get("", response_model=EvaluationRunList, summary="Recent evaluation runs")
async def list_evaluations(
    session: DbSession,
    principal: CurrentUser,
    limit: int = Query(default=20, ge=1, le=100),
) -> EvaluationRunList:
    runs = await session.scalars(
        select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
    )
    return EvaluationRunList(runs=[EvaluationRunRead.from_run(run) for run in runs])


@router.get(
    "/{run_id}",
    response_model=EvaluationRunDetail,
    summary="Metrics and per-case results for a run",
    responses={404: {"description": "No such run."}},
)
async def get_evaluation(
    run_id: uuid.UUID,
    session: DbSession,
    principal: CurrentUser,
    failures_only: bool = Query(
        default=False, description="Only the cases that did not pass - the useful view."
    ),
) -> EvaluationRunDetail:
    run = await session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(selectinload(EvaluationRun.results))
    )
    if run is None:
        raise NotFoundError("Evaluation run not found.")

    cases = await _cases_for(session, [result.case_id for result in run.results])
    results = [
        EvaluationResultRead.from_result(result, cases[result.case_id])
        for result in sorted(run.results, key=lambda item: item.created_at)
        if result.case_id in cases
    ]
    if failures_only:
        results = [result for result in results if not result.passed]

    return EvaluationRunDetail(run=EvaluationRunRead.from_run(run), results=results)


async def _cases_for(
    session: AsyncSession, case_ids: list[uuid.UUID]
) -> dict[uuid.UUID, EvaluationCase]:
    if not case_ids:
        return {}
    return {
        case.id: case
        for case in await session.scalars(
            select(EvaluationCase).where(EvaluationCase.id.in_(case_ids))
        )
    }


# --- the background half --------------------------------------------------


async def _execute(
    app: FastAPI,
    *,
    run_id: uuid.UUID,
    dataset_version: str,
    config: RetrievalConfig,
    limit: int | None,
    triggered_by: uuid.UUID,
) -> None:
    """Run the dataset on a session of its own.

    The request's session is closed by the time a background task runs, so this
    opens another. Any failure is written to the run row before it propagates:
    a run that vanished and a run that failed look identical to a poller
    otherwise, and only one of them is worth investigating.
    """
    state = app.state
    settings: Settings = state.settings
    rag = RagService(
        retriever=HybridRetriever(
            store=state.vector_store,
            embedder=state.embedder,
            reranker=state.reranker,
            config=config,
        ),
        llm=state.llm,
    )

    async with get_sessionmaker()() as session:
        try:
            dataset = load_dataset(dataset_path(dataset_version))
            run = await session.get(EvaluationRun, run_id)
            await EvaluationHarness(rag=rag, config=config).run(
                session,
                dataset,
                run=run,
                triggered_by=triggered_by,
                environment={
                    "embedding_model": settings.embedding_model,
                    "llm_provider": settings.llm_provider.value,
                    "launched_from": "api",
                },
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001 - recorded on the row, not swallowed
            logger.exception("evaluation_run_failed", run_id=str(run_id))
            await session.rollback()
            failed = await session.get(EvaluationRun, run_id)
            if failed is not None:
                failed.status = FAILED
                failed.metrics = {"error": f"{type(exc).__name__}: {exc}"}
                await session.commit()
