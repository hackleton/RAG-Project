"""The generated-SQL tool against a real database.

`test_sql_guard.py` checks the guard's reasoning; these check that the
reasoning survives contact with a query planner. The scoping trick - CTEs that
take the names of the base tables - is only worth anything if the database
genuinely resolves the generated statement against them, and that is not
something a unit test on strings can tell you.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.registry import ToolCall, ToolRegistry
from app.core.config import Settings
from app.models import AuditAction, AuditLog, LabResult, Patient
from tests.conftest import principal_for


@pytest.fixture
async def labs(session: AsyncSession, patients: list[Patient]) -> list[LabResult]:
    """Two patients with distinguishable values, so scope is visible in results."""
    rows = [
        LabResult(
            patient_id=patients[0].id,
            test_name="Hemoglobin A1c",
            value=7.4,
            unit="%",
            reference_low=4.0,
            reference_high=5.6,
            collected_at=datetime(2026, 3, 1, tzinfo=UTC),
        ),
        LabResult(
            patient_id=patients[1].id,
            test_name="Hemoglobin A1c",
            value=9.9,
            unit="%",
            reference_low=4.0,
            reference_high=5.6,
            collected_at=datetime(2026, 3, 1, tzinfo=UTC),
        ),
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def run_sql(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    principal: object,
    **arguments: object,
) -> dict:
    outcome = await registry.dispatch(
        session,
        principal,  # type: ignore[arg-type]
        settings,
        ToolCall(name="run_sql", arguments=arguments),
    )
    assert outcome.allowed, outcome.denial
    assert outcome.result is not None
    return outcome.result.data


# --- scoping, against the database ---------------------------------------


async def test_a_clinician_sees_only_their_granted_patients_rows(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    """The statement asks for every row; the CTE is what makes that untrue."""
    # Clinicians do not hold `run_sql`, so this is an admin narrowed to one
    # patient's grants - the smallest change that isolates the scoping itself.
    scoped = replace(
        principal_for(users["admin"]), accessible_patient_ids=frozenset({patients[0].id})
    )

    data = await run_sql(
        registry,
        session,
        settings,
        scoped,
        sql="SELECT value FROM lab_results ORDER BY value",
    )

    assert [row["value"] for row in data["rows"]] == [7.4]


async def test_an_admin_with_unrestricted_scope_sees_every_row(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    data = await run_sql(
        registry,
        session,
        settings,
        principal_for(users["admin"]),
        sql="SELECT value FROM lab_results ORDER BY value",
    )

    assert [row["value"] for row in data["rows"]] == [7.4, 9.9]


async def test_a_join_is_scoped_on_every_table_it_reads(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    scoped = replace(
        principal_for(users["admin"]), accessible_patient_ids=frozenset({patients[0].id})
    )

    data = await run_sql(
        registry,
        session,
        settings,
        scoped,
        sql=(
            "SELECT p.external_id, l.value FROM patients p "
            "JOIN lab_results l ON l.patient_id = p.id"
        ),
    )

    assert {row["external_id"] for row in data["rows"]} == {patients[0].external_id}


# --- rejection, end to end ------------------------------------------------


async def test_a_write_is_rejected_the_rows_survive_and_it_is_audited(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    before = await session.scalar(select(func.count(LabResult.id)))

    outcome = await registry.dispatch(
        session,
        principal_for(users["admin"]),
        settings,
        ToolCall(name="run_sql", arguments={"sql": "DELETE FROM lab_results"}),
    )

    # The tool ran and reported a refusal; the statement did not run at all.
    assert outcome.allowed
    assert outcome.result is not None
    assert outcome.result.data["rejected"] is True

    # The assertion that actually matters: the rows are still there.
    assert await session.scalar(select(func.count(LabResult.id))) == before

    await session.commit()
    rejections = list(
        await session.scalars(select(AuditLog).where(AuditLog.action == AuditAction.SQL_REJECTED))
    )
    assert len(rejections) == 1
    assert rejections[0].allowed is False
    # Which rule fired is recorded, so a refusal is diagnosable afterwards.
    assert rejections[0].details["rule"] == outcome.result.data["rule"]
    assert rejections[0].details["statement"] == "DELETE FROM lab_results"


async def test_a_successful_query_is_audited_without_its_values(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    await run_sql(
        registry,
        session,
        settings,
        principal_for(users["admin"]),
        sql="SELECT test_name, value FROM lab_results",
    )

    await session.commit()
    executed = list(
        await session.scalars(select(AuditLog).where(AuditLog.action == AuditAction.SQL_EXECUTED))
    )
    assert len(executed) == 1
    details = executed[0].details
    assert details["rows"] == 2
    assert details["tables"] == ["lab_results"]
    assert details["columns"] == ["test_name", "value"]
    # Column names, counts and outcomes - never a result value.
    assert "7.4" not in str(details)


# --- the analyst's cohort path -------------------------------------------


async def test_an_analyst_can_aggregate_across_the_cohort(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    """No patient grants, yet a cohort statistic is exactly what the role is for."""
    data = await run_sql(
        registry,
        session,
        settings,
        principal_for(users["analyst"], []),
        sql="SELECT test_name, AVG(value) AS mean FROM lab_results GROUP BY test_name",
    )

    assert len(data["rows"]) == 1
    assert data["rows"][0]["mean"] == pytest.approx(8.65)


async def test_an_analyst_cannot_break_the_cohort_down_by_patient(
    registry: ToolRegistry,
    session: AsyncSession,
    settings: Settings,
    users: dict,
    patients: list[Patient],
    labs: list[LabResult],
) -> None:
    outcome = await registry.dispatch(
        session,
        principal_for(users["analyst"], []),
        settings,
        ToolCall(
            name="run_sql",
            arguments={"sql": "SELECT patient_id, AVG(value) FROM lab_results GROUP BY patient_id"},
        ),
    )

    assert outcome.result is not None
    assert outcome.result.data["rule"] == "identifying_column"


# --- registration ---------------------------------------------------------


def test_the_sql_tool_is_absent_when_it_is_disabled(
    settings: Settings, registry: ToolRegistry
) -> None:
    """Off by default in a deployment; the registry must not advertise it."""
    from app.agents.factory import build_registry
    from app.rag.retriever import RetrievalConfig

    assert "run_sql" in registry.names

    disabled = build_registry(
        settings=settings.model_copy(update={"sql_tool_enabled": False}),
        retriever=None,  # type: ignore[arg-type]
        config=RetrievalConfig(),
        llm=None,  # type: ignore[arg-type]
    )
    assert "run_sql" not in disabled.names
