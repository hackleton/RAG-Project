"""The SQL safety chain, against SQL Server.

The unit tests exercise the guard against SQLite, which shares the CTE
shadowing behaviour the scoping depends on - but "SQLite agrees" is not
evidence about T-SQL, and the scoping is only worth anything if the server's
query planner resolves the generated statement against the shadow tables.
This runs the same path against the real database.

It drives the tool directly rather than through `/agents/run`, because the
planner cannot be made to emit a chosen statement - and that is the point of
the design. The guard treats SQL as adversarial whatever its origin, so
handing it hostile statements is exactly how it is meant to be checked.

    docker compose run --rm api python scripts/smoke_sql_tool.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace

from sqlalchemy import func, select

from app.agents.factory import build_registry
from app.agents.registry import ToolCall, ToolRegistry
from app.core.config import Settings, get_settings
from app.core.db import (
    dispose_engine,
    dispose_readonly_engine,
    get_sessionmaker,
    init_engine,
    init_readonly_engine,
)
from app.models import LabResult, Patient, Role, User
from app.providers.llm import build_llm_provider
from app.rag.retriever import RetrievalConfig
from app.security.principal import Principal

# Statements that must never reach the database, whoever wrote them.
HOSTILE = [
    ("DELETE FROM lab_results", "a bare write"),
    ("SELECT * INTO copied_patients FROM patients", "a write that begins with SELECT"),
    ("SELECT value FROM dbo.lab_results", "reaching past the scoping CTE"),
    (
        "WITH lab_results AS (SELECT 1 AS value) SELECT value FROM lab_results",
        "redefining a base table",
    ),
    ("SELECT hashed_password FROM users", "a table outside the allow-list"),
    ("SELECT value FROM lab_results -- and more", "a comment"),
]


class SmokeFailure(AssertionError):
    pass


async def dispatch(
    registry: ToolRegistry, settings: Settings, principal: Principal, sql: str
) -> dict:
    async with get_sessionmaker()() as session:
        outcome = await registry.dispatch(
            session, principal, settings, ToolCall(name="run_sql", arguments={"sql": sql})
        )
        await session.commit()
    if not outcome.allowed or outcome.result is None:
        raise SmokeFailure(f"The tool itself was refused: {outcome.denial}")
    return outcome.result.data


async def main() -> int:
    settings = get_settings()
    if not settings.sql_tool_enabled:
        print("SQL_TOOL_ENABLED is false; nothing to check.", file=sys.stderr)
        return 1

    init_engine(settings)
    init_readonly_engine(settings)
    try:
        async with get_sessionmaker()() as session:
            admin = await session.scalar(select(User).where(User.role == Role.ADMIN))
            patients = list(await session.scalars(select(Patient).order_by(Patient.external_id)))
            total = await session.scalar(select(func.count(LabResult.id))) or 0
        if admin is None or len(patients) < 2:
            print("Run scripts/seed.py first.", file=sys.stderr)
            return 1

        registry = build_registry(
            settings=settings,
            retriever=None,  # type: ignore[arg-type]  # never reached by run_sql
            config=RetrievalConfig(),
            llm=build_llm_provider(settings),
        )
        unrestricted = Principal(id=admin.id, email=admin.email, name=admin.name, role=admin.role)
        scoped = replace(unrestricted, accessible_patient_ids=frozenset({patients[0].id}))

        print("- hostile statements are refused before execution")
        for sql, description in HOSTILE:
            data = await dispatch(registry, settings, unrestricted, sql)
            if not data.get("rejected"):
                raise SmokeFailure(f"{description} was NOT rejected: {sql}")
            print(f"    {data['rule']:<20} {description}")

        print("- row scope is applied by shadowing, on SQL Server")
        everything = await dispatch(
            registry, settings, unrestricted, "SELECT value FROM lab_results"
        )
        narrowed = await dispatch(registry, settings, scoped, "SELECT value FROM lab_results")

        if everything.get("rejected") or narrowed.get("rejected"):
            raise SmokeFailure("A legitimate query was rejected.")
        if len(everything["rows"]) != min(total, settings.sql_max_rows):
            raise SmokeFailure(
                f"Unrestricted scope returned {len(everything['rows'])} of {total} rows."
            )
        if not narrowed["rows"]:
            raise SmokeFailure("The scoped caller got no rows at all - the CTE matched nothing.")
        if len(narrowed["rows"]) >= len(everything["rows"]):
            raise SmokeFailure(
                f"Scoping had no effect: {len(narrowed['rows'])} rows for one patient "
                f"versus {len(everything['rows'])} for all of them."
            )
        print(
            f"    identical statement returned {len(everything['rows'])} rows unrestricted "
            f"and {len(narrowed['rows'])} scoped to one patient"
        )

        print("- a join is scoped on every table it reads")
        joined = await dispatch(
            registry,
            settings,
            scoped,
            "SELECT p.external_id, l.value FROM patients p JOIN lab_results l "
            "ON l.patient_id = p.id",
        )
        identifiers = {row["external_id"] for row in joined["rows"]}
        if identifiers != {patients[0].external_id}:
            raise SmokeFailure(f"A join leaked other patients: {sorted(identifiers)}.")
        print(f"    join returned only {sorted(identifiers)}")
    finally:
        await dispose_readonly_engine()
        await dispose_engine()

    print("\nSQL tool smoke checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except SmokeFailure as failure:
        print(f"  FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
