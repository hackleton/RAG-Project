"""Free-form analytical SQL, end to end.

Implements the rest of plan section 10 around `sql_guard`:

    SQL generation -> [guard: validation, RBAC, scoping] -> parameterized
      execution -> result validation -> LLM interpretation

Off unless `SQL_TOOL_ENABLED=true`. It is the only tool whose input is a
generated program, and a deployment should opt into that rather than inherit
it. When it is off the registry does not advertise it, so the planner cannot
select it and there is nothing to reject at dispatch time.

The tool accepts SQL directly as well as generating it from a question. That
is not a hole - it is the design. The guard treats every statement as hostile
regardless of origin, so a statement written by a model, typed by an
administrator, or smuggled in through a document all meet the same wall. Being
able to feed it hostile SQL in a test is what makes the guard checkable at all.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import decimal
import time
import uuid
from typing import Any

from pydantic import Field
from sqlalchemy import bindparam, text
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.types import Uuid

from app.core.logging import get_logger
from app.models.enums import AuditAction
from app.providers.llm import LLMProvider
from app.security.audit import record_audit
from app.tools import sql_guard
from app.tools.base import Evidence, ToolArguments, ToolContext, ToolResult
from app.tools.sql_guard import SqlRejected

logger = get_logger(__name__)

SCHEMA_DOCUMENTATION = """\
patients(id, external_id, display_name, birth_year, sex)
lab_results(id, patient_id, test_name, test_code, value, unit,
            reference_low, reference_high, abnormal_flag, collected_at)
documents(id, patient_id, filename, document_type, status, page_count,
          chunk_count, access_level, created_at)

Known test_name values: 'Hemoglobin A1c', 'Fasting Glucose', 'Creatinine',
'LDL Cholesterol', 'Systolic Blood Pressure'.
"""

SQL_SYSTEM_PROMPT = f"""\
You translate a question into a single read-only SQL SELECT statement.

Schema:
{SCHEMA_DOCUMENTATION}

Rules:
1. Output only the SQL. No explanation, no code fences, no trailing semicolon.
2. SELECT only. Never write, never use INTO, never use comments.
3. Use unqualified table names exactly as spelled above.
4. Do not add access-control conditions. Row-level scope is applied by the \
application after you, and duplicating it here would be wrong as often as \
it was right.
5. Prefer aggregates when the question asks about a group rather than a person.
"""


class RunSqlArguments(ToolArguments):
    """Either a question to translate, or a statement to validate and run."""

    question: str | None = Field(default=None, max_length=1000)
    sql: str | None = Field(default=None, max_length=sql_guard.MAX_SQL_LENGTH)


class RunSqlTool:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        args: RunSqlArguments = arguments
        policy = sql_guard.policy_for(
            context.principal.role, max_rows=context.settings.sql_max_rows
        )

        statement, generated = await self._statement(args)
        if statement is None:
            return ToolResult(summary="No question or SQL was supplied.")

        try:
            validated = sql_guard.validate(statement, policy)
        except SqlRejected as rejection:
            return await self._rejected(context, statement, rejection, generated)

        scoped = sql_guard.apply_scope(
            validated,
            policy,
            accessible_patient_ids=context.principal.accessible_patient_ids,
            schema_prefix=sql_guard.schema_prefix_for(_backend(context)),
        )

        try:
            rows, columns, truncated, elapsed_ms = await self._execute(context, scoped)
        except TimeoutError:
            return await self._rejected(
                context,
                statement,
                SqlRejected(
                    f"The query exceeded {context.settings.readonly_query_timeout_seconds}s "
                    f"and was cancelled.",
                    rule="timeout",
                ),
                generated,
            )

        # Only identifiers, outcomes and counts - never a value from the result
        # set, which is patient data (plan section 15).
        await record_audit(
            context.session,
            action=AuditAction.SQL_EXECUTED,
            principal=context.principal,
            resource_type="sql",
            allowed=True,
            details={
                "tables": sorted(validated.tables),
                "columns": columns,
                "rows": len(rows),
                "truncated": truncated,
                "row_scope_applied": scoped.row_scope_applied,
                "aggregate_query": validated.aggregates,
                "generated_by_model": generated,
                "elapsed_ms": elapsed_ms,
            },
        )
        logger.info(
            "sql_executed",
            tables=sorted(validated.tables),
            rows=len(rows),
            row_scope_applied=scoped.row_scope_applied,
            elapsed_ms=elapsed_ms,
        )

        return ToolResult(
            summary=(
                f"Query returned {len(rows)} row(s) from {', '.join(sorted(validated.tables))}."
            ),
            evidence=[
                Evidence(
                    text=_as_table(columns, rows, statement),
                    source=", ".join(sorted(validated.tables)),
                    kind="database",
                )
            ],
            data={"columns": columns, "rows": rows, "truncated": truncated, "sql": statement},
            trace={
                "rules_passed": list(validated.rules_passed),
                "row_scope_applied": scoped.row_scope_applied,
                "rows": len(rows),
                "truncated": truncated,
                "elapsed_ms": elapsed_ms,
                "generated_by_model": generated,
            },
        )

    # --- steps ------------------------------------------------------------

    async def _statement(self, args: RunSqlArguments) -> tuple[str | None, bool]:
        """The SQL to validate, and whether a model wrote it."""
        if args.sql:
            return args.sql, False
        if not args.question:
            return None, False
        completion = await self._llm.complete(
            system=SQL_SYSTEM_PROMPT, user=f"Question: {args.question}\nSQL:"
        )
        return _strip_fences(completion.text), True

    async def _execute(
        self, context: ToolContext, scoped: sql_guard.ScopedSql
    ) -> tuple[list[dict[str, Any]], list[str], bool, int]:
        """Run the scoped statement, bounded in both time and rows.

        The row cap is applied to the *result* rather than spliced into the
        statement: a TOP or LIMIT inserted into generated SQL can change what
        it means, and truncating what comes back cannot. Note what that does
        and does not bound - the driver may still materialise the whole result
        set, so the cap protects the answer and the response payload, and the
        query timeout is what protects the process.
        """
        started = time.perf_counter()
        async with (
            context.readonly() as session,
            asyncio.timeout(context.settings.readonly_query_timeout_seconds),
        ):
            result = await session.execute(_bound(scoped))
            columns = list(result.keys())
            fetched = result.fetchmany(scoped.max_rows + 1)

        truncated = len(fetched) > scoped.max_rows
        rows = [
            {column: _json_safe(value) for column, value in zip(columns, row, strict=True)}
            for row in fetched[: scoped.max_rows]
        ]
        return rows, columns, truncated, int((time.perf_counter() - started) * 1000)

    async def _rejected(
        self,
        context: ToolContext,
        statement: str,
        rejection: SqlRejected,
        generated: bool,
    ) -> ToolResult:
        """Record the refusal and report it as a result, not an exception.

        The statement text is logged. It is a query the caller asked for, not
        patient data, and a rejection nobody can read afterwards teaches
        nothing about what the model tried to do.
        """
        await record_audit(
            context.session,
            action=AuditAction.SQL_REJECTED,
            principal=context.principal,
            resource_type="sql",
            allowed=False,
            details={
                "rule": rejection.rule,
                "reason": rejection.reason,
                "statement": statement[:1000],
                "generated_by_model": generated,
            },
        )
        logger.warning("sql_rejected", rule=rejection.rule, generated_by_model=generated)
        return ToolResult(
            summary=f"The query was rejected before execution ({rejection.rule}).",
            data={"rejected": True, "rule": rejection.rule, "reason": rejection.reason},
            trace={"rejected": True, "rule": rejection.rule},
        )


def _bound(scoped: sql_guard.ScopedSql) -> TextClause:
    """Bind the scope parameters with an explicit type.

    `text()` carries no type information, so the driver would receive whatever
    Python object it was handed. Declaring `Uuid` lets SQLAlchemy render each
    value the way the backend stores it, which is the difference between the
    scoping filter matching the caller's rows and matching nothing at all.
    """
    statement = text(scoped.sql)
    if not scoped.parameters:
        return statement
    return statement.bindparams(
        *(
            bindparam(name, value=value, type_=Uuid(as_uuid=True))
            for name, value in scoped.parameters.items()
        )
    )


def _backend(context: ToolContext) -> str:
    url = context.settings.sql_session_url
    return "sqlite" if url.startswith("sqlite") else "mssql"


def _strip_fences(text_value: str) -> str:
    """Models wrap SQL in code fences however firmly you ask them not to."""
    cleaned = text_value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def _json_safe(value: Any) -> Any:
    """Coerce driver types the JSON encoder cannot represent.

    Anything still unrecognised becomes its string form rather than raising:
    a result set that cannot be serialised should degrade to something
    readable, not turn a successful query into a 500.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        # Binary columns have no business in an answer, and echoing them into
        # a prompt is a way to spend a lot of tokens on nothing.
        return f"<{len(bytes(value))} bytes>"
    return str(value)


def _as_table(columns: list[str], rows: list[dict[str, Any]], statement: str) -> str:
    header = f"Query result (SQL: {' '.join(statement.split())})"
    lines = [" | ".join(columns)]
    lines.extend(" | ".join(str(row[column]) for column in columns) for row in rows)
    return f"{header}\n" + "\n".join(lines)
