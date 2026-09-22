"""Structured lab results.

The safest SQL agent is the one that does not generate SQL. This tool answers
the lab questions people actually ask - one patient, one test, over time - with
a query whose shape was fixed at development time and whose only variables are
bound parameters. Nothing a model or a user writes can change which tables it
reads or which rows it returns.

`run_sql` exists for the questions this cannot express. It is not the default,
and the distinction is deliberate: reach for the generated-SQL path when the
typed one genuinely cannot answer the question, not because it is more general.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import Field
from sqlalchemy import select

from app.core.logging import get_logger
from app.models.patient import LabResult
from app.tools.base import Evidence, PatientScopedArguments, ToolContext, ToolResult

logger = get_logger(__name__)

MAX_RESULTS = 200


class QueryLabResultsArguments(PatientScopedArguments):
    test_name: str | None = Field(
        default=None,
        max_length=128,
        description="Exact or partial test name, e.g. 'Hemoglobin A1c'.",
    )
    since_days: int | None = Field(
        default=None, ge=1, le=3650, description="Only results collected within this many days."
    )
    limit: int = Field(default=50, ge=1, le=MAX_RESULTS)


class QueryLabResultsTool:
    """Parameterized lab history for one patient, on the read-only connection."""

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        args: QueryLabResultsArguments = arguments
        if context.patient_id is None:
            return ToolResult(
                summary="No patient was identified, and lab results are always patient-scoped."
            )

        # Columns, not entities. The tool wants values, and selecting the ORM
        # object would tie those values to the session's lifetime - they detach
        # the moment the read-only session closes, which is one line later.
        # Selecting columns also skips the identity map for rows nobody will
        # mutate.
        query = select(
            LabResult.test_name,
            LabResult.test_code,
            LabResult.value,
            LabResult.unit,
            LabResult.reference_low,
            LabResult.reference_high,
            LabResult.abnormal_flag,
            LabResult.collected_at,
            # The patient id comes from the dispatcher, which resolved it
            # against the caller's grants. It is never read from the arguments.
        ).where(LabResult.patient_id == context.patient_id)
        if args.test_name:
            query = query.where(LabResult.test_name.ilike(f"%{args.test_name}%"))
        if args.since_days:
            query = query.where(
                LabResult.collected_at >= datetime.now(UTC) - timedelta(days=args.since_days)
            )
        query = query.order_by(LabResult.test_name, LabResult.collected_at.desc()).limit(args.limit)

        async with context.readonly() as readonly:
            results = [
                {
                    "test_name": row.test_name,
                    "test_code": row.test_code,
                    "value": row.value,
                    "unit": row.unit,
                    "reference_low": row.reference_low,
                    "reference_high": row.reference_high,
                    "abnormal_flag": row.abnormal_flag,
                    "collected_at": row.collected_at.isoformat(),
                }
                for row in (await readonly.execute(query)).all()
            ]

        if not results:
            return ToolResult(
                summary=f"No lab results matched for {context.patient_external_id}.",
                trace={"rows": 0},
            )
        return ToolResult(
            summary=f"{len(results)} lab result(s) for {context.patient_external_id}.",
            evidence=[
                Evidence(
                    text=_as_table(context.patient_external_id, results),
                    source="lab_results",
                    kind="database",
                )
            ],
            data={"patient_external_id": context.patient_external_id, "results": results},
            trace={"rows": len(results), "tests": len({item["test_name"] for item in results})},
        )


def _as_table(external_id: str | None, results: list[dict[str, Any]]) -> str:
    """Render rows as text the model can read and cite.

    Reference ranges are included because an answer that gives a value without
    one is not useful, and the model must not supply the range from memory.
    """
    header = f"Lab results for patient {external_id} (source: lab_results table)"
    lines = ["Test | Result | Units | Reference range | Collected | Flag"]
    for item in results:
        reference = (
            f"{item['reference_low']}-{item['reference_high']}"
            if item["reference_low"] is not None
            else "n/a"
        )
        lines.append(
            f"{item['test_name']} | {item['value']} | {item['unit']} | {reference} | "
            f"{item['collected_at'][:10]} | {item['abnormal_flag'] or '-'}"
        )
    return f"{header}\n" + "\n".join(lines)
