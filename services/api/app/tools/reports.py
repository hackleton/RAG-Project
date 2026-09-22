"""Patient summary reports.

A composite tool: it calls other tools rather than reading anything itself.
The calls go back through the registry's dispatcher instead of straight to the
tool objects, which costs an indirection and buys two things.

First, every sub-call meets the same authorisation gate - so a report can
never reach data the caller could not have asked for directly. Second, each
sub-call is audited in its own right, so the trail shows what a report was
built from rather than only that one was produced.

The dispatcher arrives by injection because the registry owns the tools and
this tool is one of them; passing it in is what keeps that from being a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from app.core.logging import get_logger
from app.tools.base import Evidence, PatientScopedArguments, ToolContext, ToolResult

if TYPE_CHECKING:
    from app.agents.registry import DispatchOutcome, ToolCall

logger = get_logger(__name__)

# What a summary is made of, in the order it reads best. Fixed rather than
# chosen by a model: a report whose sections vary run to run is not a report.
REPORT_SECTIONS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("Patient", "get_patient_record", {}),
    ("Medications", "retrieve_medications", {"query": "current medications"}),
    ("Recent laboratory results", "query_lab_results", {"limit": 20, "since_days": 365}),
    (
        "Clinical summary",
        "search_documents",
        {"query": "diagnosis assessment plan discharge summary", "top_k": 4},
    ),
)


class Dispatcher(Protocol):
    async def __call__(self, call: ToolCall, context: ToolContext) -> DispatchOutcome: ...


class GenerateReportArguments(PatientScopedArguments):
    focus: str | None = Field(
        default=None,
        max_length=200,
        description="Optional emphasis, e.g. 'diabetes control'.",
    )


class GenerateReportTool:
    def __init__(self, dispatcher: Dispatcher) -> None:
        self._dispatch = dispatcher

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        from app.agents.registry import ToolCall

        args: GenerateReportArguments = arguments
        if context.patient_id is None:
            return ToolResult(summary="A report needs a patient.")

        evidence: list[Evidence] = []
        sections: list[dict[str, Any]] = []
        for title, tool_name, tool_arguments in REPORT_SECTIONS:
            call_arguments = dict(tool_arguments)
            if args.focus and "query" in call_arguments:
                call_arguments["query"] = f"{call_arguments['query']} {args.focus}"
            call_arguments["patient_external_id"] = context.patient_external_id

            outcome = await self._dispatch(
                ToolCall(name=tool_name, arguments=call_arguments, reason=f"Report: {title}"),
                context,
            )
            if not outcome.allowed or outcome.result is None:
                # A section the caller may not have is omitted, and the report
                # says so. Silently dropping it would produce a document that
                # looks complete and is not.
                sections.append({"title": title, "tool": tool_name, "status": "unavailable"})
                continue

            evidence.extend(outcome.result.evidence)
            sections.append(
                {
                    "title": title,
                    "tool": tool_name,
                    "status": "ok",
                    "evidence": len(outcome.result.evidence),
                }
            )

        available = sum(1 for section in sections if section["status"] == "ok")
        logger.info(
            "report_composed",
            patient=context.patient_external_id,
            sections=available,
            evidence=len(evidence),
        )
        return ToolResult(
            summary=(
                f"Report for {context.patient_external_id}: {available} of "
                f"{len(sections)} sections, {len(evidence)} pieces of evidence."
            ),
            evidence=evidence,
            data={
                "patient_external_id": context.patient_external_id,
                "sections": sections,
                "focus": args.focus,
            },
            trace={"sections": sections},
        )
