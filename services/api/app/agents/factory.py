"""Wiring the registry.

The single place that says which tools exist and which roles may use them.
Both facts are configuration in the strict sense - they are read from here,
never inferred at a call site - so "who can do what" is one list to review
rather than a property distributed across seven modules.

The role assignments are worth reading as a set:

* `search_documents` and `search_knowledge` are open to every role. They are
  not a privilege escalation, because the retrieval filter already restricts
  each caller to their own scope, and an analyst with no patient grants
  retrieves only unattributed material.
* `retrieve_medications`, `get_patient_record`, `query_lab_results` and
  `generate_report` need patient grants, so an analyst is refused at the gate.
  That refusal is the plan's section 12 "call an unauthorized tool" case.
* `run_sql` goes to admins and analysts, not clinicians. Clinicians have typed
  tools for everything they need; handing them a generated-SQL path would add
  the project's least predictable input for no question it can answer that the
  typed tools cannot.
"""

from __future__ import annotations

from app.agents.registry import DispatchOutcome, ToolCall, ToolRegistry, ToolSpec
from app.core.config import Settings
from app.models.enums import AuditAction, Role
from app.providers.llm import LLMProvider
from app.rag.retriever import HybridRetriever, RetrievalConfig
from app.tools.base import ToolContext
from app.tools.documents import (
    GetPatientRecordArguments,
    GetPatientRecordTool,
    RetrieveMedicationsArguments,
    RetrieveMedicationsTool,
    SearchDocumentsArguments,
    SearchDocumentsTool,
    SearchKnowledgeArguments,
    SearchKnowledgeTool,
)
from app.tools.labs import QueryLabResultsArguments, QueryLabResultsTool
from app.tools.reports import GenerateReportArguments, GenerateReportTool
from app.tools.sql import RunSqlArguments, RunSqlTool

EVERY_ROLE = frozenset({Role.ADMIN, Role.CLINICIAN, Role.ANALYST})
PATIENT_ROLES = frozenset({Role.ADMIN, Role.CLINICIAN})
COHORT_ROLES = frozenset({Role.ADMIN, Role.ANALYST})


def build_registry(
    *,
    settings: Settings,
    retriever: HybridRetriever,
    config: RetrievalConfig,
    llm: LLMProvider,
) -> ToolRegistry:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="search_documents",
                description=(
                    "Search the clinical document corpus for passages relevant to a "
                    "question. Returns cited chunks with their section and page."
                ),
                arguments=SearchDocumentsArguments,
                allowed_roles=EVERY_ROLE,
                tool=SearchDocumentsTool(retriever, config),
            ),
            ToolSpec(
                name="search_knowledge",
                description=(
                    "Search reference material that belongs to no patient - clinical "
                    "guidelines and drug references. Never returns patient records."
                ),
                arguments=SearchKnowledgeArguments,
                allowed_roles=EVERY_ROLE,
                tool=SearchKnowledgeTool(retriever, config),
            ),
            ToolSpec(
                name="retrieve_medications",
                description=(
                    "Find the medication list for a patient from their notes and "
                    "discharge summaries."
                ),
                arguments=RetrieveMedicationsArguments,
                allowed_roles=PATIENT_ROLES,
                tool=RetrieveMedicationsTool(retriever, config),
            ),
            ToolSpec(
                name="get_patient_record",
                description="Demographics and the document inventory for one patient.",
                arguments=GetPatientRecordArguments,
                allowed_roles=PATIENT_ROLES,
                tool=GetPatientRecordTool(),
            ),
            ToolSpec(
                name="query_lab_results",
                description=(
                    "Structured laboratory results for one patient, optionally filtered "
                    "by test name and recency. Returns values with reference ranges."
                ),
                arguments=QueryLabResultsArguments,
                allowed_roles=PATIENT_ROLES,
                tool=QueryLabResultsTool(),
            ),
        ]
    )

    if settings.sql_tool_enabled:
        registry.register(
            ToolSpec(
                name="run_sql",
                description=(
                    "Answer an analytical question by running a validated read-only "
                    "SELECT over patients, lab_results and documents."
                ),
                arguments=RunSqlArguments,
                allowed_roles=COHORT_ROLES,
                tool=RunSqlTool(llm),
            )
        )

    # Composite tools last: they dispatch back through the registry, so it has
    # to exist - and so the tools they compose must already be registered.
    async def dispatch(call: ToolCall, context: ToolContext) -> DispatchOutcome:
        return await registry.dispatch(
            context.session,
            context.principal,
            context.settings,
            call,
            default_patient_external_id=context.patient_external_id,
        )

    registry.register(
        ToolSpec(
            name="generate_report",
            description=(
                "Compose a patient summary from the record, medications, recent labs "
                "and clinical documents."
            ),
            arguments=GenerateReportArguments,
            allowed_roles=PATIENT_ROLES,
            tool=GenerateReportTool(dispatch),
            # Plan section 11 calls out generated reports specifically.
            audit_action=AuditAction.REPORT_GENERATED,
        )
    )
    return registry
