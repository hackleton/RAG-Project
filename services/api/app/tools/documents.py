"""Tools that read documents.

Three of the four are the retriever with different narrowings, and that is the
point: `search_documents`, `retrieve_medications` and `search_knowledge` share
one access filter, one fusion, one reranker. A tool that reimplemented
retrieval would be a second place for the pre-search filter to be wrong.

What each tool contributes is *intent* - which subset of the corpus the
question is really about, and what to say when the answer is empty.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import Field
from sqlalchemy import func, select

from app.models.document import Document
from app.models.patient import Patient
from app.rag.retriever import HybridRetriever, RetrievalConfig
from app.tools.base import (
    Evidence,
    PatientScopedArguments,
    ToolArguments,
    ToolContext,
    ToolResult,
)

# Section headings the corpus uses for medication lists. The tool knows what a
# medication section is called; the user should not have to.
MEDICATION_TERMS = "current medications discharge medications prescription dose daily"


class SearchDocumentsArguments(PatientScopedArguments):
    query: str = Field(min_length=1, max_length=1000)
    document_type: str | None = Field(
        default=None, max_length=64, description="clinical_note, lab_report, discharge_summary"
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrieveMedicationsArguments(PatientScopedArguments):
    query: str = Field(default="current medications", min_length=1, max_length=1000)


class SearchKnowledgeArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=1000)


class GetPatientRecordArguments(PatientScopedArguments):
    pass


class SearchDocumentsTool:
    """Hybrid retrieval over the documents the caller may see."""

    def __init__(self, retriever: HybridRetriever, config: RetrievalConfig) -> None:
        self._retriever = retriever
        self._config = config

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        args: SearchDocumentsArguments = arguments
        config = self._config if args.top_k is None else _with_top_k(self._config, args.top_k)
        result = await self._retriever.retrieve(
            context.session,
            context.principal,
            args.query,
            patient_id=context.patient_id,
            document_types=frozenset({args.document_type}) if args.document_type else None,
            config=config,
        )
        return _retrieval_result(result.chunks, result.trace, "document")


class RetrieveMedicationsTool:
    """Retrieval aimed at medication sections.

    The query is expanded with the headings the corpus actually uses. That is
    domain knowledge, and encoding it in the tool is the honest place for it:
    the alternative is hoping the user phrases the question the way the
    documents are written.
    """

    def __init__(self, retriever: HybridRetriever, config: RetrievalConfig) -> None:
        self._retriever = retriever
        self._config = config

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        args: RetrieveMedicationsArguments = arguments
        result = await self._retriever.retrieve(
            context.session,
            context.principal,
            f"{args.query} {MEDICATION_TERMS}",
            patient_id=context.patient_id,
            config=self._config,
        )
        outcome = _retrieval_result(result.chunks, result.trace, "medication")
        outcome.trace["query_expanded"] = True
        return outcome


class SearchKnowledgeTool:
    """Reference material only - guidelines and drug references, never records.

    Narrowed to documents with no patient link. That is a real restriction
    rather than a hint: an empty patient scope matches only unattributed
    documents in the index filter, so this tool cannot return a patient
    document even for an administrator.
    """

    def __init__(self, retriever: HybridRetriever, config: RetrievalConfig) -> None:
        self._retriever = retriever
        self._config = config

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        args: SearchKnowledgeArguments = arguments
        result = await self._retriever.retrieve(
            context.session,
            context.principal,
            args.query,
            exclude_patient_documents=True,
            config=self._config,
        )
        outcome = _retrieval_result(result.chunks, result.trace, "knowledge")
        outcome.trace["patient_documents_excluded"] = True
        return outcome


class GetPatientRecordTool:
    """Demographics and the document inventory for one patient.

    Structured, parameterized ORM reads - no generated SQL. The safest query is
    the one whose shape was decided at development time; free-form SQL exists
    for the questions this cannot express, not for the ones it can.
    """

    async def run(self, context: ToolContext, arguments: Any) -> ToolResult:
        if context.patient_id is None:
            return ToolResult(summary="No patient was identified in the request.")

        patient = await context.session.get(Patient, context.patient_id)
        if patient is None:
            return ToolResult(summary="No such patient.")

        rows = (
            await context.session.execute(
                select(
                    Document.document_type,
                    func.count(Document.id),
                    func.sum(Document.chunk_count),
                )
                .where(Document.patient_id == patient.id)
                .group_by(Document.document_type)
            )
        ).all()
        inventory = [
            {"document_type": document_type, "documents": count, "chunks": int(chunks or 0)}
            for document_type, count, chunks in rows
        ]

        record = {
            "external_id": patient.external_id,
            "display_name": patient.display_name,
            "birth_year": patient.birth_year,
            "sex": patient.sex,
            "is_synthetic": patient.is_synthetic,
            "documents": inventory,
        }
        lines = [
            f"Patient {patient.external_id}: {patient.display_name}",
            f"Year of birth {patient.birth_year}, sex {patient.sex}.",
            *(
                f"{item['documents']} {item['document_type']} document(s), "
                f"{item['chunks']} indexed chunks."
                for item in inventory
            ),
        ]
        return ToolResult(
            summary=(
                f"Patient record for {patient.external_id}: "
                f"{sum(item['documents'] for item in inventory)} document(s)."
            ),
            evidence=[Evidence(text="\n".join(lines), source="patients", kind="database")],
            data=record,
            trace={"document_types": len(inventory)},
        )


def _retrieval_result(chunks: list[Any], trace: dict[str, Any], label: str) -> ToolResult:
    evidence = [Evidence.from_chunk(chunk) for chunk in chunks]
    return ToolResult(
        summary=f"Retrieved {len(evidence)} {label} chunk(s).",
        evidence=evidence,
        data={
            "chunks": [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "filename": chunk.document_filename,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                }
                for chunk in chunks
            ]
        },
        trace=trace,
    )


def _with_top_k(config: RetrievalConfig, top_k: int) -> RetrievalConfig:
    return replace(config, top_k=min(top_k, config.candidate_pool))
