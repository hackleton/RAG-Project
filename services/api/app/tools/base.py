"""What a tool is.

Every tool returns the same three things, whether it read a document index or
a relational table:

* `summary` - one line, safe to log and to put in an audit row. Identifiers,
  counts and outcomes only, never values (project plan section 15).
* `evidence` - the material the final answer may be grounded in, each piece
  carrying enough provenance to be cited and traced back.
* `data` - a JSON-safe structured payload for the caller, so the frontend can
  render a lab table rather than a paragraph about one.

Unifying document text and database rows behind one `Evidence` type is what
lets synthesis cite both in a single numbered list: "[2]" can be a chunk of a
discharge summary or eight rows of `lab_results`, and the citation is
verifiable either way.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.db import readonly_engine_is_shared, readonly_session
from app.rag.retriever import RetrievedChunk
from app.security.principal import Principal

EvidenceKind = Literal["document", "database"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """One citable piece of material."""

    text: str
    # Human-readable origin: a filename, or a table name for structured data.
    source: str
    kind: EvidenceKind
    chunk_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    section: str | None = None
    page_number: int | None = None
    score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> Evidence:
        return cls(
            text=chunk.text,
            source=chunk.document_filename,
            kind="document",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section=chunk.section,
            page_number=chunk.page_number,
            score=chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score,
        )


@dataclass(slots=True)
class ToolResult:
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    # Per-tool detail for the run trace: candidate counts, row counts, the SQL
    # validation rules that fired. Never values from the data itself.
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to know about the caller.

    The principal is the authority for scope. A tool that needs to know which
    patients the caller may see reads it here, never from its own arguments -
    arguments can narrow a request, they can never widen it.
    """

    session: AsyncSession
    principal: Principal
    settings: Settings
    # Resolved and authorised by the dispatcher before the tool runs, when the
    # tool's arguments named a patient.
    patient_id: uuid.UUID | None = None
    patient_external_id: str | None = None

    @asynccontextmanager
    async def readonly(self) -> AsyncIterator[AsyncSession]:
        """A session for reading clinical data as the unprivileged login.

        When a separate read-only login is configured this is a genuinely
        independent session, rolled back unconditionally.

        When one is not - local, CI, SQLite, which has no roles to separate -
        the read-only engine *is* the application engine, and opening a second
        session would share its single pooled connection. Rolling that back
        would discard the caller's transaction along with it: the audit rows
        written earlier in the same request would vanish, which is precisely
        the record worth keeping. So the caller's own session is used instead,
        and the in-process SQL guard remains the control, exactly as it is
        documented to be.
        """
        if readonly_engine_is_shared():
            yield self.session
        else:
            async with readonly_session() as session:
                yield session


class Tool(Protocol):
    async def run(self, context: ToolContext, arguments: Any) -> ToolResult: ...


class ToolArguments(BaseModel):
    """Base for every tool's argument model.

    `extra="forbid"` is the point. A planner that invents an argument - or a
    document that talks one into it - gets a refusal, not a field quietly
    dropped on the floor while the tool runs with defaults the caller never
    asked for.
    """

    model_config = ConfigDict(extra="forbid")


class PatientScopedArguments(ToolArguments):
    """Argument models that name a patient.

    The dispatcher looks for `patient_external_id` and resolves it once, so
    patient authorisation is not something an individual tool can forget to do.
    """

    patient_external_id: str | None = None
