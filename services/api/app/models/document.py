"""Documents and their retrievable chunks.

Every chunk keeps full provenance (project plan section 7) so an answer can
cite document, page and section, and so retrieval can be filtered by patient
and access level before the LLM ever sees the text.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Unicode,
    UnicodeText,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import AccessLevel, DocumentStatus

if TYPE_CHECKING:
    from app.models.patient import Patient


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_patient_status", "patient_id", "status"),)

    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(Unicode(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Deduplicates re-uploads and pins the exact bytes an answer was grounded in.
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    document_type: Mapped[str] = mapped_column(Unicode(64), nullable=False)
    blob_uri: Mapped[str | None] = mapped_column(String(1024))
    access_level: Mapped[AccessLevel] = mapped_column(
        Enum(AccessLevel, native_enum=False, length=32, validate_strings=True),
        default=AccessLevel.RESTRICTED,  # fail closed
        nullable=False,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False, length=32, validate_strings=True),
        default=DocumentStatus.UPLOADED,
        nullable=False,
        index=True,
    )
    status_detail: Mapped[str | None] = mapped_column(UnicodeText)
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Chunks carrying instruction-like text, counted at ingestion. A column
    # rather than a JSON field because the question it answers - "which
    # documents in this corpus carry planted instructions?" - is a query, and
    # it is the first one asked after an incident.
    injection_flag_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    uploaded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    patient: Mapped[Patient | None] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The unit of retrieval.

    Chunk text is stored here as the system of record; the vector index holds
    a copy keyed by `id`. Keeping the text in the database means the index can
    be rebuilt from scratch without re-running extraction, which is exactly what
    the chunk-size experiments in section 14 require.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index("ix_document_chunks_patient_access", "patient_id", "access_level"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # Denormalised from documents so the retrieval filter is a single-table read.
    patient_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), index=True)
    access_level: Mapped[AccessLevel] = mapped_column(
        Enum(AccessLevel, native_enum=False, length=32, validate_strings=True),
        default=AccessLevel.RESTRICTED,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Unicode(512))
    text: Mapped[str] = mapped_column(UnicodeText, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    # Points at the vector-index record; null until the chunk is indexed.
    embedding_reference: Mapped[str | None] = mapped_column(String(256), index=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")
