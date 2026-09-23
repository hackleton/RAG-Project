"""Document API payloads."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import AccessLevel, DocumentStatus


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID | None
    filename: str
    content_type: str
    size_bytes: int
    document_type: str
    access_level: AccessLevel
    status: DocumentStatus
    status_detail: str | None
    page_count: int | None
    chunk_count: int
    # Chunks carrying instruction-like text, counted at ingestion. Surfaced so
    # the corpus view answers "which documents were tampered with" without a
    # database session (project plan section 12).
    injection_flag_count: int
    checksum_sha256: str
    created_at: datetime
    indexed_at: datetime | None


class UploadResponse(BaseModel):
    document: DocumentRead
    #  True when identical bytes for this patient were already ingested and the
    #  existing document was returned instead of a second copy.
    deduplicated: bool = False
    warnings: list[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int
    limit: int
    offset: int


class ChunkRead(BaseModel):
    """A chunk with the provenance required by project plan section 7."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    patient_id: uuid.UUID | None
    chunk_index: int
    page_number: int | None
    section: str | None
    text: str
    token_count: int | None
    access_level: AccessLevel
    embedding_reference: str | None
    chunk_metadata: dict[str, Any]


class ChunkListResponse(BaseModel):
    document_id: uuid.UUID
    items: list[ChunkRead]
    total: int
