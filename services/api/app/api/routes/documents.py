"""Document ingestion and inspection.

Ingestion runs inline rather than on a queue. The upload is a few hundred
kilobytes of synthetic text and the work is local, so a request-scoped run keeps
the whole pipeline observable in one trace and one transaction - a failed
extraction rolls the document row back rather than leaving a dangling record.
`DocumentPipeline` is a standalone object taking a session, so moving it behind
a worker in Sprint 7 means enqueuing instead of awaiting. See decision 15.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status

from app.api.deps import CurrentUser, DbSession, client_ip, get_document_service, require_roles
from app.documents.service import DocumentService, UploadRequest
from app.models.enums import AccessLevel, DocumentStatus, Role
from app.schemas.documents import (
    ChunkListResponse,
    ChunkRead,
    DocumentListResponse,
    DocumentRead,
    UploadResponse,
)
from app.security.principal import Principal

router = APIRouter(prefix="/documents", tags=["documents"])

ServiceDep = Annotated[DocumentService, Depends(get_document_service)]


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a synthetic healthcare document and ingest it",
    responses={
        403: {"description": "The caller may not upload for this patient."},
        404: {"description": "Unknown patient."},
        422: {"description": "Empty file, unsupported type, or over the size limit."},
    },
)
async def upload_document(
    request: Request,
    session: DbSession,
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or MD.")],
    document_type: Annotated[str, Form()] = "clinical_note",
    patient_external_id: Annotated[str | None, Form(max_length=64)] = None,
    access_level: Annotated[AccessLevel, Form()] = AccessLevel.RESTRICTED,
    principal: Principal = Depends(require_roles(Role.CLINICIAN, Role.ADMIN)),
) -> UploadResponse:
    # Read once: the size check needs the real length, and the content-length
    # header is caller-supplied. The limit is enforced in the service.
    data = await file.read()

    document, result = await service.upload(
        session,
        principal,
        UploadRequest(
            filename=file.filename or "document",
            content_type=file.content_type or "",
            data=data,
            patient_external_id=patient_external_id,
            document_type=document_type,
            access_level=access_level,
        ),
        ip_address=client_ip(request),
    )
    return UploadResponse(
        document=DocumentRead.model_validate(document),
        deduplicated=result is None,
        warnings=result.warnings if result else [],
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents the caller may see",
)
async def list_documents(
    session: DbSession,
    service: ServiceDep,
    principal: CurrentUser,
    patient_external_id: Annotated[str | None, Query(max_length=64)] = None,
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    documents, total = await service.list_documents(
        session,
        principal,
        patient_external_id=patient_external_id,
        status=document_status,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        items=[DocumentRead.model_validate(document) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Document metadata and ingestion status",
    responses={404: {"description": "No such document, or not visible to the caller."}},
)
async def get_document(
    document_id: uuid.UUID,
    session: DbSession,
    service: ServiceDep,
    principal: CurrentUser,
) -> DocumentRead:
    document = await service.get(session, principal, document_id)
    return DocumentRead.model_validate(document)


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="The chunks a document produced, with their provenance",
    description=(
        "Inspection endpoint. Chunk boundaries, sections and page numbers are "
        "what the chunk-size experiments in the project plan compare, so they "
        "need to be visible without going through retrieval."
    ),
)
async def get_document_chunks(
    document_id: uuid.UUID,
    session: DbSession,
    service: ServiceDep,
    principal: CurrentUser,
) -> ChunkListResponse:
    chunks = await service.chunks(session, principal, document_id)
    return ChunkListResponse(
        document_id=document_id,
        items=[ChunkRead.model_validate(chunk) for chunk in chunks],
        total=len(chunks),
    )
