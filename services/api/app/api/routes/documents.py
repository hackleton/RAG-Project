"""Document ingestion endpoints.

Contract is fixed in Sprint 1 so the web app and the evaluation harness can be
written against it; the pipeline behind it lands in Sprint 2 (blob upload ->
Document Intelligence -> normalisation -> chunking -> metadata).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import CurrentUser, DbSession, require_roles
from app.core.errors import NotImplementedYetError
from app.models.enums import Role
from app.security.principal import Principal

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a synthetic healthcare document for ingestion",
)
async def upload_document(
    session: DbSession,
    file: UploadFile = File(...),
    patient_external_id: str | None = Form(default=None),
    document_type: str = Form(default="clinical_note"),
    principal: Principal = Depends(require_roles(Role.CLINICIAN, Role.ADMIN)),
) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 2 - document pipeline")


@router.get("/{document_id}", summary="Document metadata and ingestion status")
async def get_document(
    document_id: uuid.UUID, session: DbSession, principal: CurrentUser
) -> dict[str, str]:
    raise NotImplementedYetError("Sprint 2 - document pipeline")
