"""Document upload and retrieval, with access control applied.

Every read path here filters by the caller's `Principal` rather than trusting an
id in the URL: a document id is a UUID, but "unguessable" is not an access
control. A document the caller may not see returns 404, not 403 - a 403 would
confirm that the id exists, which for a patient-linked document is itself
disclosure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.documents.pipeline import DocumentPipeline, IngestionResult
from app.models.document import Document, DocumentChunk
from app.models.enums import AccessLevel, AuditAction, DocumentStatus
from app.models.patient import Patient
from app.providers.extraction import SUPPORTED_EXTENSIONS
from app.providers.storage import DocumentStore, checksum
from app.security.audit import record_audit
from app.security.principal import Principal
from app.security.rbac import assert_patient_access

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UploadRequest:
    filename: str
    content_type: str
    data: bytes
    patient_external_id: str | None
    document_type: str
    access_level: AccessLevel


class DocumentService:
    def __init__(
        self,
        *,
        store: DocumentStore,
        pipeline: DocumentPipeline,
        max_upload_bytes: int,
    ) -> None:
        self._store = store
        self._pipeline = pipeline
        self._max_upload_bytes = max_upload_bytes

    # --- write ------------------------------------------------------------

    async def upload(
        self,
        session: AsyncSession,
        principal: Principal,
        request: UploadRequest,
        *,
        ip_address: str | None = None,
    ) -> tuple[Document, IngestionResult | None]:
        self._validate(request)
        patient = await self._resolve_patient(session, principal, request.patient_external_id)

        # A restricted document with no patient cannot be scoped to anyone, and
        # a patient document that is not restricted defeats patient filtering.
        access_level = request.access_level
        if patient is not None and access_level is not AccessLevel.RESTRICTED:
            raise ValidationError(
                "A patient-linked document must be uploaded as 'restricted'.",
                details={"access_level": access_level.value},
            )

        digest = checksum(request.data)
        if existing := await self._find_duplicate(session, digest, patient):
            # Same bytes, same patient: return the original rather than paying
            # for extraction again and creating a second citable copy.
            logger.info(
                "document_upload_deduplicated",
                document_id=str(existing.id),
                checksum=digest[:12],
            )
            await record_audit(
                session,
                action=AuditAction.DOCUMENT_UPLOADED,
                principal=principal,
                resource_type="document",
                resource_id=existing.id,
                allowed=True,
                ip_address=ip_address,
                details={"deduplicated": True, "checksum_prefix": digest[:12]},
            )
            return existing, None

        document = Document(
            id=uuid.uuid4(),
            patient_id=patient.id if patient else None,
            filename=request.filename,
            content_type=request.content_type or "application/octet-stream",
            size_bytes=len(request.data),
            checksum_sha256=digest,
            document_type=request.document_type,
            access_level=access_level,
            status=DocumentStatus.UPLOADED,
            uploaded_by_id=principal.id,
        )
        session.add(document)
        await session.flush()

        stored = await self._store.put(
            document_id=document.id,
            patient_id=document.patient_id,
            filename=document.filename,
            data=request.data,
        )
        document.blob_uri = stored.uri
        document.status = DocumentStatus.EXTRACTING
        await session.flush()

        result = await self._pipeline.run(session, document, request.data)

        await record_audit(
            session,
            action=AuditAction.DOCUMENT_UPLOADED,
            principal=principal,
            resource_type="document",
            resource_id=document.id,
            allowed=True,
            ip_address=ip_address,
            details={
                "document_type": document.document_type,
                "access_level": access_level.value,
                "size_bytes": document.size_bytes,
                "status": result.status.value,
                "chunk_count": result.chunk_count,
                # Never the document text itself (project plan section 15).
                "checksum_prefix": digest[:12],
            },
        )
        return document, result

    # --- read -------------------------------------------------------------

    async def get(
        self, session: AsyncSession, principal: Principal, document_id: uuid.UUID
    ) -> Document:
        document = await session.scalar(self._visible(principal).where(Document.id == document_id))
        if document is None:
            # Deliberately 404 rather than 403 - see the module docstring.
            raise NotFoundError("Document not found.")
        return document

    async def list_documents(
        self,
        session: AsyncSession,
        principal: Principal,
        *,
        patient_external_id: str | None = None,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        query = self._visible(principal)
        if status is not None:
            query = query.where(Document.status == status)
        if patient_external_id is not None:
            patient = await session.scalar(
                select(Patient).where(Patient.external_id == patient_external_id)
            )
            if patient is None:
                return [], 0
            assert_patient_access(principal, patient.id)
            query = query.where(Document.patient_id == patient.id)

        total = await session.scalar(
            select(func.count()).select_from(query.order_by(None).subquery())
        )
        documents = await session.scalars(
            query.order_by(Document.created_at.desc()).offset(offset).limit(limit)
        )
        return list(documents), int(total or 0)

    async def chunks(
        self, session: AsyncSession, principal: Principal, document_id: uuid.UUID
    ) -> list[DocumentChunk]:
        # Re-checks visibility, so the chunk endpoint cannot be used to read a
        # document the caller could not fetch directly.
        await self.get(session, principal, document_id)
        rows = await session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(rows)

    # --- internals --------------------------------------------------------

    def _visible(self, principal: Principal) -> Select[tuple[Document]]:
        """The base query, already narrowed to what this caller may see."""
        query = select(Document)

        levels = principal.visible_access_levels
        query = query.where(Document.access_level.in_([level.value for level in levels]))

        if principal.accessible_patient_ids is not None:
            # Documents with no patient are non-patient material (guidelines,
            # drug references) and are not patient-scoped.
            query = query.where(
                Document.patient_id.is_(None)
                | Document.patient_id.in_(principal.accessible_patient_ids)
            )
        return query

    def _validate(self, request: UploadRequest) -> None:
        if not request.data:
            raise ValidationError("The uploaded file is empty.")
        if len(request.data) > self._max_upload_bytes:
            raise ValidationError(
                "The file is larger than the upload limit.",
                details={
                    "size_bytes": len(request.data),
                    "limit_bytes": self._max_upload_bytes,
                },
            )
        suffix = "." + request.filename.rsplit(".", maxsplit=1)[-1].lower()
        if "." not in request.filename or suffix not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                "Unsupported file type.",
                details={"supported": sorted(SUPPORTED_EXTENSIONS)},
            )

    async def _resolve_patient(
        self, session: AsyncSession, principal: Principal, external_id: str | None
    ) -> Patient | None:
        if not external_id:
            return None
        patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
        if patient is None:
            raise NotFoundError("Unknown patient.", details={"patient_external_id": external_id})
        # Uploading *to* a patient is as sensitive as reading from one.
        assert_patient_access(principal, patient.id)
        return patient

    async def _find_duplicate(
        self, session: AsyncSession, digest: str, patient: Patient | None
    ) -> Document | None:
        query = select(Document).where(Document.checksum_sha256 == digest)
        query = (
            query.where(Document.patient_id == patient.id)
            if patient is not None
            else query.where(Document.patient_id.is_(None))
        )
        return await session.scalar(query.limit(1))
