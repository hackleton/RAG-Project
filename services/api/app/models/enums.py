"""Domain enumerations.

Stored as VARCHAR + CHECK constraint (`native_enum=False`) rather than a
Postgres ENUM type: adding a value later is a cheap constraint swap instead of
an `ALTER TYPE` that locks and cannot run inside every migration context.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Application roles, ordered least to most privileged."""

    ANALYST = "analyst"  # aggregate/cohort data only, no patient documents
    CLINICIAN = "clinician"  # patient documents they are granted access to
    ADMIN = "admin"  # full access plus ingestion and evaluation control


class AccessLevel(StrEnum):
    """Sensitivity of a document/chunk, enforced at retrieval time."""

    PUBLIC = "public"  # clinical guidelines, drug references
    INTERNAL = "internal"  # de-identified operational documents
    RESTRICTED = "restricted"  # patient-linked documents


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class AuditAction(StrEnum):
    LOGIN_SUCCEEDED = "login_succeeded"
    LOGIN_FAILED = "login_failed"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_ACCESSED = "document_accessed"
    TOOL_CALLED = "tool_called"
    TOOL_DENIED = "tool_denied"
    SQL_EXECUTED = "sql_executed"
    SQL_REJECTED = "sql_rejected"
    REPORT_GENERATED = "report_generated"
    INJECTION_BLOCKED = "injection_blocked"


# Roles that may see each access level. Consulted by the retrieval filter
# (Sprint 6) and by document download authorisation.
ACCESS_LEVEL_ROLES: dict[AccessLevel, frozenset[Role]] = {
    AccessLevel.PUBLIC: frozenset({Role.ANALYST, Role.CLINICIAN, Role.ADMIN}),
    AccessLevel.INTERNAL: frozenset({Role.ANALYST, Role.CLINICIAN, Role.ADMIN}),
    AccessLevel.RESTRICTED: frozenset({Role.CLINICIAN, Role.ADMIN}),
}
