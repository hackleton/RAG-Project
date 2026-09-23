"""ORM models.

Importing this package registers every table on `Base.metadata`, which is what
Alembic autogenerate and the test schema bootstrap rely on. Keep the re-export
list complete.
"""

from app.models.audit import AuditLog
from app.models.base import Base
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, DocumentChunk
from app.models.enums import (
    ACCESS_LEVEL_ROLES,
    AccessLevel,
    AuditAction,
    ChatRole,
    DocumentStatus,
    Role,
)
from app.models.evaluation import EvaluationCase, EvaluationResult, EvaluationRun
from app.models.patient import LabResult, Patient
from app.models.token import RevokedToken
from app.models.user import PatientAccessGrant, User

__all__ = [
    "ACCESS_LEVEL_ROLES",
    "AccessLevel",
    "AuditAction",
    "AuditLog",
    "Base",
    "ChatMessage",
    "ChatRole",
    "ChatSession",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "EvaluationCase",
    "EvaluationResult",
    "EvaluationRun",
    "LabResult",
    "Patient",
    "PatientAccessGrant",
    "RevokedToken",
    "Role",
    "User",
]
