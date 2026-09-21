"""Users and their access grants."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Unicode, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import Role

if TYPE_CHECKING:
    from app.models.patient import Patient


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(Unicode(200), nullable=False)
    role: Mapped[Role] = mapped_column(
        Enum(Role, native_enum=False, length=32, validate_strings=True), nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    patient_grants: Mapped[list[PatientAccessGrant]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email} role={self.role}>"


class PatientAccessGrant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Which patients a non-admin user may retrieve data for.

    Patient-level filtering (project plan section 11) is applied from this
    table before retrieval, not inferred by the model at generation time.
    """

    __tablename__ = "patient_access_grants"
    __table_args__ = (UniqueConstraint("user_id", "patient_id"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )

    user: Mapped[User] = relationship(back_populates="patient_grants")
    patient: Mapped[Patient] = relationship(back_populates="access_grants")
