"""Synthetic patients and their structured clinical data.

Project plan section 2: no real patient data ever lands in this system. The
`is_synthetic` flag is asserted on write so a seeding mistake is visible.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Unicode,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.user import PatientAccessGrant


class Patient(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "patients"

    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(Unicode(200), nullable=False)
    birth_year: Mapped[int | None] = mapped_column()
    sex: Mapped[str | None] = mapped_column(String(16))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (CheckConstraint("is_synthetic = 1", name="must_be_synthetic"),)

    documents: Mapped[list[Document]] = relationship(back_populates="patient")
    lab_results: Mapped[list[LabResult]] = relationship(back_populates="patient")
    access_grants: Mapped[list[PatientAccessGrant]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class LabResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Structured measurements - the SQL tool's data source."""

    __tablename__ = "lab_results"
    __table_args__ = (
        # The SQL tool's dominant access pattern: one patient, one test, over time.
        Index("ix_lab_results_patient_test_time", "patient_id", "test_name", "collected_at"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    test_name: Mapped[str] = mapped_column(Unicode(128), index=True, nullable=False)
    test_code: Mapped[str | None] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(Unicode(32), nullable=False)
    reference_low: Mapped[float | None] = mapped_column(Float)
    reference_high: Mapped[float | None] = mapped_column(Float)
    abnormal_flag: Mapped[str | None] = mapped_column(String(8))
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="lab_results")
