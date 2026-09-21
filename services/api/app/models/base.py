"""Declarative base, shared column types and mixins.

Two conventions that matter on SQL Server:

* **Unicode by default.** `String`/`Text` map to `VARCHAR`/`VARCHAR(max)` on
  SQL Server, which cannot hold characters outside the database collation's
  code page. Clinical text, patient names and filenames are `NVARCHAR`
  (`Unicode`/`UnicodeText`) instead. Only machine-generated ASCII values -
  enum members, checksums, correlation ids - use `String`.
* **Named constraints.** Without the naming convention Alembic emits unnamed
  constraints, which a downgrade then cannot drop by name.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, Uuid

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# NVARCHAR(max) on SQL Server, TEXT on SQLite. SQL Server 2022 has no native
# JSON type; 2025 does, and moving to it is a type swap here plus a migration.
JSONType = JSON()


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict: JSONType}


class UUIDPrimaryKeyMixin:
    """UUID keys: ids are generated client-side and are safe to expose in URLs.

    Deliberately not an IDENTITY column. Sequential integer keys would let a
    caller enumerate other patients' documents by incrementing a URL, and the
    ingestion pipeline needs an id before the row is inserted so that storage
    paths and vector-index records can be keyed on it.

    The cost on SQL Server is a random clustered key, which fragments the index
    on insert. At this scale that is irrelevant; if it ever is not, the answer
    is a non-clustered primary key with a clustered index on `created_at`.
    """

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    # func.now() compiles to CURRENT_TIMESTAMP on both SQL Server and SQLite.
    # The containers run with TZ=UTC so the stored value is UTC.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
