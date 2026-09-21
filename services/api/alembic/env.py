"""Alembic environment.

Runs against the same async engine the application uses, so there is no second
database driver to keep in sync. The URL always comes from the environment via
`app.core.config` - `alembic.ini` deliberately carries no credentials.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from alembic import context
from app.core.config import get_settings
from app.models import Base  # noqa: F401 - registers every table on the metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


# SQL Server reflects a CURRENT_TIMESTAMP default back as "(getdate())", which
# never string-matches what the metadata renders. Rather than switching server
# default comparison off wholesale - and losing a real check on every other
# column - treat the known-equivalent "now" spellings as equal and defer
# everything else to Alembic's own comparison.
_NOW_DEFAULTS = frozenset(
    {"current_timestamp", "getdate()", "sysutcdatetime()", "sysdatetimeoffset()"}
)


def _normalise_default(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def _compare_server_default(
    _context: object,
    _inspected_column: object,
    _metadata_column: object,
    inspected_default: object,
    _metadata_default: object,
    rendered_metadata_default: object,
) -> bool | None:
    """True = differs, False = same, None = let Alembic decide."""
    if (
        _normalise_default(inspected_default) in _NOW_DEFAULTS
        and _normalise_default(rendered_metadata_default) in _NOW_DEFAULTS
    ):
        return False
    return None


def _configure(connection: object) -> None:
    context.configure(
        connection=connection,  # type: ignore[arg-type]
        target_metadata=target_metadata,
        compare_type=True,  # catch NVARCHAR length / numeric precision drift
        compare_server_default=_compare_server_default,
        render_as_batch=False,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a DBAPI connection (`alembic upgrade --sql`)."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: object) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,  # a migration process should not hold a pool
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_run_sync)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
