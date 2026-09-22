"""Database engine and session management.

One async engine per process, created lazily at startup and disposed at
shutdown. Request handlers get a session through the `get_db_session`
dependency, which commits on success and rolls back on any exception, so no
handler has to remember transaction bookkeeping.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None

# A second, deliberately separate engine for the agent's SQL tool. It connects
# as a login with no write privilege, so a defect in the in-process SQL guard
# still cannot mutate anything. Kept apart from the application engine so the
# two can never be confused at a call site: the type is the same, the identity
# is not, and `get_readonly_session` is the only way to reach it.
_readonly_engine: AsyncEngine | None = None
_readonly_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def create_engine(settings: Settings) -> AsyncEngine:
    url = make_url(settings.database_url)
    kwargs: dict[str, Any] = {
        "echo": settings.database_echo,
        "pool_pre_ping": True,  # survive server restarts / idle connection drops
        "future": True,
    }
    if url.get_backend_name() == "sqlite":
        # The unit-test suite runs on in-memory SQLite. Every connection would
        # otherwise get its own empty database, so pin the whole engine to one.
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        kwargs["pool_recycle"] = 1800
    return create_async_engine(settings.database_url, **kwargs)


def init_engine(settings: Settings) -> AsyncEngine:
    """Create the process-wide engine. Called once from the app lifespan."""
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_engine(settings)
        _sessionmaker = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised; call init_engine() first.")
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("Database engine not initialised; call init_engine() first.")
    return _sessionmaker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one transaction per request."""
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


# --- the agent's read-only connection ------------------------------------


def init_readonly_engine(settings: Settings) -> AsyncEngine:
    """Create the engine the SQL tool runs on.

    Sized small on purpose. Analytical queries written by a language model are
    the least predictable work this service does, so they get a pool that
    cannot starve the request path of connections.
    """
    global _readonly_engine, _readonly_sessionmaker
    if _readonly_engine is not None:
        return _readonly_engine

    if settings.readonly_database_url is None:
        # No separate login configured - local and CI, where SQLite has no
        # roles to separate. Reuse the application engine rather than opening a
        # second one: two engines on `sqlite:///:memory:` are two different
        # empty databases, and a read-only path that cannot see the data is
        # worse than one that is honest about sharing a connection. The
        # in-process guard still runs, the session is still never committed,
        # and `validate_production_safety` refuses this outside local/CI.
        _readonly_engine = init_engine(settings)
        _readonly_sessionmaker = get_sessionmaker()
        logger.warning(
            "readonly_engine_shares_application_login",
            reason="READONLY_DATABASE_URL is not set",
        )
        return _readonly_engine

    url = make_url(settings.sql_session_url)
    kwargs: dict[str, Any] = {"echo": settings.database_echo, "pool_pre_ping": True}
    if url.get_backend_name() == "sqlite":
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 2
        kwargs["max_overflow"] = 2
        kwargs["pool_recycle"] = 1800

    _readonly_engine = create_async_engine(settings.sql_session_url, **kwargs)
    _apply_query_timeout(_readonly_engine, settings.readonly_query_timeout_seconds)
    _readonly_sessionmaker = async_sessionmaker(
        _readonly_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    logger.info(
        "readonly_engine_ready",
        backend=url.get_backend_name(),
        # Whether this is genuinely a separate, unprivileged login. In local
        # and CI it often is not, and that is worth seeing in the log rather
        # than assuming.
        separate_login=settings.readonly_database_url is not None,
        timeout_seconds=settings.readonly_query_timeout_seconds,
    )
    return _readonly_engine


async def dispose_readonly_engine() -> None:
    global _readonly_engine, _readonly_sessionmaker
    # Only dispose an engine this module owns; when it is the application
    # engine, `dispose_engine` is responsible for it.
    if _readonly_engine is not None and _readonly_engine is not _engine:
        await _readonly_engine.dispose()
    _readonly_engine = None
    _readonly_sessionmaker = None


def get_readonly_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _readonly_sessionmaker is None:
        raise RuntimeError("Read-only engine not initialised; call init_readonly_engine().")
    return _readonly_sessionmaker


def readonly_engine_is_shared() -> bool:
    """True when the read-only engine is the application engine.

    Callers need to know, because a session opened on a *shared* engine is not
    independent: on SQLite's single pooled connection it sits in the caller's
    transaction, so rolling it back would discard the caller's work.
    """
    return _readonly_engine is not None and _readonly_engine is _engine


@asynccontextmanager
async def readonly_session() -> AsyncIterator[AsyncSession]:
    """A session that is rolled back unconditionally.

    Never committed, even on success: the tool has nothing to commit, and a
    transaction that cannot commit is one more thing that has to go wrong
    before a write could land.
    """
    async with get_readonly_sessionmaker()() as session:
        try:
            yield session
        finally:
            await session.rollback()


def _apply_query_timeout(engine: AsyncEngine, seconds: int) -> None:
    """Ask the driver to cancel statements that run too long.

    SQL Server has no per-login statement timeout, so unlike the read-only
    privilege this cannot be pinned to the login - it is set on each connection
    as it is opened. Not every driver honours it, so the executor *also*
    bounds the await; this is the half that stops the server still working on
    a query nobody is waiting for. If the driver refuses, say so rather than
    leave a control that looks enforced and is not.
    """
    if seconds <= 0:
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        candidates = (
            getattr(dbapi_connection, "driver_connection", None),
            getattr(getattr(dbapi_connection, "driver_connection", None), "_conn", None),
            dbapi_connection,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                candidate.timeout = seconds
            except Exception:  # noqa: BLE001, S112 - probe for driver support
                continue
            return
        logger.info("readonly_query_timeout_not_set_by_driver", seconds=seconds)
