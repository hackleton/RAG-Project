"""Database engine and session management.

One async engine per process, created lazily at startup and disposed at
shutdown. Request handlers get a session through the `get_db_session`
dependency, which commits on success and rolls back on any exception, so no
handler has to remember transaction bookkeeping.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import Settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def create_engine(settings: Settings) -> AsyncEngine:
    url = make_url(settings.database_url)
    kwargs: dict[str, Any] = {
        "echo": settings.database_echo,
        "pool_pre_ping": True,  # survive Postgres restarts / idle connection drops
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
