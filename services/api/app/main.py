"""Application factory and ASGI entrypoint.

`create_app()` takes an optional Settings so tests can build an app against a
throwaway database without mutating process state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.middleware import RequestContextMiddleware
from app.api.router import api_router, root_router
from app.core.config import Settings, get_settings
from app.core.db import dispose_engine, init_engine
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Grounded question answering over synthetic healthcare documents and structured
patient data.

* Retrieved content is treated as untrusted data. Instructions embedded in a
  document are never executed.
* Access is enforced by application controls (role, patient scope, document
  sensitivity), not by prompt instructions.
* Synthetic data only. No real patient data belongs in this system.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        init_engine(settings)
        logger.info(
            "api_startup",
            version=__version__,
            environment=settings.app_env.value,
            api_prefix=settings.api_prefix,
        )
        try:
            yield
        finally:
            await dispose_engine()
            logger.info("api_shutdown")

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # Dependencies read the settings from here (see app.api.deps.get_app_settings).
    app.state.settings = settings

    # Order matters: the request-context middleware must wrap everything so the
    # correlation id is set before any other layer can log or fail.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(root_router)
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


app = create_app()
