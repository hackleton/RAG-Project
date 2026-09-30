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
from app.cache.backend import build_cache
from app.cache.embeddings import CachingEmbeddingProvider
from app.cache.ranking import RankingCache
from app.core.config import LLMProvider as LLMProviderName
from app.core.config import Settings, get_settings
from app.core.db import (
    dispose_engine,
    dispose_readonly_engine,
    init_engine,
    init_readonly_engine,
)
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.observability.instruments import MeasuredLLM
from app.providers.embedding import build_embedding_provider
from app.providers.llm import build_llm_provider
from app.providers.reranker import build_reranker
from app.providers.vectorstore import QdrantVectorStore

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
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        init_engine(settings)
        # The agent's SQL tool connects as an unprivileged login. Built at
        # start-up so a misconfigured read-only URL fails the boot rather than
        # the first analytical question somebody asks.
        init_readonly_engine(settings)

        # Built once per process. The embedding model and reranker load their
        # weights lazily on first use, so start-up stays fast and a missing
        # model surfaces on the first query rather than blocking the boot.
        #
        # The cache is constructed before the things that use it, and never
        # connects here: a Redis that is down must delay nothing at boot, and
        # a deployment that cannot reach it has to keep serving. `build_cache`
        # therefore only builds a client; the first request finds out.
        application.state.cache = build_cache(settings)

        embedder = build_embedding_provider(settings)
        if settings.cache_enabled and settings.cache_embeddings:
            embedder = CachingEmbeddingProvider(
                embedder,
                application.state.cache,
                prefix=settings.cache_prefix,
                ttl_seconds=settings.cache_embedding_ttl_seconds,
            )
        application.state.embedder = embedder
        application.state.ranking_cache = (
            RankingCache(
                application.state.cache,
                prefix=settings.cache_prefix,
                ttl_seconds=settings.cache_ranking_ttl_seconds,
            )
            if settings.cache_enabled and settings.cache_rankings
            else None
        )

        application.state.reranker = build_reranker(settings)
        application.state.vector_store = QdrantVectorStore(settings)
        # Wrapped so every completion is timed and counted whichever caller
        # made it - chat, the planner, synthesis or the evaluation judge.
        application.state.llm = MeasuredLLM(build_llm_provider(settings))

        logger.info(
            "api_startup",
            version=__version__,
            environment=settings.app_env.value,
            api_prefix=settings.api_prefix,
            embedding_provider=settings.embedding_provider.value,
            llm_provider=settings.llm_provider.value,
            llm_model=settings.llm_model,
            retrieval_mode=settings.retrieval_mode,
            planner=settings.effective_planner.value,
            sql_tool_enabled=settings.sql_tool_enabled,
            cache_backend=settings.cache_backend.value,
        )
        if not settings.is_local and not settings.shared_cache:
            # Not fatal - the system is correct without a shared cache - but
            # the failed-login limiter silently becomes per-replica, which is
            # the weakness decision 38 recorded and Sprint 7 came to fix.
            logger.warning(
                "cache_not_shared",
                backend=settings.cache_backend.value,
                consequence="login throttling and ranking reuse are per-process",
            )
        if settings.llm_provider is LLMProviderName.OPENROUTER:
            # Said once, at boot, where an operator reads it: this deployment
            # sends prompts - questions and retrieved document text - off the
            # machine. The corpus is synthetic by rule; see decision 48.
            logger.info(
                "llm_prompts_leave_host",
                provider="openrouter",
                model=settings.llm_model,
                stays_local="embeddings, reranking, storage",
            )
        try:
            yield
        finally:
            await application.state.llm.aclose()
            await application.state.vector_store.close()
            await application.state.cache.close()
            await dispose_readonly_engine()
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
