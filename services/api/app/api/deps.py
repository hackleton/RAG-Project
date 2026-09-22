"""Shared FastAPI dependencies.

`get_current_principal` is the single place a request turns into an identity.
It resolves the caller's patient scope once per request so that retrieval
filters, tool authorisation and audit records all read the same scope.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_db_session
from app.core.errors import AuthenticationError
from app.core.logging import user_id_ctx
from app.core.security import decode_access_token
from app.documents.chunking import ChunkingConfig
from app.documents.pipeline import DocumentPipeline
from app.documents.service import DocumentService
from app.models.enums import Role
from app.models.user import PatientAccessGrant, User
from app.providers.embedding import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.providers.storage import LocalFileStore
from app.providers.vectorstore import VectorStore
from app.rag.indexer import Indexer
from app.rag.retriever import HybridRetriever, RetrievalConfig
from app.rag.service import RagService
from app.security.principal import Principal
from app.security.rbac import assert_role

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{get_settings().api_prefix}/auth/token", auto_error=False
)


def get_app_settings(request: Request) -> Settings:
    """The settings this app was constructed with.

    Reading `request.app.state` rather than the cached `get_settings()`
    singleton keeps the app self-contained: the test suite builds an app with
    throwaway settings without mutating process-wide state.
    """
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_principal(
    request: Request,
    session: DbSession,
    settings: SettingsDep,
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> Principal:
    if not token:
        raise AuthenticationError("Missing bearer token.")

    payload = decode_access_token(token, settings)
    user_id = payload["sub"]

    user = await session.scalar(select(User).where(User.id == _as_uuid(user_id)))
    if user is None or not user.is_active:
        # A token can outlive the account it was issued for.
        raise AuthenticationError("User account is not active.")

    # The role in the token is a hint only; the database is authoritative, so a
    # role downgrade takes effect immediately rather than at token expiry.
    if user.role is Role.ADMIN:
        accessible: frozenset[Any] | None = None
    else:
        grants = await session.scalars(
            select(PatientAccessGrant.patient_id).where(PatientAccessGrant.user_id == user.id)
        )
        accessible = frozenset(grants)

    principal = Principal(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        accessible_patient_ids=accessible,
        request_id=getattr(request.state, "request_id", None),
    )
    user_id_ctx.set(str(user.id))
    return principal


CurrentUser = Annotated[Principal, Depends(get_current_principal)]


def require_roles(*allowed: Role) -> Callable[..., Coroutine[Any, Any, Principal]]:
    """Dependency factory: `principal: Principal = Depends(require_roles(Role.ADMIN))`."""

    async def _dependency(principal: CurrentUser) -> Principal:
        assert_role(principal, allowed)
        return principal

    return _dependency


def _as_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise AuthenticationError("Malformed token subject.") from exc


# The embedding model, reranker and Qdrant client are expensive to construct -
# a model load is ~1s of CPU and a client owns a connection pool - so they live
# on the app, built once in the lifespan, not per request. Everything cheap
# (services, configs) is still built per request from them.


def get_embedder(request: Request) -> EmbeddingProvider:
    provider: EmbeddingProvider = request.app.state.embedder
    return provider


def get_vector_store(request: Request) -> VectorStore:
    store: VectorStore = request.app.state.vector_store
    return store


def get_llm(request: Request) -> LLMProvider:
    llm: LLMProvider = request.app.state.llm
    return llm


EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]


def get_retrieval_config(settings: SettingsDep) -> RetrievalConfig:
    """The deployed retrieval defaults. A request may narrow them."""
    return RetrievalConfig(
        top_k=settings.retrieval_top_k,
        candidate_pool=settings.retrieval_candidate_pool,
        rrf_k=settings.rrf_k,
        mode=settings.retrieval_mode,
        rerank=settings.enable_reranker,
        rerank_top_n=settings.rerank_top_n,
        rerank_score_threshold=settings.rerank_score_threshold,
    )


RetrievalConfigDep = Annotated[RetrievalConfig, Depends(get_retrieval_config)]


def get_indexer(store: VectorStoreDep, embedder: EmbedderDep) -> Indexer:
    return Indexer(store=store, embedder=embedder)


def get_document_service(
    settings: SettingsDep, indexer: Annotated[Indexer, Depends(get_indexer)]
) -> DocumentService:
    """Wire the document pipeline from settings.

    Built per request rather than as a module-level singleton so an app
    constructed with different settings - the test suite, or an experiment run
    sweeping chunk sizes - gets its own configuration. The costly collaborators
    come from app state; the rest is cheap to assemble.
    """
    return DocumentService(
        store=LocalFileStore(settings.storage_dir),
        pipeline=DocumentPipeline(
            chunking=ChunkingConfig(
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            ),
            indexer=indexer,
        ),
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_rag_service(
    request: Request,
    store: VectorStoreDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    config: RetrievalConfigDep,
) -> RagService:
    return RagService(
        retriever=HybridRetriever(
            store=store,
            embedder=embedder,
            reranker=request.app.state.reranker,
            config=config,
        ),
        llm=llm,
    )


RagServiceDep = Annotated[RagService, Depends(get_rag_service)]


def client_ip(request: Request) -> str | None:
    """Best-effort client IP for audit rows (trusts the first XFF hop)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None
