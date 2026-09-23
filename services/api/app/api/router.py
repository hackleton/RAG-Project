"""API router assembly.

Health probes are mounted at the root so orchestrators do not have to track the
versioned prefix; everything else lives under `settings.api_prefix`.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    agents,
    audit,
    auth,
    chat,
    documents,
    evaluations,
    health,
    search,
)

root_router = APIRouter()
root_router.include_router(health.router)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(search.router)
api_router.include_router(agents.router)
api_router.include_router(evaluations.router)
api_router.include_router(audit.router)
