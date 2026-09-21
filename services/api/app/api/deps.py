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
from app.models.enums import Role
from app.models.user import PatientAccessGrant, User
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


def client_ip(request: Request) -> str | None:
    """Best-effort client IP for audit rows (trusts the first XFF hop)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None
