"""Token revocation, and the cookie the browser app authenticates with.

Revocation is a deny-list keyed on the JWT's `jti`. The alternative - a session
table the token points at - turns every request into a session lookup that must
succeed, where this one turns it into a lookup that is expected to miss. The
difference matters on the failure path: a deny-list that is briefly unreachable
fails *open* for tokens nobody revoked, while a session table that is briefly
unreachable logs everybody out.

That trade is stated rather than hidden: a revoked token stays usable for as
long as a replica cannot read the table. With a one-hour token lifetime and a
single local database, that window is the same outage that takes the rest of
the API down with it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from app.core.config import Settings
from app.models.token import RevokedToken


async def revoke(
    session: AsyncSession, *, jti: str, user_id: uuid.UUID | None, expires_at: datetime
) -> None:
    """Add a token to the deny-list. Revoking twice is not an error."""
    already = await session.scalar(select(RevokedToken).where(RevokedToken.jti == jti))
    if already is not None:
        return
    session.add(RevokedToken(jti=jti, user_id=user_id, expires_at=expires_at))


async def is_revoked(session: AsyncSession, jti: str | None) -> bool:
    """Whether this token has been given up.

    A token with no `jti` cannot be revoked and is treated as not revoked -
    every token this application issues has one, so the only way to arrive
    without it is a token minted before Sprint 1's claim set, which the
    signature check has already had its say about.
    """
    if not jti:
        return False
    return await session.scalar(select(RevokedToken.id).where(RevokedToken.jti == jti)) is not None


async def purge_expired(session: AsyncSession, *, now: datetime | None = None) -> int:
    """Drop rows for tokens that would have expired anyway.

    Called on logout rather than on a schedule: the table only grows when
    somebody logs out, so that is exactly when it is worth trimming, and it
    saves the deployment a cron job it would otherwise have to own.
    """
    cutoff = now or datetime.now(UTC)
    # `session.execute` is typed as returning `Result`, which has no
    # `rowcount`; a DML statement always yields a `CursorResult`, which does.
    result = cast(
        CursorResult[Any],
        await session.execute(delete(RevokedToken).where(RevokedToken.expires_at < cutoff)),
    )
    return result.rowcount or 0


def set_auth_cookie(response: Response, token: str, *, settings: Settings, max_age: int) -> None:
    """Put the access token in an httpOnly cookie.

    `httponly` is the point: script on the page cannot read it, so an injected
    script cannot exfiltrate the session the way it can read `sessionStorage`.
    `samesite=lax` is what stands in for CSRF protection - the API is only ever
    called with unsafe methods from the app's own origin, and `lax` withholds
    the cookie from cross-site POSTs while still sending it on a top-level
    navigation back into the app.

    The token is also still returned in the response body. That is deliberate:
    the Swagger UI's Authorize button and the operational scripts use the
    bearer header, and taking it away would break both to protect a browser
    that no longer reads it.
    """
    response.set_cookie(
        settings.auth_cookie_name,
        token,
        max_age=max_age,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )


def clear_auth_cookie(response: Response, *, settings: Settings) -> None:
    response.delete_cookie(
        settings.auth_cookie_name,
        path="/",
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
    )
