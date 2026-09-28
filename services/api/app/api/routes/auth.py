"""Authentication endpoints.

Two entry points for the same credential check: a JSON `/auth/login` for the
web app, and an OAuth2 password-form `/auth/token` so the generated OpenAPI
docs have a working "Authorize" button. Both issue the same token two ways -
in the response body for the Swagger UI and the scripts, and in an httpOnly
cookie for the browser app, which reads neither (decision 36).

Failure handling is uniform on purpose - unknown email and wrong password
return the same 401 and both still run a bcrypt comparison, so response time
does not disclose whether an account exists.

Sprint 6 adds two things in front of and behind that: a failed-attempt limit
per email+IP, and `/auth/logout`, which revokes the presented token's `jti`
rather than only asking the browser to forget it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, SettingsDep, bearer_or_cookie, client_ip
from app.core.config import Settings
from app.core.errors import AuthenticationError, RateLimitedError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password_async,
)
from app.models.enums import AuditAction
from app.models.user import User
from app.schemas.auth import CurrentUserResponse, LoginRequest, LogoutResponse, TokenResponse
from app.security import tokens
from app.security.audit import record_audit, record_audit_independent
from app.security.throttle import LoginThrottle, SharedLoginThrottle

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

# Compared against when the email is unknown, so the bcrypt cost is paid either
# way and login timing does not leak account existence.
_DUMMY_HASH = hash_password("timing-equalisation-placeholder", rounds=12)


def _throttle(request: Request, settings: Settings) -> SharedLoginThrottle:
    """The app's throttle, created on first use.

    On `app.state` rather than at module scope so the test suite - which builds
    several apps in one process - does not carry one app's failed attempts into
    the next, and so the limits come from *this* app's settings.

    The shared counter lives in the cache backend; the in-process one lives
    here. Both are consulted, and either can refuse - see
    `app.security.throttle`.
    """
    existing: SharedLoginThrottle | None = getattr(request.app.state, "login_throttle", None)
    if existing is None:
        existing = SharedLoginThrottle(
            request.app.state.cache,
            LoginThrottle(
                max_attempts=settings.login_max_attempts,
                window_seconds=settings.login_attempt_window_seconds,
            ),
            prefix=settings.cache_prefix,
            max_attempts=settings.login_max_attempts,
            window_seconds=settings.login_attempt_window_seconds,
        )
        request.app.state.login_throttle = existing
    return existing


async def _authenticate(
    session: DbSession,
    settings: SettingsDep,
    request: Request,
    response: Response,
    *,
    email: str,
    password: str,
) -> TokenResponse:
    normalised = email.strip().lower()
    ip = client_ip(request)
    throttle = _throttle(request, settings)

    if not await throttle.check(normalised, ip):
        # Refused before the password is looked at, so a locked-out attacker
        # cannot keep the server paying bcrypt's cost. Audited out of band:
        # raising rolls this request's transaction back.
        await record_audit_independent(
            action=AuditAction.LOGIN_THROTTLED,
            actor_email=normalised,
            allowed=False,
            ip_address=ip,
            details={"window_seconds": settings.login_attempt_window_seconds},
        )
        logger.warning("login_throttled", email=normalised, ip=ip)
        raise RateLimitedError(
            "Too many failed sign-in attempts. Try again shortly.",
            details={"retry_after_seconds": settings.login_attempt_window_seconds},
        )

    user = await session.scalar(select(User).where(User.email == normalised))

    if user is None:
        # Still pay the bcrypt cost so an unknown email is not faster to reject.
        await verify_password_async(password, _DUMMY_HASH)
        password_ok = False
    else:
        password_ok = await verify_password_async(password, user.hashed_password)

    if user is None or not password_ok or not user.is_active:
        attempts = await throttle.record_failure(normalised, ip)
        await record_audit_independent(
            action=AuditAction.LOGIN_FAILED,
            actor_email=normalised,
            allowed=False,
            ip_address=ip,
            details={"reason": "invalid_credentials", "attempts_in_window": attempts},
        )
        raise AuthenticationError("Incorrect email or password.")

    await throttle.reset(normalised, ip)
    token, expires_in = create_access_token(
        subject=str(user.id), role=user.role.value, settings=settings
    )
    user.last_login_at = datetime.now(UTC)
    await record_audit(
        session,
        action=AuditAction.LOGIN_SUCCEEDED,
        actor_email=user.email,
        resource_type="user",
        resource_id=user.id,
        allowed=True,
        ip_address=ip,
        details={"role": user.role.value},
    )
    if settings.auth_cookie_enabled:
        tokens.set_auth_cookie(response, token, settings=settings, max_age=expires_in)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for an access token (JSON)",
    responses={
        401: {"description": "Incorrect email or password."},
        429: {"description": "Too many failed attempts from this address."},
    },
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: DbSession,
    settings: SettingsDep,
) -> TokenResponse:
    return await _authenticate(
        session, settings, request, response, email=payload.email, password=payload.password
    )


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2 password flow (used by the Swagger UI)",
    include_in_schema=True,
)
async def token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    response: Response,
    session: DbSession,
    settings: SettingsDep,
) -> TokenResponse:
    # OAuth2 calls the identifier `username`; this API uses the email address.
    return await _authenticate(
        session, settings, request, response, email=form.username, password=form.password
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Revoke the presented token and clear the auth cookie",
)
async def logout(
    response: Response,
    session: DbSession,
    settings: SettingsDep,
    principal: CurrentUser,
    presented: Annotated[str | None, Depends(bearer_or_cookie)] = None,
) -> LogoutResponse:
    """Clearing the cookie is not revocation; both happen here.

    A logout that only deletes the cookie leaves a token that is still valid
    for the rest of its hour to anyone who captured it. `jti` has been in every
    token since Sprint 1 for exactly this.
    """
    revoked = False
    if presented:
        # `decode_access_token` requires `exp`, so it is present or this raised.
        claims = decode_access_token(presented, settings)
        expires = datetime.fromtimestamp(int(claims["exp"]), tz=UTC)
        await tokens.revoke(
            session, jti=str(claims.get("jti", "")), user_id=principal.id, expires_at=expires
        )
        revoked = True
        # The only moment the table is known to have grown, so the only moment
        # worth trimming it. Saves the deployment a scheduled job.
        await tokens.purge_expired(session)

    tokens.clear_auth_cookie(response, settings=settings)
    await record_audit(
        session,
        action=AuditAction.LOGOUT,
        principal=principal,
        resource_type="user",
        resource_id=principal.id,
        allowed=True,
        details={"token_revoked": revoked},
    )

    # Committed here rather than left to the request teardown, which runs after
    # the response has gone out. Without this the endpoint answers "revoked"
    # while the row is still uncommitted, and a request arriving in the next
    # few milliseconds is still authenticated by the token it just gave up -
    # measured at roughly a 50ms window against SQL Server. Everywhere else
    # that ordering is harmless; here the caller's next request is precisely
    # what the write exists to affect.
    await session.commit()
    return LogoutResponse(revoked=revoked)


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="The authenticated caller and the patient scope applied to their queries",
    status_code=status.HTTP_200_OK,
)
async def me(principal: CurrentUser, session: DbSession) -> CurrentUserResponse:
    user = await session.get(User, principal.id)
    if user is None:  # pragma: no cover - resolved principal implies the row exists
        raise AuthenticationError("User account is not active.")
    return CurrentUserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        accessible_patient_ids=(
            None
            if principal.accessible_patient_ids is None
            else sorted(principal.accessible_patient_ids)
        ),
    )
