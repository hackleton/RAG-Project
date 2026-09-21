"""Authentication endpoints.

Two entry points for the same credential check: a JSON `/auth/login` for the
web app, and an OAuth2 password-form `/auth/token` so the generated OpenAPI
docs have a working "Authorize" button.

Failure handling is uniform on purpose - unknown email and wrong password
return the same 401 and both still run a bcrypt comparison, so response time
does not disclose whether an account exists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, SettingsDep, client_ip
from app.core.errors import AuthenticationError
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password_async
from app.models.enums import AuditAction
from app.models.user import User
from app.schemas.auth import CurrentUserResponse, LoginRequest, TokenResponse
from app.security.audit import record_audit, record_audit_independent

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

# Compared against when the email is unknown, so the bcrypt cost is paid either
# way and login timing does not leak account existence.
_DUMMY_HASH = hash_password("timing-equalisation-placeholder", rounds=12)


async def _authenticate(
    session: DbSession,
    settings: SettingsDep,
    request: Request,
    *,
    email: str,
    password: str,
) -> TokenResponse:
    normalised = email.strip().lower()
    user = await session.scalar(select(User).where(User.email == normalised))

    if user is None:
        # Still pay the bcrypt cost so an unknown email is not faster to reject.
        await verify_password_async(password, _DUMMY_HASH)
        password_ok = False
    else:
        password_ok = await verify_password_async(password, user.hashed_password)

    if user is None or not password_ok or not user.is_active:
        # Out of band: raising below rolls back this request's transaction.
        await record_audit_independent(
            action=AuditAction.LOGIN_FAILED,
            actor_email=normalised,
            allowed=False,
            ip_address=client_ip(request),
            details={"reason": "invalid_credentials"},
        )
        raise AuthenticationError("Incorrect email or password.")

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
        ip_address=client_ip(request),
        details={"role": user.role.value},
    )
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for an access token (JSON)",
    responses={401: {"description": "Incorrect email or password."}},
)
async def login(
    payload: LoginRequest, request: Request, session: DbSession, settings: SettingsDep
) -> TokenResponse:
    return await _authenticate(
        session, settings, request, email=payload.email, password=payload.password
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
    session: DbSession,
    settings: SettingsDep,
) -> TokenResponse:
    # OAuth2 calls the identifier `username`; this API uses the email address.
    return await _authenticate(
        session, settings, request, email=form.username, password=form.password
    )


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
