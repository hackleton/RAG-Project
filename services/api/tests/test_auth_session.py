"""The session: how a token is carried, given up, and rate limited.

Sprint 6 moves the browser app off `sessionStorage` and onto an httpOnly
cookie, gives `/auth/logout` something to do beyond clearing it, and puts a
limit on failed attempts. Each of those is a control with a failure mode worth
pinning down:

* a cookie that is not `httpOnly` is `sessionStorage` with extra steps;
* a logout that only clears the cookie leaves a valid token in the hands of
  whoever captured it;
* a throttle that counts refusals as failures extends its own lockout forever.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppEnv, Settings
from app.core.security import create_access_token, decode_access_token
from app.models import AuditLog, RevokedToken, User
from app.models.enums import AuditAction
from app.security import tokens
from app.security.throttle import LoginThrottle
from tests.conftest import CLINICIAN_PASSWORD, auth

# --- the cookie -------------------------------------------------------------


async def test_login_sets_an_httponly_cookie_alongside_the_body_token(
    client: AsyncClient, api: str, settings: Settings, users: dict[str, User]
) -> None:
    response = await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )

    assert response.status_code == 200, response.text
    cookie = response.cookies.get(settings.auth_cookie_name)
    assert cookie == response.json()["access_token"]

    # `httponly` is the whole point: script on the page cannot read it, so an
    # injected script cannot exfiltrate the session.
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header


async def test_the_cookie_authenticates_without_an_authorization_header(
    client: AsyncClient, api: str, users: dict[str, User]
) -> None:
    await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )
    # The client keeps the cookie; no header is sent.
    response = await client.get(f"{api}/auth/me")

    assert response.status_code == 200, response.text
    assert response.json()["email"] == users["clinician"].email


async def test_a_header_token_wins_over_a_stale_cookie(
    client: AsyncClient, api: str, users: dict[str, User], admin_token: str
) -> None:
    """Otherwise "sign in as someone else" depends on what the browser kept."""
    await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )
    response = await client.get(f"{api}/auth/me", headers=auth(admin_token))

    assert response.json()["email"] == users["admin"].email


async def test_a_request_with_neither_credential_is_rejected(client: AsyncClient, api: str) -> None:
    response = await client.get(f"{api}/auth/me")
    assert response.status_code == 401


# --- logout and revocation --------------------------------------------------


async def test_logout_revokes_the_token_not_merely_the_cookie(
    client: AsyncClient,
    api: str,
    session: AsyncSession,
    settings: Settings,
    clinician_token: str,
) -> None:
    """The bug this exists to prevent: a "logged out" token that still works.

    The header is used deliberately - it is the transport a captured token
    would be replayed on, and clearing a cookie does nothing about it.

    What this *cannot* check is the commit ordering. The ASGI transport runs
    in-process and awaits dependency teardown before returning, so a handler
    that left its write to the teardown would pass here and still hand a
    revoked token a working window over a real connection. That one is caught
    by `scripts/smoke_security.py`, which is where it was found.
    """
    logout = await client.post(f"{api}/auth/logout", headers=auth(clinician_token))
    assert logout.status_code == 200, logout.text
    assert logout.json()["revoked"] is True

    replayed = await client.get(f"{api}/auth/me", headers=auth(clinician_token))
    assert replayed.status_code == 401
    assert replayed.json()["error"]["message"] == "This token has been revoked."

    jti = decode_access_token(clinician_token, settings)["jti"]
    assert await session.scalar(select(RevokedToken).where(RevokedToken.jti == jti)) is not None


async def test_logout_clears_the_cookie(
    client: AsyncClient, api: str, settings: Settings, users: dict[str, User]
) -> None:
    await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )
    await client.post(f"{api}/auth/logout")

    assert not client.cookies.get(settings.auth_cookie_name)


async def test_logging_out_twice_is_not_an_error(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    """A client retrying a logout is ordinary; a 500 for it is not."""
    first = await client.post(f"{api}/auth/logout", headers=auth(clinician_token))
    second = await client.post(f"{api}/auth/logout", headers=auth(clinician_token))

    assert first.status_code == 200
    # The second call presents a token that is now revoked, so it cannot
    # authenticate at all - which is the correct outcome, not an error.
    assert second.status_code == 401


async def test_revoking_one_token_leaves_another_session_alone(
    client: AsyncClient, api: str, users: dict[str, User]
) -> None:
    """Revocation is per token. Logging out on one device is not a global sign-out."""
    first = await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )
    second = await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )
    one, two = first.json()["access_token"], second.json()["access_token"]
    assert one != two

    await client.post(f"{api}/auth/logout", headers=auth(one))

    assert (await client.get(f"{api}/auth/me", headers=auth(one))).status_code == 401
    assert (await client.get(f"{api}/auth/me", headers=auth(two))).status_code == 200


async def test_logout_is_audited(
    client: AsyncClient, api: str, session: AsyncSession, clinician_token: str
) -> None:
    await client.post(f"{api}/auth/logout", headers=auth(clinician_token))

    rows = list(
        await session.scalars(select(AuditLog).where(AuditLog.action == AuditAction.LOGOUT))
    )
    assert len(rows) == 1
    assert rows[0].details["token_revoked"] is True


async def test_expired_revocations_are_purged_on_logout(session: AsyncSession) -> None:
    """The table is trimmed where it is known to have grown, not on a cron.

    Written against the helpers rather than through the endpoint, because the
    row that must disappear is one whose token expired an hour ago and no
    request can present it.
    """
    stale = datetime.now(UTC) - timedelta(hours=2)
    await tokens.revoke(session, jti="long-gone", user_id=None, expires_at=stale)
    await session.flush()

    removed = await tokens.purge_expired(session)

    assert removed == 1
    assert await session.scalar(select(RevokedToken).where(RevokedToken.jti == "long-gone")) is None


async def test_a_token_that_was_never_revoked_passes_the_check(
    session: AsyncSession, settings: Settings, users: dict[str, User]
) -> None:
    token, _ = create_access_token(subject=str(users["admin"].id), role="admin", settings=settings)
    claims = decode_access_token(token, settings)
    assert await tokens.is_revoked(session, claims["jti"]) is False


async def test_a_token_with_no_jti_is_not_treated_as_revoked(session: AsyncSession) -> None:
    """Fail-open is correct here and stated: an absent id cannot be looked up."""
    assert await tokens.is_revoked(session, None) is False


# --- failed-login throttling ------------------------------------------------


def test_the_throttle_refuses_after_the_configured_number_of_failures() -> None:
    throttle = LoginThrottle(max_attempts=3, window_seconds=60)

    for _ in range(3):
        assert throttle.check("a@example.com", "10.0.0.1") is True
        throttle.record_failure("a@example.com", "10.0.0.1")

    assert throttle.check("a@example.com", "10.0.0.1") is False


def test_checking_does_not_count_as_an_attempt() -> None:
    """Otherwise a burst extends its own lockout for as long as it keeps trying."""
    throttle = LoginThrottle(max_attempts=2, window_seconds=60)
    throttle.record_failure("a@example.com", "10.0.0.1")

    for _ in range(50):
        throttle.check("a@example.com", "10.0.0.1")

    assert throttle.check("a@example.com", "10.0.0.1") is True


def test_attempts_fall_out_of_the_window() -> None:
    throttle = LoginThrottle(max_attempts=2, window_seconds=60)
    throttle.record_failure("a@example.com", "10.0.0.1", now=0.0)
    throttle.record_failure("a@example.com", "10.0.0.1", now=1.0)

    assert throttle.check("a@example.com", "10.0.0.1", now=30.0) is False
    assert throttle.check("a@example.com", "10.0.0.1", now=120.0) is True


def test_one_address_cannot_lock_out_another() -> None:
    """Keyed on email *and* IP: on email alone, anyone can lock out anyone."""
    throttle = LoginThrottle(max_attempts=1, window_seconds=60)
    throttle.record_failure("victim@example.com", "10.0.0.1")

    assert throttle.check("victim@example.com", "10.0.0.2") is True
    assert throttle.check("victim@example.com", "10.0.0.1") is False


def test_a_successful_login_clears_the_count() -> None:
    throttle = LoginThrottle(max_attempts=2, window_seconds=60)
    throttle.record_failure("a@example.com", "10.0.0.1")
    throttle.reset("a@example.com", "10.0.0.1")

    assert throttle.check("a@example.com", "10.0.0.1") is True


def test_the_key_map_cannot_grow_without_limit() -> None:
    """A spray across many addresses must not become the memory leak it prevents."""
    throttle = LoginThrottle(max_attempts=5, window_seconds=60, max_keys=8)
    for index in range(200):
        throttle.record_failure(f"user{index}@example.com", "10.0.0.1")

    assert len(throttle._attempts) <= 8


async def test_too_many_failed_logins_are_refused_with_429(
    client: AsyncClient, api: str, users: dict[str, User], settings: Settings
) -> None:
    payload = {"email": users["clinician"].email, "password": "wrong-password"}
    codes = [
        (await client.post(f"{api}/auth/login", json=payload)).status_code
        for _ in range(settings.login_max_attempts + 1)
    ]

    assert codes[: settings.login_max_attempts] == [401] * settings.login_max_attempts
    assert codes[-1] == 429


async def test_a_throttled_attempt_is_audited(
    client: AsyncClient, api: str, session: AsyncSession, users: dict[str, User], settings: Settings
) -> None:
    payload = {"email": users["clinician"].email, "password": "wrong-password"}
    for _ in range(settings.login_max_attempts + 1):
        await client.post(f"{api}/auth/login", json=payload)

    rows = list(
        await session.scalars(
            select(AuditLog).where(AuditLog.action == AuditAction.LOGIN_THROTTLED)
        )
    )
    assert rows, "a refusal is the event most worth recording"


async def test_the_correct_password_still_works_below_the_limit(
    client: AsyncClient, api: str, users: dict[str, User]
) -> None:
    """The throttle must not lock out the legitimate owner of the account."""
    await client.post(
        f"{api}/auth/login", json={"email": users["clinician"].email, "password": "wrong"}
    )
    response = await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )

    assert response.status_code == 200


# --- configuration guard rails ---------------------------------------------


def test_production_refuses_an_insecure_auth_cookie() -> None:
    with pytest.raises(ValueError, match="AUTH_COOKIE_SECURE"):
        Settings(
            app_env=AppEnv.PRODUCTION,
            jwt_secret_key="a" * 48,
            auth_cookie_enabled=True,
            auth_cookie_secure=False,
        )


def test_production_refuses_to_run_with_the_injection_filter_off() -> None:
    """It is configuration so an experiment can measure without it, not so a
    deployment can ship without it."""
    with pytest.raises(ValueError, match="INJECTION_FILTER_ENABLED"):
        Settings(
            app_env=AppEnv.PRODUCTION,
            jwt_secret_key="a" * 48,
            auth_cookie_secure=True,
            injection_filter_enabled=False,
        )
