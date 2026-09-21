"""Authentication endpoints and token handling."""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import create_access_token
from app.models import AuditAction, AuditLog, Role, User
from tests.conftest import ADMIN_PASSWORD, CLINICIAN_PASSWORD, auth


async def test_login_returns_a_bearer_token(
    client: AsyncClient, api: str, users: dict[str, User]
) -> None:
    response = await client.post(
        f"{api}/auth/login",
        json={"email": users["admin"].email, "password": ADMIN_PASSWORD},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["access_token"]


async def test_login_is_case_insensitive_on_email(
    client: AsyncClient, api: str, users: dict[str, User]
) -> None:
    response = await client.post(
        f"{api}/auth/login",
        json={"email": users["admin"].email.upper(), "password": ADMIN_PASSWORD},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("admin@example.com", "wrong-password"),
        ("nobody@example.com", ADMIN_PASSWORD),
    ],
    ids=["wrong_password", "unknown_email"],
)
async def test_bad_credentials_are_indistinguishable(
    client: AsyncClient, api: str, users: dict[str, User], email: str, password: str
) -> None:
    """Wrong password and unknown account must return the same thing."""
    response = await client.post(f"{api}/auth/login", json={"email": email, "password": password})

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "authentication_failed"
    assert body["error"]["message"] == "Incorrect email or password."
    assert response.headers["www-authenticate"] == "Bearer"


async def test_failed_login_is_audited(
    client: AsyncClient, api: str, users: dict[str, User], session: AsyncSession
) -> None:
    await client.post(
        f"{api}/auth/login",
        json={"email": users["admin"].email, "password": "wrong-password"},
    )

    entry = await session.scalar(
        select(AuditLog).where(AuditLog.action == AuditAction.LOGIN_FAILED)
    )
    assert entry is not None
    assert entry.allowed is False
    assert entry.actor_email == users["admin"].email
    # The attempted password must never reach the audit trail.
    assert "wrong-password" not in str(entry.details)


async def test_successful_login_is_audited(
    client: AsyncClient, api: str, users: dict[str, User], session: AsyncSession
) -> None:
    await client.post(
        f"{api}/auth/login",
        json={"email": users["admin"].email, "password": ADMIN_PASSWORD},
    )

    entry = await session.scalar(
        select(AuditLog).where(AuditLog.action == AuditAction.LOGIN_SUCCEEDED)
    )
    assert entry is not None
    assert entry.allowed is True
    assert entry.details["role"] == Role.ADMIN.value


async def test_oauth2_form_flow_works(
    client: AsyncClient, api: str, users: dict[str, User]
) -> None:
    """The Swagger "Authorize" button posts a form, not JSON."""
    response = await client.post(
        f"{api}/auth/token",
        data={"username": users["admin"].email, "password": ADMIN_PASSWORD},
    )

    assert response.status_code == 200
    assert response.json()["access_token"]


async def test_inactive_account_cannot_log_in(
    client: AsyncClient, api: str, users: dict[str, User], session: AsyncSession
) -> None:
    user = await session.get(User, users["clinician"].id)
    assert user is not None
    user.is_active = False
    await session.commit()

    response = await client.post(
        f"{api}/auth/login",
        json={"email": users["clinician"].email, "password": CLINICIAN_PASSWORD},
    )

    assert response.status_code == 401


async def test_me_returns_identity_and_patient_scope(
    client: AsyncClient, api: str, clinician_token: str, patients: list
) -> None:
    response = await client.get(f"{api}/auth/me", headers=auth(clinician_token))

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == Role.CLINICIAN.value
    # The clinician is granted exactly one of the two synthetic patients.
    assert body["accessible_patient_ids"] == [str(patients[0].id)]


async def test_me_reports_unrestricted_scope_for_admins(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    response = await client.get(f"{api}/auth/me", headers=auth(admin_token))

    assert response.status_code == 200
    assert response.json()["accessible_patient_ids"] is None


async def test_me_requires_a_token(client: AsyncClient, api: str) -> None:
    response = await client.get(f"{api}/auth/me")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


async def test_garbage_token_is_rejected(client: AsyncClient, api: str) -> None:
    response = await client.get(f"{api}/auth/me", headers=auth("not-a-jwt"))

    assert response.status_code == 401


async def test_expired_token_is_rejected(
    client: AsyncClient, api: str, settings: Settings, users: dict[str, User]
) -> None:
    token, _ = create_access_token(
        subject=str(users["admin"].id),
        role=Role.ADMIN.value,
        settings=settings,
        expires_delta=timedelta(seconds=-1),
    )

    response = await client.get(f"{api}/auth/me", headers=auth(token))

    assert response.status_code == 401
    assert "expired" in response.json()["error"]["message"].lower()


async def test_token_signed_with_another_key_is_rejected(
    client: AsyncClient, api: str, settings: Settings, users: dict[str, User]
) -> None:
    forged_settings = settings.model_copy(
        update={"jwt_secret_key": "an-attackers-key-padded-to-realistic-length"}
    )
    token, _ = create_access_token(
        subject=str(users["admin"].id), role=Role.ADMIN.value, settings=forged_settings
    )

    response = await client.get(f"{api}/auth/me", headers=auth(token))

    assert response.status_code == 401


async def test_deactivated_user_loses_access_immediately(
    client: AsyncClient,
    api: str,
    admin_token: str,
    users: dict[str, User],
    session: AsyncSession,
) -> None:
    """A still-valid token must not outlive the account it was issued for."""
    user = await session.get(User, users["admin"].id)
    assert user is not None
    user.is_active = False
    await session.commit()

    response = await client.get(f"{api}/auth/me", headers=auth(admin_token))

    assert response.status_code == 401


async def test_role_change_takes_effect_without_reissuing_the_token(
    client: AsyncClient,
    api: str,
    admin_token: str,
    users: dict[str, User],
    session: AsyncSession,
) -> None:
    """The database is authoritative for the role, not the claim in the token."""
    user = await session.get(User, users["admin"].id)
    assert user is not None
    user.role = Role.ANALYST
    await session.commit()

    response = await client.get(f"{api}/auth/me", headers=auth(admin_token))

    assert response.status_code == 200
    assert response.json()["role"] == Role.ANALYST.value
