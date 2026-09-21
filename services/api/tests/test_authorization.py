"""Role and scope enforcement.

These tests assert only that the authorisation layer let the request through or
did not - never what the handler then did. For an endpoint whose behaviour is
still ahead of us, a 501 makes that distinction cleanly: it means the role check
passed and the handler ran. For one that is implemented, its own test module
covers the behaviour and the check here stays about access.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.models import ACCESS_LEVEL_ROLES, AccessLevel, Role
from app.security.principal import Principal
from tests.conftest import auth


def _principal(role: Role, patients: frozenset[uuid.UUID] | None = None) -> Principal:
    return Principal(
        id=uuid.uuid4(),
        email=f"{role.value}@example.com",
        name=role.value.title(),
        role=role,
        accessible_patient_ids=patients,
    )


async def test_admin_only_endpoint_rejects_an_analyst(
    client: AsyncClient, api: str, analyst_token: str
) -> None:
    response = await client.post(f"{api}/evaluations/run", headers=auth(analyst_token))

    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "permission_denied"
    assert body["error"]["details"]["required_roles"] == ["admin"]
    assert body["error"]["details"]["actual_role"] == "analyst"


async def test_admin_only_endpoint_admits_an_admin(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    response = await client.post(f"{api}/evaluations/run", headers=auth(admin_token))

    # 501: authorisation passed, the handler ran, the behaviour lands in Sprint 5.
    assert response.status_code == 501
    assert response.json()["error"]["details"]["sprint"].startswith("Sprint 5")


async def test_upload_rejects_an_analyst(client: AsyncClient, api: str, analyst_token: str) -> None:
    response = await client.post(
        f"{api}/documents/upload",
        headers=auth(analyst_token),
        files={"file": ("note.txt", b"synthetic content", "text/plain")},
    )

    assert response.status_code == 403


async def test_upload_admits_a_clinician(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    response = await client.post(
        f"{api}/documents/upload",
        headers=auth(clinician_token),
        files={"file": ("note.txt", b"HISTORY\n\nSynthetic clinical content.\n", "text/plain")},
        data={"access_level": "internal"},
    )

    # Implemented in Sprint 2; the ingestion behaviour itself is covered in
    # tests/test_documents.py. Here it only has to prove the role check passed.
    assert response.status_code == 201


async def test_protected_endpoints_reject_anonymous_callers(client: AsyncClient, api: str) -> None:
    for method, path in [
        ("POST", f"{api}/chat"),
        ("POST", f"{api}/search"),
        ("POST", f"{api}/agents/run"),
        ("POST", f"{api}/evaluations/run"),
        ("GET", f"{api}/evaluations/{uuid.uuid4()}"),
        ("GET", f"{api}/documents/{uuid.uuid4()}"),
    ]:
        response = await client.request(method, path)
        assert response.status_code == 401, f"{method} {path} was reachable anonymously"


# --- Unit-level checks on the pure predicates -------------------------------


def test_admin_scope_is_unrestricted() -> None:
    admin = _principal(Role.ADMIN, patients=None)

    assert admin.can_access_patient(uuid.uuid4()) is True
    assert admin.is_admin is True


def test_scoped_user_is_limited_to_granted_patients() -> None:
    granted, other = uuid.uuid4(), uuid.uuid4()
    clinician = _principal(Role.CLINICIAN, patients=frozenset({granted}))

    assert clinician.can_access_patient(granted) is True
    assert clinician.can_access_patient(other) is False


def test_user_with_no_grants_sees_no_patients() -> None:
    clinician = _principal(Role.CLINICIAN, patients=frozenset())

    assert clinician.can_access_patient(uuid.uuid4()) is False


def test_documents_with_no_patient_link_are_not_patient_scoped() -> None:
    """Clinical guidelines and drug references have no patient to be scoped to."""
    clinician = _principal(Role.CLINICIAN, patients=frozenset())

    assert clinician.can_access_patient(None) is True


@pytest.mark.parametrize(
    ("role", "level", "expected"),
    [
        (Role.ANALYST, AccessLevel.PUBLIC, True),
        (Role.ANALYST, AccessLevel.INTERNAL, True),
        (Role.ANALYST, AccessLevel.RESTRICTED, False),
        (Role.CLINICIAN, AccessLevel.RESTRICTED, True),
        (Role.ADMIN, AccessLevel.RESTRICTED, True),
    ],
)
def test_access_level_matrix(role: Role, level: AccessLevel, expected: bool) -> None:
    assert _principal(role).can_access_level(level) is expected


def test_every_access_level_has_an_explicit_role_list() -> None:
    """A new sensitivity level must not default to being visible to everyone."""
    assert set(ACCESS_LEVEL_ROLES) == set(AccessLevel)
