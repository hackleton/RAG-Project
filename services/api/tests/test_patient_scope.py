"""Asking for a patient you may not see, on every endpoint that accepts one.

The rule (app/security/patient_scope.py): an unknown patient and a patient you
are not assigned get the *same* refusal, the refusal carries no details, and
the attempt is audited with the real reason.

Before Sprint 8, chat, search and the agent answered 403 for one case and 404
for the other, the document list answered an empty 200 against a 403, the 403
carried the patient's internal UUID, and none of it was audited. Found by
checking whether the demo's refusals appeared in the audit trail. They didn't.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from httpx import AsyncClient

from app.models import Patient
from tests.conftest import auth

# The seeded clinician is granted SYN-0001 only (see tests/conftest.py).
FORBIDDEN = "SYN-0002"
UNKNOWN = "SYN-9999"


async def request(client: AsyncClient, api: str, token: str, surface: str, patient: str) -> Any:
    headers = auth(token)
    if surface == "chat":
        return await client.post(
            f"{api}/chat",
            headers=headers,
            json={"question": "labs", "patient_external_id": patient},
        )
    if surface == "search":
        return await client.post(
            f"{api}/search", headers=headers, json={"query": "labs", "patient_external_id": patient}
        )
    if surface == "agent":
        return await client.post(
            f"{api}/agents/run",
            headers=headers,
            json={"question": "labs", "patient_external_id": patient},
        )
    if surface == "documents":
        return await client.get(
            f"{api}/documents", headers=headers, params={"patient_external_id": patient}
        )
    if surface == "upload":
        return await client.post(
            f"{api}/documents/upload",
            headers=headers,
            files={"file": ("note.txt", io.BytesIO(b"CLINICAL NOTE\n\nText."), "text/plain")},
            data={
                "document_type": "clinical_note",
                "access_level": "restricted",
                "patient_external_id": patient,
            },
        )
    raise AssertionError(surface)


SURFACES = ("chat", "search", "agent", "documents", "upload")


@pytest.mark.parametrize("surface", SURFACES)
async def test_unknown_and_forbidden_are_indistinguishable(
    client: AsyncClient, api: str, clinician_token: str, patients: list[Patient], surface: str
) -> None:
    unknown = await request(client, api, clinician_token, surface, UNKNOWN)
    forbidden = await request(client, api, clinician_token, surface, FORBIDDEN)

    assert unknown.status_code == forbidden.status_code == 403
    assert unknown.json()["error"] == forbidden.json()["error"]


@pytest.mark.parametrize("surface", SURFACES)
async def test_a_refusal_carries_no_identifier(
    client: AsyncClient, api: str, clinician_token: str, patients: list[Patient], surface: str
) -> None:
    """The 403 used to include the refused patient's internal UUID."""
    response = await request(client, api, clinician_token, surface, FORBIDDEN)
    forbidden_id = str(next(p.id for p in patients if p.external_id == FORBIDDEN))

    assert response.status_code == 403
    assert not response.json()["error"].get("details")
    assert forbidden_id not in response.text
    assert FORBIDDEN not in response.text


@pytest.mark.parametrize("surface", SURFACES)
async def test_every_refusal_is_audited_with_its_reason(
    client: AsyncClient,
    api: str,
    clinician_token: str,
    admin_token: str,
    patients: list[Patient],
    surface: str,
) -> None:
    """The response hides the reason; the audit trail, which only an
    administrator can read, keeps it, so probing for IDs is visible as probing."""
    await request(client, api, clinician_token, surface, UNKNOWN)
    await request(client, api, clinician_token, surface, FORBIDDEN)

    trail = await client.get(
        f"{api}/audit",
        headers=auth(admin_token),
        params={"action": "access_denied", "resource_type": "patient"},
    )
    assert trail.status_code == 200, trail.text
    rows = [row for row in trail.json()["items"] if row["details"].get("surface") == surface]

    reasons = {(row["resource_id"], row["details"]["reason"]) for row in rows}
    assert reasons == {(UNKNOWN, "unknown"), (FORBIDDEN, "not_granted")}
    assert all(row["actor_email"] == "clinician@example.com" for row in rows)
    assert all(row["allowed"] is False for row in rows)


async def test_a_granted_patient_still_works(
    client: AsyncClient, api: str, clinician_token: str, patients: list[Patient]
) -> None:
    """The fix must not turn into refusing everyone."""
    for surface in ("chat", "search", "agent", "documents"):
        response = await request(client, api, clinician_token, surface, "SYN-0001")
        assert response.status_code == 200, (surface, response.text)


async def test_an_oversized_identifier_is_rejected_at_the_edge(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    """Bounded before it reaches the resolver and its 128-character audit column."""
    response = await request(client, api, clinician_token, "search", "X" * 500)
    assert response.status_code == 422
