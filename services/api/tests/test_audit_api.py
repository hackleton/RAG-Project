"""Reading the audit trail back.

The trail has been written since Sprint 1 and nothing could read it. These
tests are about the read path being usable and being as guarded as the data it
describes: rows name who looked at which patient's document, so an endpoint
that returns them is exactly as sensitive as the documents themselves.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog, Patient, User
from app.models.enums import AuditAction
from tests.conftest import auth
from tests.test_retrieval import DIABETES_NOTE, upload


async def entries(client: AsyncClient, api: str, token: str, **params: object) -> dict:
    response = await client.get(f"{api}/audit", headers=auth(token), params=params)
    assert response.status_code == 200, response.text
    result: dict = response.json()
    return result


# --- who may read it --------------------------------------------------------


async def test_an_analyst_cannot_read_the_audit_trail(
    client: AsyncClient, api: str, analyst_token: str
) -> None:
    response = await client.get(f"{api}/audit", headers=auth(analyst_token))
    assert response.status_code == 403


async def test_a_clinician_cannot_read_the_audit_trail(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    """Rows name other clinicians and their patients; seeing one's own actions
    is a different feature, not a smaller version of this one."""
    response = await client.get(f"{api}/audit", headers=auth(clinician_token))
    assert response.status_code == 403


async def test_an_anonymous_caller_cannot_read_the_audit_trail(
    client: AsyncClient, api: str
) -> None:
    assert (await client.get(f"{api}/audit")).status_code == 401


# --- what it returns --------------------------------------------------------


async def test_a_login_is_visible_in_the_trail(
    client: AsyncClient, api: str, admin_token: str, users: dict[str, User]
) -> None:
    page = await entries(client, api, admin_token, action="login_succeeded")

    assert page["total"] >= 1
    assert {item["actor_email"] for item in page["items"]} <= {
        user.email for user in users.values()
    }
    assert all(item["allowed"] is True for item in page["items"])


async def test_a_document_upload_is_recorded_with_its_resource(
    client: AsyncClient,
    api: str,
    admin_token: str,
    patients: list[Patient],
) -> None:
    document_id = await upload(
        client,
        api,
        admin_token,
        DIABETES_NOTE,
        "audited.txt",
        patient_external_id=patients[0].external_id,
        # A patient-linked document may only be uploaded as restricted; the
        # helper's default of "internal" is for the unattributed material.
        access_level="restricted",
    )
    page = await entries(client, api, admin_token, action="document_uploaded")

    assert document_id in {item["resource_id"] for item in page["items"]}


async def test_newest_rows_come_first(
    client: AsyncClient, api: str, admin_token: str, session: AsyncSession
) -> None:
    page = await entries(client, api, admin_token, limit=50)
    stamps = [item["created_at"] for item in page["items"]]

    assert stamps == sorted(stamps, reverse=True)


async def test_the_total_counts_the_same_rows_the_page_lists(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    """A total computed over different predicates than the page is worse than none."""
    everything = await entries(client, api, admin_token, limit=1)
    logins = await entries(client, api, admin_token, action="login_succeeded", limit=1)

    assert logins["total"] <= everything["total"]
    assert logins["total"] >= 1


async def test_a_role_denial_at_the_http_gate_is_recorded(
    client: AsyncClient, api: str, admin_token: str, analyst_token: str
) -> None:
    """The gap Sprint 6 closed.

    The registry recorded a denied *tool* call and the document service
    recorded a denied patient, but a caller turned away at the HTTP role gate
    left no row at all - and that is where someone probing what they can reach
    shows up first.
    """
    refused = await client.post(
        f"{api}/documents/upload",
        headers=auth(analyst_token),
        files={"file": ("nope.txt", b"content", "text/plain")},
        data={"document_type": "clinical_note"},
    )
    assert refused.status_code == 403

    page = await entries(client, api, admin_token, action="access_denied")

    assert page["total"] >= 1
    row = page["items"][0]
    assert row["allowed"] is False
    assert row["resource_id"].endswith("/documents/upload")
    assert row["details"]["actual_role"] == "analyst"


async def test_refusals_can_be_singled_out_whatever_produced_them(
    client: AsyncClient, api: str, admin_token: str, analyst_token: str
) -> None:
    """The refusals are the rows a reviewer is actually looking for."""
    await client.post(
        f"{api}/documents/upload",
        headers=auth(analyst_token),
        files={"file": ("nope.txt", b"content", "text/plain")},
        data={"document_type": "clinical_note"},
    )
    page = await entries(client, api, admin_token, allowed=False)

    assert page["total"] >= 1
    assert all(item["allowed"] is False for item in page["items"])


async def test_filtering_by_actor_is_case_insensitive(
    client: AsyncClient, api: str, admin_token: str, users: dict[str, User]
) -> None:
    page = await entries(client, api, admin_token, actor_email=users["admin"].email.upper())
    assert page["total"] >= 1


async def test_a_since_filter_excludes_older_rows(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    page = await entries(client, api, admin_token, since=future)

    assert page["total"] == 0
    assert page["items"] == []


async def test_paging_does_not_repeat_a_row(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    first = await entries(client, api, admin_token, limit=2, offset=0)
    second = await entries(client, api, admin_token, limit=2, offset=2)

    assert not {item["id"] for item in first["items"]} & {item["id"] for item in second["items"]}


# --- the read is itself an event -------------------------------------------


async def test_reading_the_trail_is_recorded_in_it(
    client: AsyncClient, api: str, session: AsyncSession, admin_token: str
) -> None:
    """Otherwise the one action that reveals everything is the one that leaves
    no trace."""
    await entries(client, api, admin_token, action="login_succeeded")

    rows = list(
        await session.scalars(select(AuditLog).where(AuditLog.action == AuditAction.AUDIT_ACCESSED))
    )
    assert len(rows) == 1
    assert rows[0].details["filters"]["action"] == "login_succeeded"


# --- the summary ------------------------------------------------------------


async def test_the_summary_counts_actions_and_breaks_out_denials(
    client: AsyncClient, api: str, admin_token: str, analyst_token: str
) -> None:
    await client.get(f"{api}/audit", headers=auth(analyst_token))  # 403
    await client.post(
        f"{api}/documents/upload",
        headers=auth(analyst_token),
        files={"file": ("nope.txt", b"content", "text/plain")},
        data={"document_type": "clinical_note"},
    )

    response = await client.get(f"{api}/audit/summary", headers=auth(admin_token))
    assert response.status_code == 200, response.text
    body = response.json()

    counts = {row["action"]: row for row in body["counts"]}
    assert body["total"] == sum(row["total"] for row in body["counts"])
    assert counts["login_succeeded"]["denied"] == 0
    assert counts["login_succeeded"]["total"] >= 1


async def test_the_summary_is_admin_only(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    response = await client.get(f"{api}/audit/summary", headers=auth(clinician_token))
    assert response.status_code == 403
