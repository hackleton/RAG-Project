"""Document upload, ingestion and access control, end to end through the API."""

from __future__ import annotations

import io
import uuid
import zipfile

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditAction, AuditLog, DocumentChunk, DocumentStatus, Patient
from app.providers.storage import sanitise_filename
from tests.conftest import auth

NOTE = b"""CLINICAL PROGRESS NOTE

CHIEF COMPLAINT

Routine follow-up for type 2 diabetes mellitus.

CURRENT MEDICATIONS

- Metformin 1000 mg twice daily
- Lisinopril 10 mg once daily

ALLERGIES

No known drug allergies.

LABORATORY RESULTS

Haemoglobin A1c 7.4% (reference 4.0-5.6%).

ASSESSMENT AND PLAN

Diabetes remains suboptimally controlled. Continue current therapy and repeat
the laboratory panel in three months.
"""


def upload_file(
    content: bytes = NOTE,
    name: str = "note.txt",
    content_type: str = "text/plain",
) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (name, content, content_type)}


async def do_upload(
    client: AsyncClient,
    api: str,
    token: str,
    *,
    content: bytes = NOTE,
    name: str = "note.txt",
    content_type: str = "text/plain",
    data: dict[str, str] | None = None,
):  # type: ignore[no-untyped-def]
    return await client.post(
        f"{api}/documents/upload",
        headers=auth(token),
        files=upload_file(content, name, content_type),
        data=data or {"document_type": "clinical_note", "access_level": "internal"},
    )


# --- ingestion --------------------------------------------------------------


async def test_upload_runs_the_pipeline_and_produces_chunks(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    response = await do_upload(client, api, clinician_token)

    assert response.status_code == 201, response.text
    document = response.json()["document"]
    assert document["status"] == DocumentStatus.INDEXING.value
    assert document["chunk_count"] > 0
    assert document["size_bytes"] == len(NOTE)
    assert len(document["checksum_sha256"]) == 64


async def test_chunks_carry_the_provenance_the_plan_requires(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    """Section 7: document, patient, section, page, chunk index, access level."""
    upload = await do_upload(client, api, clinician_token)
    document_id = upload.json()["document"]["id"]

    response = await client.get(
        f"{api}/documents/{document_id}/chunks", headers=auth(clinician_token)
    )

    assert response.status_code == 200
    chunks = response.json()["items"]
    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk["document_id"] == document_id
        assert chunk["chunk_index"] == index  # contiguous, citable
        assert chunk["page_number"] == 1
        assert chunk["section"]
        assert chunk["access_level"] == "internal"
        assert chunk["token_count"] > 0
        assert chunk["chunk_metadata"]["filename"] == "note.txt"
        # Not embedded yet - that is Sprint 3.
        assert chunk["embedding_reference"] is None


async def test_sections_are_preserved_and_not_mixed(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    upload = await do_upload(client, api, clinician_token)
    document_id = upload.json()["document"]["id"]

    chunks = (
        await client.get(f"{api}/documents/{document_id}/chunks", headers=auth(clinician_token))
    ).json()["items"]

    sections = {chunk["section"] for chunk in chunks}
    assert "CURRENT MEDICATIONS" in sections
    assert "ASSESSMENT AND PLAN" in sections

    medications = " ".join(
        chunk["text"] for chunk in chunks if chunk["section"] == "CURRENT MEDICATIONS"
    )
    assert "Metformin" in medications
    assert "suboptimally controlled" not in medications


async def test_reuploading_identical_bytes_is_deduplicated(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    first = await do_upload(client, api, clinician_token)
    second = await do_upload(client, api, clinician_token)

    assert second.status_code == 201
    assert second.json()["deduplicated"] is True
    assert second.json()["document"]["id"] == first.json()["document"]["id"]


async def test_upload_is_audited_without_recording_the_content(
    client: AsyncClient, api: str, clinician_token: str, session: AsyncSession
) -> None:
    await do_upload(client, api, clinician_token)

    entry = await session.scalar(
        select(AuditLog).where(AuditLog.action == AuditAction.DOCUMENT_UPLOADED)
    )
    assert entry is not None
    assert entry.allowed is True
    assert entry.details["chunk_count"] > 0
    # Project plan section 15: identifiers and counts only.
    assert "Metformin" not in str(entry.details)
    assert "A1c" not in str(entry.details)


# --- failure handling -------------------------------------------------------


async def test_an_unreadable_pdf_fails_with_a_reason(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    """A scan has no text layer. It must fail loudly, not index as empty."""
    response = await do_upload(
        client,
        api,
        clinician_token,
        content=b"%PDF-1.4\nnot really a pdf",
        name="scan.pdf",
        content_type="application/pdf",
    )

    assert response.status_code == 201
    document = response.json()["document"]
    assert document["status"] == DocumentStatus.FAILED.value
    assert document["status_detail"]
    assert document["chunk_count"] == 0


async def test_an_empty_file_is_rejected(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    response = await do_upload(client, api, clinician_token, content=b"")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_an_unsupported_type_is_rejected(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("payload.txt", "data")

    response = await do_upload(
        client,
        api,
        clinician_token,
        content=payload.getvalue(),
        name="archive.zip",
        content_type="application/zip",
    )

    assert response.status_code == 422
    assert ".pdf" in str(response.json()["error"]["details"]["supported"])


async def test_a_file_over_the_limit_is_rejected(
    client: AsyncClient, api: str, clinician_token: str, settings
) -> None:
    oversized = b"x" * (settings.max_upload_bytes + 1)

    response = await do_upload(client, api, clinician_token, content=oversized)

    assert response.status_code == 422
    assert response.json()["error"]["details"]["limit_bytes"] == settings.max_upload_bytes


# --- access control ---------------------------------------------------------


async def test_an_analyst_cannot_upload(client: AsyncClient, api: str, analyst_token: str) -> None:
    response = await do_upload(client, api, analyst_token)

    assert response.status_code == 403


async def test_uploading_for_an_out_of_scope_patient_is_refused(
    client: AsyncClient, api: str, clinician_token: str, patients: list[Patient]
) -> None:
    """The clinician is granted patients[0] only."""
    response = await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "restricted",
            "patient_external_id": patients[1].external_id,
        },
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "permission_denied"


async def test_uploading_for_a_granted_patient_is_allowed(
    client: AsyncClient, api: str, clinician_token: str, patients: list[Patient]
) -> None:
    response = await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "restricted",
            "patient_external_id": patients[0].external_id,
        },
    )

    assert response.status_code == 201
    assert response.json()["document"]["patient_id"] == str(patients[0].id)


async def test_a_patient_document_must_be_restricted(
    client: AsyncClient, api: str, clinician_token: str, patients: list[Patient]
) -> None:
    """Otherwise patient filtering is defeated by the sensitivity filter."""
    response = await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "internal",
            "patient_external_id": patients[0].external_id,
        },
    )

    assert response.status_code == 422
    assert "restricted" in response.json()["error"]["message"]


async def test_an_unknown_patient_is_a_404(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    response = await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "restricted",
            "patient_external_id": "SYN-9999",
        },
    )

    assert response.status_code == 404


async def test_an_analyst_cannot_read_a_restricted_document(
    client: AsyncClient,
    api: str,
    clinician_token: str,
    analyst_token: str,
    patients: list[Patient],
) -> None:
    upload = await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "restricted",
            "patient_external_id": patients[0].external_id,
        },
    )
    document_id = upload.json()["document"]["id"]

    response = await client.get(f"{api}/documents/{document_id}", headers=auth(analyst_token))

    # 404, not 403: a 403 would confirm the document exists.
    assert response.status_code == 404


async def test_chunks_are_not_a_way_around_document_access(
    client: AsyncClient,
    api: str,
    clinician_token: str,
    analyst_token: str,
    patients: list[Patient],
) -> None:
    upload = await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "restricted",
            "patient_external_id": patients[0].external_id,
        },
    )
    document_id = upload.json()["document"]["id"]

    response = await client.get(
        f"{api}/documents/{document_id}/chunks", headers=auth(analyst_token)
    )

    assert response.status_code == 404


async def test_listing_only_returns_visible_documents(
    client: AsyncClient,
    api: str,
    admin_token: str,
    clinician_token: str,
    analyst_token: str,
    patients: list[Patient],
) -> None:
    # Restricted, for a patient the analyst has no grant for.
    await do_upload(
        client,
        api,
        clinician_token,
        data={
            "document_type": "clinical_note",
            "access_level": "restricted",
            "patient_external_id": patients[0].external_id,
        },
    )
    # Internal guidance with no patient link.
    await do_upload(
        client,
        api,
        admin_token,
        content=b"CLINICAL GUIDELINE\n\nScreening cadence for adults.\n",
        name="guideline.txt",
        data={"document_type": "guideline", "access_level": "internal"},
    )

    as_analyst = (await client.get(f"{api}/documents", headers=auth(analyst_token))).json()
    as_admin = (await client.get(f"{api}/documents", headers=auth(admin_token))).json()

    assert as_analyst["total"] == 1
    assert as_analyst["items"][0]["document_type"] == "guideline"
    assert as_admin["total"] == 2


async def test_an_unknown_document_is_a_404(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    response = await client.get(f"{api}/documents/{uuid.uuid4()}", headers=auth(clinician_token))

    assert response.status_code == 404


# --- storage safety ---------------------------------------------------------


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Windows\\System32\\config.sys", "config.sys"),
        ("note.txt", "note.txt"),
        ("..", "document"),
        ("", "document"),
        ("réport final.PDF", "report-final.pdf"),
    ],
)
def test_upload_filenames_cannot_escape_the_storage_root(supplied: str, expected: str) -> None:
    assert sanitise_filename(supplied) == expected


async def test_a_traversal_filename_is_stored_safely(
    client: AsyncClient, api: str, clinician_token: str, session: AsyncSession
) -> None:
    response = await do_upload(client, api, clinician_token, name="../../../evil.txt")

    assert response.status_code == 201
    document_id = uuid.UUID(response.json()["document"]["id"])
    chunk = await session.scalar(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    assert chunk is not None
    # The row keeps what the caller sent; the path on disk does not.
    assert chunk.chunk_metadata["filename"] == "../../../evil.txt"
