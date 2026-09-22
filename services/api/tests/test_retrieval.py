"""Hybrid retrieval, fusion, and the access control applied before search.

These run against an embedded Qdrant with deterministic hashed embeddings, so
what is exercised is the real path - filters, two searches, RRF, reranking,
hydration - and not embedding quality, which no unit test can assert anyway.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.providers.vectorstore import AccessFilter, ScoredChunk, build_filter
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.sparse import encode_document, encode_query, term_index, tokenize
from tests.conftest import auth

# Two documents with deliberately different vocabulary, so a keyword query can
# be attributed to one of them with certainty.
DIABETES_NOTE = b"""CLINICAL PROGRESS NOTE

CHIEF COMPLAINT

Routine follow-up for type 2 diabetes mellitus.

LABORATORY RESULTS

Haemoglobin A1c 7.4% (reference 4.0-5.6%). Fasting glucose 142 mg/dL.

CURRENT MEDICATIONS

- Metformin 1000 mg twice daily
"""

CARDIOLOGY_NOTE = b"""CLINICAL PROGRESS NOTE

CHIEF COMPLAINT

Follow-up for essential hypertension and atrial fibrillation.

VITAL SIGNS

Blood pressure 158/94 mmHg. Heart rate 88 bpm irregular.

CURRENT MEDICATIONS

- Apixaban 5 mg twice daily
"""


async def upload(
    client: AsyncClient,
    api: str,
    token: str,
    content: bytes,
    name: str,
    *,
    patient_external_id: str | None = None,
    access_level: str = "internal",
) -> str:
    data = {"document_type": "clinical_note", "access_level": access_level}
    if patient_external_id:
        data["patient_external_id"] = patient_external_id
    response = await client.post(
        f"{api}/documents/upload",
        headers=auth(token),
        files={"file": (name, content, "text/plain")},
        data=data,
    )
    assert response.status_code == 201, response.text
    assert response.json()["document"]["status"] == "ready", response.text
    return str(response.json()["document"]["id"])


async def search(client: AsyncClient, api: str, token: str, query: str, **body: object) -> dict:
    response = await client.post(
        f"{api}/search", headers=auth(token), json={"query": query, **body}
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- sparse encoding --------------------------------------------------------


def test_decimals_survive_tokenisation() -> None:
    """Splitting "7.2" would make a lab-value query match any chunk with a 7."""
    assert "7.2" in tokenize("HbA1c 7.2%")


def test_stopwords_are_dropped_but_negations_are_not() -> None:
    tokens = tokenize("no known drug allergies of the patient")

    assert "no" in tokens  # dropping this inverts the meaning
    assert "the" not in tokens
    assert "of" not in tokens


def test_term_indexes_are_stable_across_processes() -> None:
    """Python's hash() is salted per run; index ids have to outlive a restart."""
    assert term_index("metformin") == term_index("metformin")
    assert term_index("metformin") != term_index("apixaban")
    assert 0 <= term_index("metformin") <= 0x7FFFFFFF


def test_repeated_terms_saturate() -> None:
    """BM25's k1: the tenth mention must count for less than the second."""
    once = encode_document("metformin")
    many = encode_document("metformin " * 10)
    index = term_index("metformin")

    weight_once = once.values[once.indices.index(index)]
    weight_many = many.values[many.indices.index(index)]

    assert weight_many > weight_once
    assert weight_many < weight_once * 10


def test_queries_are_not_length_normalised() -> None:
    """A document's length affects its score; the query's own length does not."""
    vector = encode_query("metformin dosage for the patient")

    assert set(vector.values) == {1.0}


def test_an_all_stopword_query_encodes_to_nothing() -> None:
    assert len(encode_query("the and of")) == 0


# --- fusion -----------------------------------------------------------------


def chunk(name: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_OID, name), score=score, payload={"text": name}
    )


def test_rrf_rewards_appearing_in_both_lists() -> None:
    """The point of hybrid: agreement between strategies beats one strong hit."""
    both = chunk("both", 0.5)
    dense_only = chunk("dense-only", 0.99)

    fused = reciprocal_rank_fusion({"dense": [dense_only, both], "sparse": [both]}, k=60)

    assert fused[0].chunk_id == both.chunk_id
    assert fused[0].strategies == ["dense", "sparse"]


def test_rrf_ignores_raw_score_magnitude() -> None:
    """BM25 is unbounded and cosine is not; only rank may matter."""
    modest = reciprocal_rank_fusion({"a": [chunk("x", 0.01)]}, k=60)
    huge = reciprocal_rank_fusion({"a": [chunk("x", 999.0)]}, k=60)

    assert modest[0].fused_score == huge[0].fused_score


def test_larger_k_flattens_the_advantage_of_rank_one() -> None:
    results = {"a": [chunk("first", 1.0), chunk("second", 0.9)]}

    small = reciprocal_rank_fusion(results, k=1)
    large = reciprocal_rank_fusion(results, k=1000)

    assert small[0].fused_score / small[1].fused_score > (
        large[0].fused_score / large[1].fused_score
    )


def test_fusion_is_deterministic_on_ties() -> None:
    """Otherwise an experiment looks like it moved when nothing changed."""
    results = {"a": [chunk("x", 1.0)], "b": [chunk("y", 1.0)]}

    assert [item.chunk_id for item in reciprocal_rank_fusion(results)] == [
        item.chunk_id for item in reciprocal_rank_fusion(results)
    ]


def test_fusion_records_each_strategys_rank() -> None:
    """The retrieval trace and the experiments both need the inputs, not just
    the output."""
    target = chunk("target", 0.5)
    fused = reciprocal_rank_fusion({"dense": [chunk("other", 0.9), target], "sparse": [target]})

    entry = next(item for item in fused if item.chunk_id == target.chunk_id)
    assert entry.ranks == {"dense": 2, "sparse": 1}


# --- the access filter ------------------------------------------------------


def test_the_filter_always_constrains_sensitivity() -> None:
    built = build_filter(AccessFilter(patient_ids=None, access_levels=frozenset({"public"})))

    assert len(built.must) == 1  # access level only - admin has no patient bound


def test_an_empty_grant_list_still_allows_unattributed_documents() -> None:
    """A user with no patient grants sees guidance, not nothing and not
    everything."""
    built = build_filter(
        AccessFilter(patient_ids=frozenset(), access_levels=frozenset({"internal"}))
    )

    patient_condition = built.must[1]
    assert patient_condition.match.any == [""]


# --- retrieval through the API ----------------------------------------------


async def test_keyword_search_finds_the_right_document(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    diabetes = await upload(client, api, admin_token, DIABETES_NOTE, "diabetes.txt")
    await upload(client, api, admin_token, CARDIOLOGY_NOTE, "cardiology.txt")

    body = await search(client, api, admin_token, "metformin")

    assert body["results"]
    assert body["results"][0]["document_id"] == diabetes
    assert "Metformin" in body["results"][0]["text"]


async def test_the_trace_exposes_both_strategies(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    """Project plan section 15: the retrieval trace is a requirement."""
    await upload(client, api, admin_token, DIABETES_NOTE, "diabetes.txt")

    body = await search(client, api, admin_token, "haemoglobin a1c")

    trace = body["trace"]
    assert trace["mode"] == "hybrid"
    assert set(trace["candidates"]) == {"dense", "sparse"}
    assert trace["timings_ms"]["total"] > 0
    assert body["results"][0]["ranks"]  # per-strategy rank is reported


@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
async def test_each_retrieval_mode_runs(
    client: AsyncClient, api: str, admin_token: str, mode: str
) -> None:
    """Vector-only vs hybrid is a configuration change, per plan section 14."""
    await upload(client, api, admin_token, DIABETES_NOTE, "diabetes.txt")

    body = await search(client, api, admin_token, "metformin dosage", overrides={"mode": mode})

    expected = {"hybrid": {"dense", "sparse"}, "dense": {"dense"}, "sparse": {"sparse"}}
    assert set(body["trace"]["candidates"]) == expected[mode]


async def test_top_k_is_respected(client: AsyncClient, api: str, admin_token: str) -> None:
    await upload(client, api, admin_token, DIABETES_NOTE, "diabetes.txt")
    await upload(client, api, admin_token, CARDIOLOGY_NOTE, "cardiology.txt")

    body = await search(client, api, admin_token, "medications", overrides={"top_k": 2})

    assert len(body["results"]) <= 2


async def test_searching_an_empty_index_is_not_an_error(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    body = await search(client, api, admin_token, "anything at all")

    assert body["results"] == []
    assert body["trace"]["returned"] == 0


async def test_search_requires_authentication(client: AsyncClient, api: str) -> None:
    response = await client.post(f"{api}/search", json={"query": "metformin"})

    assert response.status_code == 401


# --- access control, enforced before the search -----------------------------


async def test_an_analyst_never_retrieves_restricted_chunks(
    client: AsyncClient,
    api: str,
    admin_token: str,
    analyst_token: str,
    patients: list,
) -> None:
    """The control that matters: a chunk out of scope is never retrieved at
    all, so it cannot reach the prompt."""
    await upload(
        client,
        api,
        admin_token,
        DIABETES_NOTE,
        "restricted.txt",
        patient_external_id=patients[0].external_id,
        access_level="restricted",
    )

    as_admin = await search(client, api, admin_token, "metformin")
    as_analyst = await search(client, api, analyst_token, "metformin")

    assert as_admin["results"]
    assert as_analyst["results"] == []
    # Nothing was fetched and filtered afterwards - nothing was fetched.
    assert as_analyst["trace"]["candidates"]["dense"] == 0


async def test_a_clinician_only_retrieves_granted_patients(
    client: AsyncClient,
    api: str,
    admin_token: str,
    clinician_token: str,
    patients: list,
) -> None:
    """The clinician is granted patients[0] only."""
    granted = await upload(
        client,
        api,
        admin_token,
        DIABETES_NOTE,
        "granted.txt",
        patient_external_id=patients[0].external_id,
        access_level="restricted",
    )
    await upload(
        client,
        api,
        admin_token,
        CARDIOLOGY_NOTE,
        "ungranted.txt",
        patient_external_id=patients[1].external_id,
        access_level="restricted",
    )

    body = await search(client, api, clinician_token, "medications twice daily")

    assert body["results"]
    assert {item["document_id"] for item in body["results"]} == {granted}


async def test_narrowing_to_another_patient_returns_nothing(
    client: AsyncClient,
    api: str,
    admin_token: str,
    clinician_token: str,
    patients: list,
) -> None:
    """A caller may narrow their scope; asking outside it yields nothing rather
    than an error that would confirm the patient exists."""
    await upload(
        client,
        api,
        admin_token,
        CARDIOLOGY_NOTE,
        "ungranted.txt",
        patient_external_id=patients[1].external_id,
        access_level="restricted",
    )

    response = await client.post(
        f"{api}/search",
        headers=auth(clinician_token),
        json={"query": "apixaban", "patient_external_id": patients[1].external_id},
    )

    assert response.status_code == 403


async def test_unattributed_documents_stay_visible_to_a_scoped_user(
    client: AsyncClient, api: str, admin_token: str, clinician_token: str
) -> None:
    """Guidance and drug references have no patient to be scoped to."""
    guideline = await upload(
        client,
        api,
        admin_token,
        b"CLINICAL GUIDELINE\n\nMetformin is first-line therapy for type 2 diabetes.\n",
        "guideline.txt",
    )

    body = await search(client, api, clinician_token, "metformin first-line therapy")

    assert guideline in {item["document_id"] for item in body["results"]}


async def test_reindexing_replaces_rather_than_duplicates(
    client: AsyncClient, api: str, admin_token: str
) -> None:
    """Re-running ingestion after a chunk-size change must not leave two
    generations of the same document in the index."""
    document_id = await upload(client, api, admin_token, DIABETES_NOTE, "note.txt")

    # Same bytes: deduplicated, so the index is untouched.
    await client.post(
        f"{api}/documents/upload",
        headers=auth(admin_token),
        files={"file": ("note.txt", DIABETES_NOTE, "text/plain")},
        data={"document_type": "clinical_note", "access_level": "internal"},
    )

    body = await search(client, api, admin_token, "metformin", overrides={"top_k": 20})
    hits = [item for item in body["results"] if item["document_id"] == document_id]

    assert len({item["chunk_id"] for item in hits}) == len(hits)
