"""Caching: the key, the backends, and the boundary the key has to hold.

The security-relevant case is at the bottom - two callers with different
patient scopes asking the *same question* must not share a cached ranking. The
rest of this file is there so that when it fails, the reason is obvious.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.backend import MemoryCache, NullCache, RedisCache
from app.cache.embeddings import CachingEmbeddingProvider
from app.cache.keys import ranking_key, scope_digest, throttle_key
from app.cache.ranking import RankingCache, config_digest
from app.models.document import Document, DocumentChunk
from app.models.enums import AccessLevel
from app.providers.vectorstore import AccessFilter, admits
from app.rag.fusion import FusedChunk
from app.rag.retriever import RetrievalConfig
from app.security.throttle import LoginThrottle, SharedLoginThrottle
from tests.conftest import auth
from tests.test_retrieval import DIABETES_NOTE, search, upload

PATIENT_A = uuid.uuid4()
PATIENT_B = uuid.uuid4()

UNRESTRICTED = AccessFilter(patient_ids=None, access_levels=frozenset({"internal"}))
NOTHING = AccessFilter(patient_ids=frozenset(), access_levels=frozenset({"internal"}))
ONLY_A = AccessFilter(patient_ids=frozenset({PATIENT_A}), access_levels=frozenset({"internal"}))


# --- the key ----------------------------------------------------------------


def test_an_unrestricted_scope_is_not_the_same_key_as_an_empty_one() -> None:
    """`None` means "every patient"; `frozenset()` means "no patient at all".

    Both are falsy, which is exactly how they end up collapsed into one key by
    accident - and collapsing them keys the widest scope in the system to the
    same entry as the narrowest.
    """
    assert scope_digest(UNRESTRICTED) != scope_digest(NOTHING)


def test_every_field_of_the_access_filter_changes_the_key() -> None:
    """A field that does not reach the digest is a field the cache ignores.

    `scope_digest` is derived from `dataclasses.fields`, so this passes for a
    field added later too - which is the point. If someone adds a scoping
    dimension and this test still enumerates only the old ones, the failure is
    in the test, not in the digest.
    """
    base = AccessFilter(
        patient_ids=frozenset({PATIENT_A}),
        access_levels=frozenset({"internal"}),
        document_ids=frozenset({PATIENT_B}),
        document_types=frozenset({"clinical_note"}),
    )
    variants = [
        AccessFilter(
            frozenset({PATIENT_B}), base.access_levels, base.document_ids, base.document_types
        ),
        AccessFilter(
            base.patient_ids, frozenset({"restricted"}), base.document_ids, base.document_types
        ),
        AccessFilter(base.patient_ids, base.access_levels, None, base.document_types),
        AccessFilter(base.patient_ids, base.access_levels, base.document_ids, None),
    ]
    digests = {scope_digest(variant) for variant in variants}
    assert scope_digest(base) not in digests
    assert len(digests) == len(variants)


def test_the_ranking_key_separates_query_config_scope_and_generation() -> None:
    config = RetrievalConfig()

    def key(**changes: object) -> str:
        arguments: dict[str, object] = {
            "prefix": "hic",
            "query": "metformin dose",
            "config_digest": config_digest(config),
            "access": ONLY_A,
            "generation": 1,
        }
        return ranking_key(**(arguments | changes))  # type: ignore[arg-type]

    baseline = key()
    assert baseline == key()
    assert baseline != key(query="apixaban dose")
    assert baseline != key(access=UNRESTRICTED)
    assert baseline != key(generation=2)
    assert baseline != key(config_digest=config_digest(RetrievalConfig(top_k=10)))


def test_a_retrieval_setting_that_changes_the_ranking_changes_the_digest() -> None:
    """Generic over the dataclass, so a knob added later is covered."""
    base = RetrievalConfig()
    assert config_digest(base) == config_digest(RetrievalConfig())
    for changed in (
        RetrievalConfig(top_k=10),
        RetrievalConfig(mode="dense"),
        RetrievalConfig(rerank=False),
        RetrievalConfig(rrf_k=10),
        RetrievalConfig(candidate_pool=10),
    ):
        assert config_digest(base) != config_digest(changed)


def test_the_query_text_never_reaches_the_key() -> None:
    """Keys are digests. A cache dump must not read as a list of questions."""
    query = "does SYN-0001 still take metformin"
    built = ranking_key(
        prefix="hic",
        query=query,
        config_digest=config_digest(RetrievalConfig()),
        access=ONLY_A,
        generation=0,
    )
    assert "metformin" not in built
    assert "SYN-0001" not in built
    assert "syn-0001" not in throttle_key(prefix="hic", email="SYN-0001@x", ip=None)


# --- the backends -----------------------------------------------------------


async def test_a_memory_entry_expires() -> None:
    cache = MemoryCache()
    await cache.set("k", b"v", ttl_seconds=0)
    assert await cache.get("k") is None


async def test_the_memory_cache_is_bounded() -> None:
    """Unbounded is the failure mode a cache introduces into a long-lived process."""
    cache = MemoryCache(max_entries=8)
    for index in range(32):
        await cache.set(f"k{index}", b"v", ttl_seconds=60)
    assert await cache.get("k0") is None
    assert await cache.get("k31") == b"v"


async def test_an_increment_does_not_extend_its_own_window() -> None:
    """Otherwise a caller being throttled keeps resetting the window it is in."""
    cache = MemoryCache()
    assert await cache.increment("c", ttl_seconds=60) == 1
    assert await cache.increment("c", ttl_seconds=60) == 2
    # A second attempt with a shorter TTL must not shorten it either.
    assert await cache.increment("c", ttl_seconds=1) == 3


async def test_a_null_cache_misses_and_cannot_count() -> None:
    cache = NullCache()
    assert await cache.set("k", b"v", ttl_seconds=60) is False
    assert await cache.get("k") is None
    # None rather than 0: "this backend cannot answer" is what lets the login
    # limiter fall back to its in-process counter instead of reading an outage
    # as nobody having tried yet.
    assert await cache.increment("k", ttl_seconds=60) is None


async def test_an_unreachable_redis_opens_the_breaker_and_stops_calling() -> None:
    """Failing open is not enough; a dead cache must also stop costing time.

    Port 1 has nothing on it, so every call fails. After the threshold the
    breaker opens and subsequent calls return without touching the socket -
    which is what stops an outage adding a connect timeout to every request.
    """
    cache = RedisCache(
        "redis://127.0.0.1:1/0",
        timeout_seconds=0.05,
        breaker_failures=3,
        breaker_cooldown_seconds=30.0,
    )
    try:
        for _ in range(3):
            assert await cache.get("k") is None
        assert cache.breaker_open

        started = asyncio.get_running_loop().time()
        for _ in range(50):
            assert await cache.get("k") is None
        assert asyncio.get_running_loop().time() - started < 0.05
    finally:
        await cache.close()


# --- the embedding cache ----------------------------------------------------


class CountingEmbedder:
    def __init__(self, dimensions: int = 4, model: str = "test-model") -> None:
        self._dimensions = dimensions
        self._model = model
        self.calls = 0

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * self._dimensions for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3, 0.4][: self._dimensions]


async def test_a_repeated_query_is_embedded_once() -> None:
    inner = CountingEmbedder()
    cached = CachingEmbeddingProvider(inner, MemoryCache(), prefix="t", ttl_seconds=60)

    first = await cached.embed_query("metformin")
    second = await cached.embed_query("metformin")

    assert first == second
    assert inner.calls == 1


async def test_a_cached_vector_round_trips_exactly() -> None:
    """A cached query that scores differently from an uncached one would make
    the cache a source of retrieval variance - which the evaluation cannot have.
    """
    inner = CountingEmbedder()
    cached = CachingEmbeddingProvider(inner, MemoryCache(), prefix="t", ttl_seconds=60)

    direct = await inner.embed_query("metformin")
    await cached.embed_query("metformin")
    assert await cached.embed_query("metformin") == direct


async def test_another_model_does_not_read_this_model_s_vectors() -> None:
    backend = MemoryCache()
    first = CachingEmbeddingProvider(
        CountingEmbedder(model="model-a"), backend, prefix="t", ttl_seconds=60
    )
    second_inner = CountingEmbedder(model="model-b")
    second = CachingEmbeddingProvider(second_inner, backend, prefix="t", ttl_seconds=60)

    await first.embed_query("metformin")
    await second.embed_query("metformin")
    assert second_inner.calls == 1


async def test_a_vector_of_the_wrong_width_is_treated_as_a_miss() -> None:
    """A dimension change with an unchanged model name would otherwise return
    a vector the index cannot search with."""
    backend = MemoryCache()
    await CachingEmbeddingProvider(
        CountingEmbedder(dimensions=4), backend, prefix="t", ttl_seconds=60
    ).embed_query("metformin")

    narrower = CountingEmbedder(dimensions=2)
    provider = CachingEmbeddingProvider(narrower, backend, prefix="t", ttl_seconds=60)
    assert len(await provider.embed_query("metformin")) == 2
    assert narrower.calls == 1


# --- the ranking cache ------------------------------------------------------


def _fused(chunk_id: uuid.UUID) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        payload={"text": "ignored"},
        fused_score=0.0328,
        ranks={"dense": 1, "sparse": 3},
        scores={"dense": 0.81, "sparse": 4.2},
    )


async def test_a_ranking_round_trips_with_its_scores() -> None:
    cache = RankingCache(MemoryCache(), prefix="t", ttl_seconds=60)
    chunk_id = uuid.uuid4()
    config = RetrievalConfig()

    await cache.put(
        query="metformin",
        config=config,
        access=ONLY_A,
        generation=1,
        selected=[_fused(chunk_id)],
        rerank_scores={chunk_id: 2.5},
        candidates={"dense": 50, "sparse": 50},
        fused_candidates=73,
    )
    restored = await cache.get(query="metformin", config=config, access=ONLY_A, generation=1)

    assert restored is not None
    assert [chunk.chunk_id for chunk in restored.selected] == [chunk_id]
    assert restored.selected[0].fused_score == pytest.approx(0.0328)
    assert restored.selected[0].ranks == {"dense": 1, "sparse": 3}
    assert restored.rerank_scores == {chunk_id: 2.5}
    # Carried so a cache hit still produces a complete trace.
    assert restored.candidates == {"dense": 50, "sparse": 50}
    assert restored.fused_candidates == 73


async def test_a_new_generation_makes_the_previous_one_unreachable() -> None:
    backend = MemoryCache()
    cache = RankingCache(backend, prefix="t", ttl_seconds=60)
    config = RetrievalConfig()
    await cache.put(
        query="metformin",
        config=config,
        access=ONLY_A,
        generation=1,
        selected=[_fused(uuid.uuid4())],
        rerank_scores={},
        candidates={},
        fused_candidates=0,
    )
    assert await cache.get(query="metformin", config=config, access=ONLY_A, generation=1)
    assert await cache.get(query="metformin", config=config, access=ONLY_A, generation=2) is None


async def test_a_corrupt_entry_is_a_miss_rather_than_an_error() -> None:
    backend = MemoryCache()
    cache = RankingCache(backend, prefix="t", ttl_seconds=60)
    config = RetrievalConfig()
    await backend.set(
        ranking_key(
            prefix="t",
            query="metformin",
            config_digest=config_digest(config),
            access=ONLY_A,
            generation=0,
        ),
        b'{"selected": [{"chunk_id": "not-a-uuid"}]}',
        ttl_seconds=60,
    )
    assert await cache.get(query="metformin", config=config, access=ONLY_A, generation=0) is None


# --- the login limiter, now shared ------------------------------------------


async def test_the_shared_counter_refuses_before_the_local_one_would() -> None:
    """The shared count is what a second replica would see; either can refuse."""
    backend = MemoryCache()
    local = LoginThrottle(max_attempts=3, window_seconds=60)
    first = SharedLoginThrottle(backend, local, prefix="t", max_attempts=3, window_seconds=60)
    # A second replica: its own in-process counter, the same shared one.
    second = SharedLoginThrottle(
        backend,
        LoginThrottle(max_attempts=3, window_seconds=60),
        prefix="t",
        max_attempts=3,
        window_seconds=60,
    )

    for _ in range(3):
        await first.record_failure("a@example.com", "10.0.0.1")

    assert await first.check("a@example.com", "10.0.0.1") is False
    # The weakness decision 38 recorded: without the shared counter this replica
    # would happily grant three more attempts.
    assert await second.check("a@example.com", "10.0.0.1") is False


async def test_a_cache_outage_leaves_the_in_process_limit_running() -> None:
    """Degrading to Sprint 6's behaviour, not to no limit at all."""
    throttle = SharedLoginThrottle(
        NullCache(),
        LoginThrottle(max_attempts=2, window_seconds=60),
        prefix="t",
        max_attempts=2,
        window_seconds=60,
    )
    for _ in range(2):
        await throttle.record_failure("a@example.com", "10.0.0.1")
    assert await throttle.check("a@example.com", "10.0.0.1") is False


async def test_a_successful_sign_in_clears_both_counters() -> None:
    backend = MemoryCache()
    throttle = SharedLoginThrottle(
        backend,
        LoginThrottle(max_attempts=2, window_seconds=60),
        prefix="t",
        max_attempts=2,
        window_seconds=60,
    )
    await throttle.record_failure("a@example.com", "10.0.0.1")
    await throttle.reset("a@example.com", "10.0.0.1")
    assert await throttle.check("a@example.com", "10.0.0.1") is True
    assert await backend.get(throttle_key(prefix="t", email="a@example.com", ip="10.0.0.1")) is None


# --- through the API --------------------------------------------------------


async def test_a_repeated_search_is_served_from_the_ranking_cache(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    await upload(
        client,
        api,
        clinician_token,
        DIABETES_NOTE,
        "note.txt",
        patient_external_id="SYN-0001",
        access_level="restricted",
    )

    first = await search(client, api, clinician_token, "haemoglobin a1c result")
    second = await search(client, api, clinician_token, "haemoglobin a1c result")

    assert first["trace"]["cache"]["ranking"] == "miss"
    assert second["trace"]["cache"]["ranking"] == "hit"
    # A cached ranking is the same ranking. If this ever diverges, the cache is
    # changing what the system answers, which is not what a cache is for.
    assert [item["chunk_id"] for item in second["results"]] == [
        item["chunk_id"] for item in first["results"]
    ]
    assert [item["text"] for item in second["results"]] == [
        item["text"] for item in first["results"]
    ]


async def test_one_caller_s_cached_ranking_is_not_served_to_another(
    client: AsyncClient, api: str, clinician_token: str, analyst_token: str
) -> None:
    """The cache must not undo the pre-search access filter.

    The clinician is granted SYN-0001 and uploads a restricted note about them;
    the analyst is granted nothing. Same question, same wording, same moment -
    and the analyst must still retrieve nothing.
    """
    await upload(
        client,
        api,
        clinician_token,
        DIABETES_NOTE,
        "note.txt",
        patient_external_id="SYN-0001",
        access_level="restricted",
    )
    question = "haemoglobin a1c result"

    warmed = await search(client, api, clinician_token, question)
    assert warmed["results"], "the clinician should see their own patient's note"

    denied = await search(client, api, analyst_token, question)
    assert denied["results"] == []
    assert denied["trace"]["cache"]["ranking"] == "miss"


async def test_indexing_a_document_invalidates_the_cached_rankings(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    """A new document has to be able to win a query that was already asked."""
    question = "apixaban anticoagulation"
    assert (await search(client, api, clinician_token, question))["trace"]["cache"]["ranking"] == (
        "miss"
    )
    before = (await search(client, api, clinician_token, question))["trace"]["cache"]
    assert before["ranking"] == "hit"

    await upload(
        client,
        api,
        clinician_token,
        DIABETES_NOTE,
        "later.txt",
        patient_external_id="SYN-0001",
        access_level="restricted",
    )

    after = await search(client, api, clinician_token, question)
    assert after["trace"]["cache"]["generation"] > before["generation"]
    assert after["trace"]["cache"]["ranking"] == "miss"


async def test_the_readiness_probe_reports_the_cache_without_depending_on_it(
    client: AsyncClient,
) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["dependencies"]["cache"]["status"] == "ok"
    assert body["status"] == "ok"


async def test_an_authenticated_request_carries_its_component_timings(
    client: AsyncClient, api: str, clinician_token: str
) -> None:
    await upload(
        client,
        api,
        clinician_token,
        DIABETES_NOTE,
        "note.txt",
        patient_external_id="SYN-0001",
        access_level="restricted",
    )
    response = await client.post(
        f"{api}/chat", headers=auth(clinician_token), json={"question": "what is the a1c?"}
    )
    assert response.status_code == 200, response.text
    trace = response.json()["trace"]

    assert "llm.complete" in trace["components_ms"]
    assert any(record["name"].startswith("retrieval.") for record in trace["spans"])


# --- the record wins over the index -----------------------------------------


GUIDELINE = b"""CLINICAL GUIDELINE

BLOOD PRESSURE TARGETS

Target clinic blood pressure below 140/90 mmHg for most adults with hypertension.
"""


def test_admits_mirrors_the_pre_search_filter() -> None:
    """Same semantics as `build_filter`, including the two easy mistakes."""
    document_id = uuid.uuid4()

    def check(access: AccessFilter, **record: object) -> bool:
        fields: dict[str, object] = {
            "access_level": "internal",
            "patient_id": None,
            "document_id": document_id,
            "document_type": "guideline",
        }
        return admits(access, **(fields | record))  # type: ignore[arg-type]

    # A document belonging to no patient stays visible to a caller with no grants.
    assert check(NOTHING)
    # ...but a patient's document does not.
    assert not check(NOTHING, patient_id=PATIENT_A)
    assert check(ONLY_A, patient_id=PATIENT_A)
    assert not check(ONLY_A, patient_id=PATIENT_B)
    # Unrestricted means every patient, not no patient.
    assert check(UNRESTRICTED, patient_id=PATIENT_B)
    # The access level is always required.
    assert not check(UNRESTRICTED, access_level="restricted")


async def _tighten(session: AsyncSession, document_id: str) -> None:
    """Make a document restricted in the record only.

    No reindex and no generation bump: exactly the state a failed-open cache
    invalidation leaves behind, and the state an access change is in until
    somebody runs `reindex.py`.
    """
    identifier = uuid.UUID(document_id)
    await session.execute(
        update(Document)
        .where(Document.id == identifier)
        .values(access_level=AccessLevel.RESTRICTED)
    )
    await session.execute(
        update(DocumentChunk)
        .where(DocumentChunk.document_id == identifier)
        .values(access_level=AccessLevel.RESTRICTED)
    )
    await session.commit()


async def test_a_cached_ranking_cannot_serve_a_chunk_that_has_since_been_tightened(
    client: AsyncClient, api: str, clinician_token: str, analyst_token: str, session: AsyncSession
) -> None:
    document_id = await upload(client, api, clinician_token, GUIDELINE, "bp.txt")
    question = "blood pressure target"

    before = await search(client, api, analyst_token, question)
    assert any(item["document_id"] == document_id for item in before["results"])

    await _tighten(session, document_id)

    after = await search(client, api, analyst_token, question)
    # The stale entry is still there and still served - the invalidation is
    # what "failed". What must not survive is the text.
    assert after["trace"]["cache"]["ranking"] == "hit"
    assert not any(item["document_id"] == document_id for item in after["results"])


async def test_a_stale_index_payload_cannot_serve_one_either(
    client: AsyncClient, api: str, clinician_token: str, analyst_token: str, session: AsyncSession
) -> None:
    """The same guarantee without the cache, which predates Sprint 7.

    Qdrant's payload still says `internal`, so the pre-search filter lets the
    chunk through; the check at hydration is what refuses it.
    """
    document_id = await upload(client, api, clinician_token, GUIDELINE, "bp.txt")
    await _tighten(session, document_id)

    fresh = await search(client, api, analyst_token, "clinic blood pressure hypertension adults")
    assert fresh["trace"]["cache"]["ranking"] == "miss"
    assert not any(item["document_id"] == document_id for item in fresh["results"])
