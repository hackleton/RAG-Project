"""Qdrant vector store.

One collection holds every chunk, with two named vectors per point:

* `dense` - the embedding model's output, cosine distance.
* `sparse` - BM25 term weights, with `modifier: idf` so Qdrant applies the
  corpus IDF at query time (see `app/rag/sparse.py`).

**The security-critical part is the filter, not the search.** Patient scope and
sensitivity are a Qdrant `Filter` applied *by the engine, before* nearest
neighbours are selected - not a post-filter over results the model has already
been shown. Two reasons that distinction matters:

1. Post-filtering silently degrades recall: ask for the top 5 and throw three
   away and you have three, not five, and no way to know what you missed.
2. More importantly, the point of the control is that a chunk the caller may
   not see is never retrieved at all. A post-filter means the text was in the
   process, one bug away from the prompt.

The payload therefore duplicates `patient_id` and `access_level` from SQL
Server. That is denormalisation with a real cost - it has to be re-synced when
a document's access level changes - and `reindex.py` exists to do that.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.sparse import SparseVector

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


@dataclass(slots=True)
class ChunkPoint:
    """A chunk on its way into the index."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    patient_id: uuid.UUID | None
    access_level: str
    dense: list[float]
    sparse: SparseVector
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: uuid.UUID
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AccessFilter:
    """The caller's scope, as the index understands it.

    Built from `Principal`, never from anything a request body supplied.
    """

    # None means unrestricted (admin); a frozenset means exactly these.
    patient_ids: frozenset[uuid.UUID] | None
    access_levels: frozenset[str]
    # Narrowing filters from the request - not security controls.
    document_ids: frozenset[uuid.UUID] | None = None
    document_types: frozenset[str] | None = None


class VectorStore(Protocol):
    async def ensure_collection(self, *, dimensions: int) -> None: ...

    async def upsert(self, points: list[ChunkPoint]) -> None: ...

    async def delete_document(self, document_id: uuid.UUID) -> None: ...

    async def search_dense(
        self, vector: list[float], *, limit: int, access: AccessFilter
    ) -> list[ScoredChunk]: ...

    async def search_sparse(
        self, vector: SparseVector, *, limit: int, access: AccessFilter
    ) -> list[ScoredChunk]: ...

    async def count(self) -> int: ...

    async def health(self) -> bool: ...


class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self._collection = settings.qdrant_collection
        # Embedded Qdrant ignores payload indexes and warns if asked to make
        # them. Filtering still works; only the index-backed speed-up is absent,
        # which does not matter at test corpus size.
        self._supports_payload_indexes = not settings.qdrant_location
        self._client = (
            AsyncQdrantClient(location=settings.qdrant_location)
            if settings.qdrant_location
            else AsyncQdrantClient(
                url=str(settings.qdrant_url),
                api_key=settings.qdrant_api_key,
                timeout=int(settings.qdrant_timeout_seconds),
            )
        )

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self, *, dimensions: int) -> None:
        """Create the collection if absent. Safe to call on every start-up."""
        if await self._client.collection_exists(self._collection):
            return

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dimensions, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    # Qdrant maintains the corpus IDF and applies it at query
                    # time; the stored weights are term frequency only.
                    modifier=models.Modifier.IDF
                )
            },
        )
        # Payload indexes on the fields every query filters by. Without them
        # Qdrant scans the payload for each candidate, which is the difference
        # between a filtered search being cheap and being the slowest step.
        for field_name, schema in (
            ("patient_id", models.PayloadSchemaType.KEYWORD),
            ("access_level", models.PayloadSchemaType.KEYWORD),
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("document_type", models.PayloadSchemaType.KEYWORD),
        ):
            if not self._supports_payload_indexes:
                continue
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=schema,
            )
        logger.info("vector_collection_created", collection=self._collection, dimensions=dimensions)

    async def upsert(self, points: list[ChunkPoint]) -> None:
        if not points:
            return
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    # The chunk's own id, so the index and SQL Server agree on
                    # identity and a re-index overwrites rather than duplicates.
                    id=str(point.chunk_id),
                    vector={
                        DENSE_VECTOR: point.dense,
                        SPARSE_VECTOR: models.SparseVector(
                            indices=point.sparse.indices, values=point.sparse.values
                        ),
                    },
                    payload={
                        "document_id": str(point.document_id),
                        # "" rather than None: Qdrant's keyword index cannot
                        # match on a null, and non-patient documents still have
                        # to be findable.
                        "patient_id": str(point.patient_id) if point.patient_id else "",
                        "access_level": point.access_level,
                        **point.payload,
                    },
                )
                for point in points
            ],
            wait=True,
        )

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def search_dense(
        self, vector: list[float], *, limit: int, access: AccessFilter
    ) -> list[ScoredChunk]:
        return await self._query(query=vector, using=DENSE_VECTOR, limit=limit, access=access)

    async def search_sparse(
        self, vector: SparseVector, *, limit: int, access: AccessFilter
    ) -> list[ScoredChunk]:
        if not vector.indices:
            # Every query term was a stopword or unknown. An empty sparse query
            # matches nothing, which is correct - not an error.
            return []
        return await self._query(
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE_VECTOR,
            limit=limit,
            access=access,
        )

    async def _query(
        self, *, query: Any, using: str, limit: int, access: AccessFilter
    ) -> list[ScoredChunk]:
        """Run one filtered search.

        A collection that does not exist yet means nothing has been indexed -
        a normal state on a fresh deployment, not a failure. Returning no
        results is the honest answer; raising would turn "no documents" into a
        500 on the first query anyone makes.
        """
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query,
                using=using,
                limit=limit,
                query_filter=build_filter(access),
                with_payload=True,
            )
        except (UnexpectedResponse, ValueError) as exc:
            if not _is_missing_collection(exc):
                raise
            logger.info("vector_collection_absent", collection=self._collection)
            return []
        return _to_scored(response.points)

    async def count(self) -> int:
        result = await self._client.count(self._collection, exact=True)
        return int(result.count)

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception as exc:  # noqa: BLE001 - probes report, never raise
            logger.warning("vector_store_unreachable", error=str(exc))
            return False
        return True

    async def close(self) -> None:
        await self._client.close()


def build_filter(access: AccessFilter) -> models.Filter:
    """Translate a caller's scope into a pre-search Qdrant filter.

    Fails closed by construction: `access_levels` is always required, and an
    empty patient grant list produces a filter that matches only non-patient
    documents rather than one that matches everything.
    """
    must: list[models.Condition] = [
        models.FieldCondition(
            key="access_level",
            match=models.MatchAny(any=sorted(access.access_levels)),
        )
    ]

    if access.patient_ids is not None:
        # "" is a document with no patient - guidance, drug references - which
        # is not patient-scoped and stays visible.
        allowed = [""] + sorted(str(patient) for patient in access.patient_ids)
        must.append(models.FieldCondition(key="patient_id", match=models.MatchAny(any=allowed)))

    if access.document_ids is not None:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=sorted(str(item) for item in access.document_ids)),
            )
        )
    if access.document_types is not None:
        must.append(
            models.FieldCondition(
                key="document_type", match=models.MatchAny(any=sorted(access.document_types))
            )
        )

    return models.Filter(must=must)


def _is_missing_collection(exc: Exception) -> bool:
    """Distinguish "nothing indexed yet" from a real vector-store failure.

    Local Qdrant raises ValueError; the server returns 404. Matching on the
    message is unpleasant but the alternative - treating every ValueError as an
    empty index - would swallow real bugs.
    """
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404
    return "not found" in str(exc).lower()


def _to_scored(points: list[Any]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk_id=uuid.UUID(str(point.id)),
            score=float(point.score),
            payload=dict(point.payload or {}),
        )
        for point in points
    ]
