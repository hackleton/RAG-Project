"""Ranking cache: the expensive half of retrieval, keyed on who is asking.

A retrieval is two vector searches, a fusion and a cross-encoder pass over the
top N candidates. Against the deterministic stack that is essentially the whole
request; against a real model it is small next to generation but still the part
that runs on every question, including the ones the model answers from cache-
warm context. It is also entirely a function of (query, configuration, access
scope, index contents), which makes it cacheable in a way an answer is not.

**Chunk ids and scores are cached. Text is not.** The retriever hydrates from
SQL Server on every request either way, so:

* No document text leaves the database for Redis. The cache holds identifiers
  and floats, which keeps the "synthetic or not, content lives in one audited
  place" property that the rest of the system has.
* An edited or deleted chunk is reflected immediately. The cache can only make
  the system return the same *ranking*, never stale content.

**Invalidation is a generation counter, not a key sweep.** Anything that
changes the index - an upload, a re-index, a deletion - increments one integer,
and every ranking key includes it, so the whole previous generation becomes
unreachable at once. The alternative, scanning Redis for keys to delete, is a
blocking operation whose cost grows with the keyspace, on the exact code path
that is already the slowest one in the system.

The cost of that choice: entries from an old generation are not deleted, they
are orphaned, and they occupy memory until their TTL expires. That is what the
TTL is for, and why it is short rather than long.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, fields
from typing import Any

from app.cache.backend import CacheBackend
from app.cache.keys import digest, generation_key, ranking_key
from app.core.logging import get_logger
from app.observability.metrics import METRICS
from app.providers.vectorstore import AccessFilter
from app.rag.fusion import FusedChunk

logger = get_logger(__name__)

NAMESPACE = "ranking"


def config_digest(config: Any) -> str:
    """Digest every field of a retrieval configuration.

    Generic over the dataclass rather than naming its fields, for the same
    reason `scope_digest` is: a tuning knob added later changes what a ranking
    means, and a hand-maintained list is one that eventually stops matching.
    """
    return digest(*(f"{item.name}={getattr(config, item.name)!r}" for item in fields(config)))


@dataclass(frozen=True, slots=True)
class CachedRanking:
    """A ranking restored from cache, in the shape the retriever hydrates."""

    selected: list[FusedChunk]
    rerank_scores: dict[uuid.UUID, float]
    candidates: dict[str, int] = field(default_factory=dict)
    fused_candidates: int = 0


class RankingCache:
    def __init__(self, cache: CacheBackend, *, prefix: str, ttl_seconds: int) -> None:
        self._cache = cache
        self._prefix = prefix
        self._ttl = ttl_seconds

    async def generation(self) -> int:
        """The current index generation. 0 when unknown, which is a valid key."""
        raw = await self._cache.get(generation_key(prefix=self._prefix))
        if raw is None:
            return 0
        try:
            return int(raw)
        except ValueError:
            return 0

    async def invalidate(self) -> int | None:
        """Move to a new generation. Called after the index changes."""
        # A long TTL: this key is the one thing that must outlive the entries
        # it versions. If it expires while old entries survive, the counter
        # restarts at 1 and can name a generation those entries already used.
        generation = await self._cache.increment(
            generation_key(prefix=self._prefix), ttl_seconds=30 * 24 * 3600
        )
        if generation is not None:
            logger.info("ranking_cache_invalidated", generation=generation)
        return generation

    async def get(
        self, *, query: str, config: Any, access: AccessFilter, generation: int
    ) -> CachedRanking | None:
        raw = await self._cache.get(self._key(query, config, access, generation))
        if raw is None:
            METRICS.record_cache(NAMESPACE, "miss")
            return None

        restored = self._decode(raw)
        METRICS.record_cache(NAMESPACE, "hit" if restored else "error")
        return restored

    async def put(
        self,
        *,
        query: str,
        config: Any,
        access: AccessFilter,
        generation: int,
        selected: list[FusedChunk],
        rerank_scores: dict[uuid.UUID, float],
        candidates: dict[str, int],
        fused_candidates: int,
    ) -> None:
        payload = {
            "candidates": candidates,
            "fused_candidates": fused_candidates,
            "selected": [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "fused_score": chunk.fused_score,
                    "ranks": chunk.ranks,
                    "scores": chunk.scores,
                    "rerank_score": rerank_scores.get(chunk.chunk_id),
                }
                for chunk in selected
            ],
        }
        stored = await self._cache.set(
            self._key(query, config, access, generation),
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            ttl_seconds=self._ttl,
        )
        METRICS.record_cache(NAMESPACE, "store" if stored else "error")

    # --- internals --------------------------------------------------------

    def _key(self, query: str, config: Any, access: AccessFilter, generation: int) -> str:
        return ranking_key(
            prefix=self._prefix,
            query=query,
            config_digest=config_digest(config),
            access=access,
            generation=generation,
        )

    def _decode(self, raw: bytes) -> CachedRanking | None:
        try:
            payload = json.loads(raw)
            selected = [
                FusedChunk(
                    chunk_id=uuid.UUID(item["chunk_id"]),
                    # The payload is only read by the reranker, which has
                    # already run by the time a ranking is cached. Hydration
                    # takes the authoritative text from SQL Server.
                    payload={},
                    fused_score=float(item["fused_score"]),
                    ranks={name: int(rank) for name, rank in item["ranks"].items()},
                    scores={name: float(score) for name, score in item["scores"].items()},
                )
                for item in payload["selected"]
            ]
            rerank_scores = {
                chunk.chunk_id: float(item["rerank_score"])
                for chunk, item in zip(selected, payload["selected"], strict=True)
                if item.get("rerank_score") is not None
            }
        except (KeyError, TypeError, ValueError) as exc:
            # An entry written by an older shape of this code. Treated as a
            # miss rather than an error the request sees - which is the whole
            # point of versioning the key prefix, but belt and braces.
            logger.warning("ranking_cache_undecodable", error=type(exc).__name__)
            return None

        return CachedRanking(
            selected=selected,
            rerank_scores=rerank_scores,
            candidates={name: int(count) for name, count in payload["candidates"].items()},
            fused_candidates=int(payload["fused_candidates"]),
        )
