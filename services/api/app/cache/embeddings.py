"""Query-embedding cache.

A decorator over `EmbeddingProvider`, so it is wired by swapping what the app
builds rather than by teaching the retriever about caching.

Only *queries* are cached. Documents are embedded once, in batches, at ingest;
caching them would add a write per chunk for a hit rate of approximately zero,
and the deduplication in the document pipeline already stops the same file
being embedded twice.

The vector is stored as float64, not float32. Halving the bytes would be a real
saving - 3KB against 1.5KB per entry - and it would also mean a cached query
scored fractionally differently from an uncached one. That makes the cache a
source of retrieval variance, which is precisely what the Sprint 5 evaluation
cannot have: a metric that moves because an entry expired is a metric nobody
can act on.
"""

from __future__ import annotations

from array import array

from app.cache.backend import CacheBackend
from app.cache.keys import embedding_key
from app.observability.metrics import METRICS
from app.observability.trace import span
from app.providers.embedding import EmbeddingProvider

NAMESPACE = "embedding"
# 'd' is IEEE 754 double in native byte order. Both ends of this cache are the
# same deployment's API processes, so byte order is not in play; the decoded
# length is checked anyway, which is what catches a changed dimension.
TYPECODE = "d"


class CachingEmbeddingProvider:
    """Wraps a provider and memoises `embed_query` through the cache backend."""

    def __init__(
        self,
        inner: EmbeddingProvider,
        cache: CacheBackend,
        *,
        prefix: str,
        ttl_seconds: int,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._prefix = prefix
        self._ttl = ttl_seconds

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        cache_key = embedding_key(
            prefix=self._prefix,
            model=self._inner.model_name,
            dimensions=self._inner.dimensions,
            text=text,
        )

        with span("embed.query", model=self._inner.model_name) as record:
            cached = self._decode(await self._cache.get(cache_key))
            if cached is not None:
                record.attributes["cached"] = True
                METRICS.record_cache(NAMESPACE, "hit")
                return cached

            record.attributes["cached"] = False
            METRICS.record_cache(NAMESPACE, "miss")
            vector = await self._inner.embed_query(text)

        stored = await self._cache.set(
            cache_key, array(TYPECODE, vector).tobytes(), ttl_seconds=self._ttl
        )
        METRICS.record_cache(NAMESPACE, "store" if stored else "error")
        return vector

    def _decode(self, raw: bytes | None) -> list[float] | None:
        if not raw:
            return None
        values = array(TYPECODE)
        try:
            values.frombytes(raw)
        except ValueError:
            return None
        # A dimension change with an unchanged model name would otherwise
        # return a vector the index cannot search with.
        if len(values) != self._inner.dimensions:
            return None
        return list(values)
