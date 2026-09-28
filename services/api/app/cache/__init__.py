"""Caching.

Project plan section 7 names Redis for a "response/semantic cache and
rate-control support". What is built here is narrower than that, deliberately:

* **Query embeddings** - a pure function of (model, text). Safe to share
  between callers because it depends on nothing about who is asking.
* **Rankings** - chunk ids and scores, keyed on the caller's access scope and
  on the index generation. Never the text; that is re-read from SQL Server.
* **Rate-control counters** - the failed-login limiter Sprint 6 left in the
  process, which decision 38 said belonged here.

**Answers are not cached, and that is a decision rather than an omission.** See
decision 42: a response cache is the one that would help most against a 30s
generation, and it is also the one that stores generated text derived from
restricted documents outside the database, and the one where a stale entry
survives the document being corrected. The hit rate on free-text clinical
questions does not pay for either.
"""

from __future__ import annotations

from app.cache.backend import (
    CacheBackend,
    MemoryCache,
    NullCache,
    RedisCache,
    build_cache,
)
from app.cache.embeddings import CachingEmbeddingProvider
from app.cache.ranking import CachedRanking, RankingCache, config_digest

__all__ = [
    "CacheBackend",
    "CachedRanking",
    "CachingEmbeddingProvider",
    "MemoryCache",
    "NullCache",
    "RankingCache",
    "RedisCache",
    "build_cache",
    "config_digest",
]
