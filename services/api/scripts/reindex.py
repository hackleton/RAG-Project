"""Rebuild the vector index from SQL Server.

The database is the record and the index is derived, so the index can always be
thrown away and rebuilt. This script is what makes that claim true rather than
aspirational, and it is the loop the project plan's section 14 experiments run:

    # change EMBEDDING_MODEL or the chunking settings, then
    docker compose run --rm api python scripts/reindex.py --all

Use it after:
* changing the embedding model - existing vectors are in the old model's space
  and comparing them to new query vectors is meaningless;
* changing a document's access level or patient - the Qdrant payload carries a
  copy of both, and retrieval filters on the copy;
* an indexing failure left documents stuck at INDEXING.

    docker compose run --rm api python scripts/reindex.py            # pending only
    docker compose run --rm api python scripts/reindex.py --all      # everything
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.backend import build_cache
from app.cache.ranking import RankingCache
from app.core.config import get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.models import Document, DocumentStatus
from app.providers.embedding import build_embedding_provider
from app.providers.vectorstore import QdrantVectorStore
from app.rag.indexer import Indexer

# Anything that has been chunked but is not searchable.
PENDING = (DocumentStatus.INDEXING, DocumentStatus.CHUNKING)


async def reindex(session: AsyncSession, indexer: Indexer, *, everything: bool) -> int:
    query = select(Document).order_by(Document.created_at)
    if not everything:
        query = query.where(Document.status.in_(PENDING))

    documents = list(await session.scalars(query))
    if not documents:
        print("Nothing to reindex.")
        return 0

    for document in documents:
        result = await indexer.index_document(session, document)
        await session.commit()
        print(
            f"  {document.filename:<44} {document.status.value:<10} "
            f"{result.indexed:>3} indexed, {result.skipped} skipped"
        )
    return len(documents)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        dest="everything",
        help="Reindex every document, not just those not yet searchable.",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    embedder = build_embedding_provider(settings)
    store = QdrantVectorStore(settings)
    # The API caches rankings by index generation. A reindex from the command
    # line changes the index the API is serving, so it has to bump the same
    # counter - otherwise a running API keeps handing out the old ranking
    # until the TTL expires, and the only symptom is a query that stubbornly
    # misses a document somebody can see in the database.
    cache = build_cache(settings)
    ranking_cache = RankingCache(
        cache, prefix=settings.cache_prefix, ttl_seconds=settings.cache_ranking_ttl_seconds
    )

    init_engine(settings)
    try:
        # Touching `dimensions` loads the model, which is the slow part and the
        # step most likely to fail (a download). Better here than mid-run.
        print(
            f"Embedding with {embedder.model_name} "
            f"({embedder.dimensions}d) into '{store.collection}'"
        )
        async with get_sessionmaker()() as session:
            count = await reindex(
                session,
                Indexer(store=store, embedder=embedder, ranking_cache=ranking_cache),
                everything=arguments.everything,
            )
        indexed = await store.count()
    finally:
        await cache.close()
        await store.close()
        await dispose_engine()

    print(f"\nReindexed {count} document(s); {indexed} points in the collection.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
