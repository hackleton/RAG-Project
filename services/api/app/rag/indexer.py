"""Embedding and indexing - the step that takes a document from CHUNKING to READY.

Runs after chunking, over chunks already committed to SQL Server. That ordering
is deliberate: the database is the record and the index is derived, so indexing
can fail, be retried, or be re-run with different settings without any risk to
the source of truth.

Indexing a document **deletes its existing points first**. Without that, a
re-index after a chunk-size change leaves both generations in the index -
retrieval then returns overlapping chunks from two different chunkings, which
is a genuinely confusing failure mode and exactly the loop the section 14
experiments run.

Indexing is also the event that makes a cached ranking wrong, so it is where
the ranking cache's generation is bumped. Done after the upsert, not before: a
failed index leaves the previous generation intact and still correct, whereas
bumping first would throw away a warm cache to describe an index that never
changed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.ranking import RankingCache
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus
from app.providers.embedding import EmbeddingProvider
from app.providers.vectorstore import ChunkPoint, VectorStore
from app.rag import sparse

logger = get_logger(__name__)

# Chunks per embedding call and per upsert. Large enough to amortise the model
# call, small enough that one document does not hold a large payload in memory.
BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class IndexResult:
    document_id: uuid.UUID
    indexed: int
    skipped: int
    average_chunk_length: float


class Indexer:
    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: EmbeddingProvider,
        ranking_cache: RankingCache | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._ranking_cache = ranking_cache

    async def index_document(self, session: AsyncSession, document: Document) -> IndexResult:
        await self._store.ensure_collection(dimensions=self._embedder.dimensions)

        chunks = list(
            await session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            )
        )
        if not chunks:
            document.status = DocumentStatus.FAILED
            document.status_detail = "Nothing to index: the document has no chunks."
            return IndexResult(document.id, indexed=0, skipped=0, average_chunk_length=0.0)

        # Replace, never append (see the module docstring).
        await self._store.delete_document(document.id)

        average_length = await self._average_chunk_length(session)
        indexed = skipped = 0

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [self._embedding_text(chunk, document) for chunk in batch]
            vectors = await self._embedder.embed_documents(texts)

            points = []
            for chunk, text, vector in zip(batch, texts, vectors, strict=True):
                sparse_vector = sparse.encode_document(text, average_length=average_length)
                if not sparse_vector.indices and not any(vector):
                    # Nothing to match on in either space.
                    skipped += 1
                    continue
                points.append(
                    ChunkPoint(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        patient_id=chunk.patient_id,
                        access_level=chunk.access_level.value,
                        dense=vector,
                        sparse=sparse_vector,
                        payload={
                            "document_type": document.document_type,
                            "filename": document.filename,
                            "section": chunk.section or "",
                            "page_number": chunk.page_number or 0,
                            "chunk_index": chunk.chunk_index,
                            # Carried so the reranker can score without a second
                            # database round trip. The authoritative copy stays
                            # in SQL Server; retrieval re-hydrates from there.
                            "text": chunk.text,
                        },
                    )
                )
                chunk.embedding_reference = str(chunk.id)
                chunk.embedding_model = self._embedder.model_name
                indexed += 1

            await self._store.upsert(points)

        document.status = DocumentStatus.READY
        document.status_detail = None
        # A Python value, not func.now(): assigning a SQL expression leaves
        # the attribute deferred, so reading it back needs a round trip the
        # response serialiser cannot make.
        document.indexed_at = datetime.now(UTC)
        await session.flush()

        if self._ranking_cache is not None and indexed:
            await self._ranking_cache.invalidate()

        logger.info(
            "document_indexed",
            document_id=str(document.id),
            indexed=indexed,
            skipped=skipped,
            model=self._embedder.model_name,
        )
        return IndexResult(
            document_id=document.id,
            indexed=indexed,
            skipped=skipped,
            average_chunk_length=average_length,
        )

    def _embedding_text(self, chunk: DocumentChunk, document: Document) -> str:
        """Prefix the chunk with its section heading before embedding.

        A chunk reading only "7.4% (reference 4.0-5.6%)" is near-meaningless on
        its own; with "LABORATORY RESULTS" in front of it, it embeds near
        queries about lab results and matches the keyword "laboratory" that the
        chunk body never contains. The stored text is untouched - this affects
        only what is embedded.
        """
        if chunk.section:
            return f"{chunk.section}\n{chunk.text}"
        return chunk.text

    async def _average_chunk_length(self, session: AsyncSession) -> float:
        """Corpus average chunk length, for BM25 length normalisation."""
        average = await session.scalar(select(func.avg(DocumentChunk.token_count)))
        return float(average) if average else sparse.DEFAULT_AVERAGE_LENGTH
