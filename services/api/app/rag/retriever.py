"""Hybrid retrieval.

    query
      -> access filter built from the Principal (never from the request body)
      -> ranking cache, keyed on that filter          [hit: skip to hydrate]
      -> dense search  +  sparse BM25 search, each over the candidate pool
      -> RRF fusion
      -> cross-encoder rerank
      -> top K, hydrated from SQL Server
      -> injection filter

Four things are deliberate and worth stating.

**The chunk text returned to the caller comes from SQL Server, not from the
Qdrant payload.** The index is derived data; the database is the record. If the
two disagree - a stale re-index, a partial upsert - the answer is grounded in
what the database says, and the discrepancy shows up as a chunk that ranked but
could not be hydrated, which is logged.

**Every retrieval carries a trace.** Candidate counts per strategy, each chunk's
rank in each list, its fused score and its reranker score. Project plan section
15 requires it, `/search` exposes it, and the Sprint 5 evaluation reads it -
none of which is possible if fusion is a black box inside the engine.

**The injection filter runs here, not in the callers.** Chat, `/search`, the
agent's document and knowledge tools and the evaluation harness all retrieve
through this class, so putting the filter at the exit is the only placement no
caller can forget - the same argument that puts tool authorisation inside
`ToolRegistry.dispatch`. It runs last, after hydration, because it edits the
text a prompt will see and must not change what ranked. The consequence is
stated in `app.security.injection`: planted text still influences *ranking*,
because it was embedded and indexed before anyone looked at it.

**The ranking cache is keyed on the access filter, and hydration still runs.**
Skipping the searches is safe only because the caller's scope is inside the
key - a cache keyed on the query alone would let the first caller's results be
served to the next one, which is the pre-search filter defeated by an
optimisation. Only chunk ids and scores are cached; the text still comes from
SQL Server on every request, so the database remains the record and an edited
chunk is never served stale. See `app.cache.ranking`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.ranking import CachedRanking, RankingCache
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk
from app.models.enums import AuditAction
from app.observability.trace import span
from app.providers.embedding import EmbeddingProvider
from app.providers.reranker import Reranker
from app.providers.vectorstore import AccessFilter, ScoredChunk, VectorStore, admits
from app.rag import sparse
from app.rag.fusion import FusedChunk, reciprocal_rank_fusion
from app.security import injection
from app.security.audit import record_audit
from app.security.principal import Principal

logger = get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 5
    candidate_pool: int = 50
    rrf_k: int = 60
    mode: str = "hybrid"  # hybrid | dense | sparse
    rerank: bool = True
    # Only the top N fused candidates are reranked. A cross-encoder is O(n) in
    # model calls, so reranking the whole pool to return five results is the
    # single largest avoidable cost in the request.
    rerank_top_n: int = 30
    rerank_score_threshold: float | None = None
    # Deliberately absent from `with_overrides`: a request may narrow its own
    # retrieval, it may not switch a security control off.
    filter_injections: bool = True

    def with_overrides(self, overrides: object | None) -> RetrievalConfig:
        """Apply a request's retrieval overrides on top of the deployed defaults.

        Only fields the caller actually set are applied, so an override object
        is a patch rather than a replacement. Bounds are enforced by the schema
        (`RetrievalOverrides`), not here - by this point the values are valid.
        """
        if overrides is None:
            return self
        changes = {
            field_name: value
            for field_name in ("top_k", "candidate_pool", "rrf_k", "mode", "rerank")
            if (value := getattr(overrides, field_name, None)) is not None
        }
        return replace(self, **changes) if changes else self


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk that survived to the answer, with its full provenance."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    patient_id: uuid.UUID | None
    document_filename: str
    document_type: str
    section: str | None
    page_number: int | None
    chunk_index: int
    text: str
    access_level: str
    fused_score: float
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    # Signal names that fired on this chunk's stored text. `text` above is the
    # filtered version; the unedited chunk stays readable through the document
    # endpoints, which are a deliberate request rather than a retrieval.
    injection_signals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    trace: dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        config: RetrievalConfig,
        ranking_cache: RankingCache | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._config = config
        self._ranking_cache = ranking_cache

    async def retrieve(
        self,
        session: AsyncSession,
        principal: Principal,
        query: str,
        *,
        patient_id: uuid.UUID | None = None,
        document_ids: frozenset[uuid.UUID] | None = None,
        document_types: frozenset[str] | None = None,
        exclude_patient_documents: bool = False,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        config = config or self._config
        started = time.perf_counter()

        access = self._access_filter(
            principal,
            patient_id=patient_id,
            document_ids=document_ids,
            document_types=document_types,
            exclude_patient_documents=exclude_patient_documents,
        )

        cached, generation = await self._cached_ranking(query, config, access)
        timings: dict[str, float] = {}

        if cached is not None:
            selected = cached.selected
            rerank_scores = cached.rerank_scores
            candidates = cached.candidates
            fused_candidates = cached.fused_candidates
        else:
            results, timings = await self._search(query, config, access)

            with span("retrieval.fuse", strategies=len(results)):
                fused = reciprocal_rank_fusion(results, k=config.rrf_k)

            rerank_started = time.perf_counter()
            with span("retrieval.rerank", candidates=min(len(fused), config.rerank_top_n)):
                ordered, rerank_scores = await self._rerank(query, fused, config)
            timings["rerank"] = round((time.perf_counter() - rerank_started) * 1000, 2)

            selected = ordered[: config.top_k]
            candidates = {name: len(items) for name, items in results.items()}
            fused_candidates = len(fused)
            await self._store_ranking(
                query, config, access, generation, selected, rerank_scores, candidates, len(fused)
            )

        with span("retrieval.hydrate", selected=len(selected)):
            chunks = await self._hydrate(session, selected, rerank_scores, access)
        with span("retrieval.screen", chunks=len(chunks)):
            screening = await self._screen(session, principal, chunks, config, query)

        trace = {
            "query_terms": len(sparse.tokenize(query)),
            "mode": config.mode,
            "candidate_pool": config.candidate_pool,
            "top_k": config.top_k,
            "rrf_k": config.rrf_k,
            "reranker": self._reranker.name if config.rerank else "disabled",
            "candidates": candidates,
            "fused_candidates": fused_candidates,
            "returned": len(chunks),
            "injection": screening,
            "cache": {
                "ranking": (
                    "disabled"
                    if self._ranking_cache is None
                    else ("hit" if cached is not None else "miss")
                ),
                "generation": generation,
            },
            "timings_ms": {
                **timings,
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
        }
        logger.info("retrieval", **trace)
        return RetrievalResult(chunks=chunks, trace=trace)

    # --- steps ------------------------------------------------------------

    def _access_filter(
        self,
        principal: Principal,
        *,
        patient_id: uuid.UUID | None,
        document_ids: frozenset[uuid.UUID] | None,
        document_types: frozenset[str] | None,
        exclude_patient_documents: bool = False,
    ) -> AccessFilter:
        """Build the pre-search filter from the caller's own scope.

        A caller may *narrow* to one patient; they can never widen. If they ask
        for a patient outside their grants the result is an empty scope, not an
        error - the same as asking for a patient with no documents, so the
        response does not disclose which patients exist.
        """
        scope = principal.accessible_patient_ids
        if patient_id is not None:
            scope = frozenset({patient_id}) if scope is None or patient_id in scope else frozenset()
        if exclude_patient_documents:
            # An empty grant set matches only documents with no patient at all
            # (see `build_filter`), which is exactly reference material. This
            # narrows every caller including an admin, so it is safe to set
            # from a tool's arguments.
            scope = frozenset()

        return AccessFilter(
            patient_ids=scope,
            access_levels=frozenset(level.value for level in principal.visible_access_levels),
            document_ids=document_ids,
            document_types=document_types,
        )

    async def _cached_ranking(
        self, query: str, config: RetrievalConfig, access: AccessFilter
    ) -> tuple[CachedRanking | None, int]:
        """Look for a ranking this caller's scope is entitled to reuse.

        Two reads rather than one: the generation counter, then the entry keyed
        on it. That is the price of invalidating a whole generation with a
        single INCR instead of scanning Redis for keys to delete, and it buys
        two vector searches and a cross-encoder pass.
        """
        if self._ranking_cache is None:
            return None, 0
        with span("retrieval.cache") as record:
            generation = await self._ranking_cache.generation()
            cached = await self._ranking_cache.get(
                query=query, config=config, access=access, generation=generation
            )
            record.attributes["hit"] = cached is not None
            record.attributes["generation"] = generation
        return cached, generation

    async def _store_ranking(
        self,
        query: str,
        config: RetrievalConfig,
        access: AccessFilter,
        generation: int,
        selected: list[FusedChunk],
        rerank_scores: dict[uuid.UUID, float],
        candidates: dict[str, int],
        fused_candidates: int,
    ) -> None:
        if self._ranking_cache is None:
            return
        await self._ranking_cache.put(
            query=query,
            config=config,
            access=access,
            generation=generation,
            selected=selected,
            rerank_scores=rerank_scores,
            candidates=candidates,
            fused_candidates=fused_candidates,
        )

    async def _search(
        self, query: str, config: RetrievalConfig, access: AccessFilter
    ) -> tuple[dict[str, list[ScoredChunk]], dict[str, float]]:
        results: dict[str, list[ScoredChunk]] = {}
        timings: dict[str, float] = {}

        if config.mode in ("hybrid", "dense"):
            started = time.perf_counter()
            with span("retrieval.dense", pool=config.candidate_pool) as record:
                vector = await self._embedder.embed_query(query)
                results[DENSE] = await self._store.search_dense(
                    vector, limit=config.candidate_pool, access=access
                )
                record.attributes["candidates"] = len(results[DENSE])
            timings["dense"] = round((time.perf_counter() - started) * 1000, 2)

        if config.mode in ("hybrid", "sparse"):
            started = time.perf_counter()
            with span("retrieval.sparse", pool=config.candidate_pool) as record:
                results[SPARSE] = await self._store.search_sparse(
                    sparse.encode_query(query), limit=config.candidate_pool, access=access
                )
                record.attributes["candidates"] = len(results[SPARSE])
            timings["sparse"] = round((time.perf_counter() - started) * 1000, 2)

        return results, timings

    async def _rerank(
        self, query: str, fused: list[FusedChunk], config: RetrievalConfig
    ) -> tuple[list[FusedChunk], dict[uuid.UUID, float]]:
        if not config.rerank or not fused:
            return fused, {}

        # Rerank a bounded window; everything below it keeps its fused order.
        window = fused[: config.rerank_top_n]
        tail = fused[config.rerank_top_n :]

        texts = [_rerank_text(chunk) for chunk in window]
        ranked = await self._reranker.rerank(query, texts)

        scores = {window[item.index].chunk_id: item.score for item in ranked}
        ordered = [window[item.index] for item in ranked] + tail

        if config.rerank_score_threshold is not None:
            kept = [
                chunk
                for chunk in ordered
                if scores.get(chunk.chunk_id, config.rerank_score_threshold)
                >= config.rerank_score_threshold
            ]
            # Never return nothing because the threshold was set too high; a
            # weak answer the user can judge beats a silent empty result.
            ordered = kept or ordered[:1]

        return ordered, scores

    async def _screen(
        self,
        session: AsyncSession,
        principal: Principal,
        chunks: list[RetrievedChunk],
        config: RetrievalConfig,
        query: str,
    ) -> dict[str, Any]:
        """Redact planted instructions out of the chunks, in place.

        The question is scanned too, but never edited. A question cannot
        escalate anything on its own - the filter and the tool gate see to that
        - and rewriting what the user typed would break legitimate queries
        about a document that happens to contain such text. It is recorded
        because an operator wants to know it was asked.
        """
        if not config.filter_injections:
            return {"enabled": False}

        asked = injection.scan(query)
        scans: list[injection.Scan] = []
        redactions = 0
        for chunk in chunks:
            result = injection.sanitise(chunk.text)
            chunk.text = result.text
            chunk.injection_signals = result.scan.signals if result.scan.flagged else []
            scans.append(result.scan)
            redactions += result.redactions

        summary = injection.summarise(scans) | {
            "enabled": True,
            "redactions": redactions,
            "query_signals": asked.signals if asked.flagged else [],
        }
        if summary["flagged"] or asked.flagged:
            # Written on the caller's transaction, so the record commits with
            # the answer it belongs to. Signal names only - putting the matched
            # sentence in an audit row would store the payload the filter just
            # removed.
            await record_audit(
                session,
                action=AuditAction.INJECTION_BLOCKED,
                principal=principal,
                resource_type="retrieval",
                allowed=True,
                details={
                    "flagged_chunks": summary["flagged"],
                    "redactions": redactions,
                    "signals": summary["signals"],
                    "query_signals": summary["query_signals"],
                    "documents": sorted(
                        {chunk.document_filename for chunk in chunks if chunk.injection_signals}
                    ),
                },
            )
            logger.warning(
                "injection_filtered",
                flagged=summary["flagged"],
                redactions=redactions,
                query_flagged=asked.flagged,
            )
        return summary

    async def _hydrate(
        self,
        session: AsyncSession,
        selected: list[FusedChunk],
        rerank_scores: dict[uuid.UUID, float],
        access: AccessFilter,
    ) -> list[RetrievedChunk]:
        """Load the authoritative text and document metadata from SQL Server.

        And re-apply the caller's scope to what the database says, not to what
        the index said. The pre-search filter ran over Qdrant's payload copy of
        `access_level` and `patient_id`; a ranking can outlive that copy - a
        cached one whose invalidation failed open, or an index not yet re-synced
        after a document was tightened. On a consistent index this drops
        nothing, which is why it costs no recall; on an inconsistent one it is
        the difference between a stale ranking and a disclosure.
        """
        if not selected:
            return []

        chunk_ids = [chunk.chunk_id for chunk in selected]
        rows = (
            await session.execute(
                select(DocumentChunk, Document)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.id.in_(chunk_ids))
            )
        ).all()
        by_id = {chunk.id: (chunk, document) for chunk, document in rows}

        hydrated: list[RetrievedChunk] = []
        for fused in selected:
            record = by_id.get(fused.chunk_id)
            if record is None:
                # The index knows about a chunk the database does not. A stale
                # index is a real operational state; drop it and say so.
                logger.warning("retrieved_chunk_missing", chunk_id=str(fused.chunk_id))
                continue
            chunk, document = record
            if not admits(
                access,
                access_level=chunk.access_level.value,
                patient_id=chunk.patient_id,
                document_id=document.id,
                document_type=document.document_type,
            ):
                # Ids only: the point of dropping it is that its text must not
                # travel any further, including into a log line.
                logger.warning(
                    "retrieved_chunk_not_admitted",
                    chunk_id=str(fused.chunk_id),
                    document_id=str(document.id),
                )
                continue
            hydrated.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    patient_id=chunk.patient_id,
                    document_filename=document.filename,
                    document_type=document.document_type,
                    section=chunk.section,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    access_level=chunk.access_level.value,
                    fused_score=fused.fused_score,
                    rerank_score=rerank_scores.get(fused.chunk_id),
                    ranks=dict(fused.ranks),
                    scores=dict(fused.scores),
                )
            )
        return hydrated


def _rerank_text(chunk: FusedChunk) -> str:
    """The text the cross-encoder scores.

    Must match what the embedder saw, section heading included. Without it the
    two stages disagree about what a chunk *is*: the dense side ranks the chunk
    with "CURRENT MEDICATIONS" in front of it, while the reranker scores the
    bare "- Metformin 1000 mg twice daily", which carries none of the signal a
    query like "current medications" is asking for.
    """
    section = str(chunk.payload.get("section") or "")
    text = str(chunk.payload.get("text", ""))
    return f"{section}\n{text}" if section else text
