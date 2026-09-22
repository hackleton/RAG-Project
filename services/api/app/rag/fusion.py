"""Reciprocal Rank Fusion.

    score(d) = sum over result lists L of  1 / (k + rank_L(d))

Fusion happens here rather than in Qdrant's own `FusionQuery` for one reason:
the experiments in project plan section 14 compare vector-only against hybrid
and need the per-strategy result lists separately, and the `/search` endpoint
has to show *why* a chunk ranked where it did. Both need the inputs to fusion,
not just its output. RRF is a dozen lines, so the cost of owning it is low.

**Why rank and not score.** Cosine similarity lives in roughly [0, 1]; BM25 is
unbounded and corpus-dependent. Adding or averaging them means the keyword side
silently dominates, and the balance shifts as the corpus grows. RRF throws the
magnitudes away and keeps only the ordering, which is the whole point: it needs
no tuning per corpus and no score normalisation that would itself need tuning.

`k` (default 60, from the original paper) damps the contribution of top ranks.
Larger `k` flattens the curve, so a result has to place well in several lists
rather than winning one outright.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.providers.vectorstore import ScoredChunk

DEFAULT_RRF_K = 60


@dataclass(slots=True)
class FusedChunk:
    chunk_id: uuid.UUID
    payload: dict[str, object]
    fused_score: float
    # Per-strategy rank (1-based) and raw score, kept for the retrieval trace.
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def strategies(self) -> list[str]:
        return sorted(self.ranks)


def reciprocal_rank_fusion(
    results: dict[str, list[ScoredChunk]], *, k: int = DEFAULT_RRF_K
) -> list[FusedChunk]:
    """Fuse named result lists. Input order within each list is its ranking."""
    fused: dict[uuid.UUID, FusedChunk] = {}

    for strategy, chunks in results.items():
        for rank, chunk in enumerate(chunks, start=1):
            entry = fused.get(chunk.chunk_id)
            if entry is None:
                entry = FusedChunk(
                    chunk_id=chunk.chunk_id, payload=dict(chunk.payload), fused_score=0.0
                )
                fused[chunk.chunk_id] = entry
            entry.fused_score += 1.0 / (k + rank)
            entry.ranks[strategy] = rank
            entry.scores[strategy] = chunk.score

    # Tie-break on chunk id so the ordering is deterministic. Without it, two
    # chunks with identical fused scores could swap between runs and make an
    # experiment look like it moved when nothing changed.
    return sorted(fused.values(), key=lambda item: (-item.fused_score, str(item.chunk_id)))
