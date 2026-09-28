"""Search and chat payloads."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.retriever import RetrievedChunk
from app.rag.service import Citation


class RetrievalOverrides(BaseModel):
    """Per-request retrieval settings.

    Exposed so an experiment can sweep a parameter without redeploying, which
    is what project plan section 14 needs. Deliberately bounded: these are
    performance knobs, and an unbounded `candidate_pool` is a cheap way for a
    caller to make the service do a lot of work.
    """

    model_config = ConfigDict(extra="forbid")

    top_k: int | None = Field(default=None, ge=1, le=20)
    candidate_pool: int | None = Field(default=None, ge=1, le=200)
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    mode: Literal["hybrid", "dense", "sparse"] | None = None
    rerank: bool | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # Narrows the caller's existing scope; it can never widen it. Bounded like
    # the agent's, because a refusal writes it into a 128-character audit column.
    patient_external_id: str | None = Field(default=None, max_length=64)
    overrides: RetrievalOverrides | None = None


class SourceRead(BaseModel):
    """A retrieved chunk with its provenance and every score behind its rank."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    document_type: str
    section: str | None
    page_number: int | None
    chunk_index: int
    text: str
    access_level: str
    fused_score: float
    rerank_score: float | None
    # Rank and raw score per strategy: {"dense": 3, "sparse": 11}.
    ranks: dict[str, int]
    scores: dict[str, float]
    # Non-empty when the filter removed planted instructions from `text`. Names
    # the signals, never the payload - see `app.security.injection.summarise`.
    injection_signals: list[str] = Field(default_factory=list)

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> SourceRead:
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.document_filename,
            document_type=chunk.document_type,
            section=chunk.section,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            access_level=chunk.access_level,
            fused_score=chunk.fused_score,
            rerank_score=chunk.rerank_score,
            ranks=chunk.ranks,
            scores=chunk.scores,
            injection_signals=chunk.injection_signals,
        )


class SearchResponse(BaseModel):
    query: str
    results: list[SourceRead]
    # Candidate counts, per-strategy timings, reranker used, fusion parameters.
    trace: dict[str, Any]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: uuid.UUID | None = None
    patient_external_id: str | None = Field(default=None, max_length=64)
    overrides: RetrievalOverrides | None = None


class CitationRead(BaseModel):
    number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    section: str | None
    page_number: int | None
    excerpt: str
    score: float | None

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationRead:
        return cls(
            number=citation.number,
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            filename=citation.filename,
            section=citation.section,
            page_number=citation.page_number,
            excerpt=citation.excerpt,
            score=citation.score,
        )


class ChatUsage(BaseModel):
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    estimated_cost_usd: float | None


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[CitationRead]
    sources: list[SourceRead]
    trace: dict[str, Any]
    usage: ChatUsage
