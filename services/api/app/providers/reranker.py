"""Cross-encoder reranking - the local stand-in for the AI Search semantic ranker.

Retrieval is a bi-encoder problem: query and document are embedded separately,
so the model never sees them together and similarity is a proxy. A cross-encoder
scores the pair jointly, which is far more accurate and far too slow to run over
a corpus - hence the standard shape, and the one the project plan describes:
retrieve a wide candidate pool cheaply, then rerank a few dozen precisely.

Project plan section 2 is explicit that the reranker must be *measured* rather
than assumed to help, which is why `ENABLE_RERANKER` is configuration and
`NoopReranker` is a first-class implementation rather than a test double: the
A/B is a settings change.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import anyio.to_thread

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int  # position in the input list
    score: float


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]: ...


class NoopReranker:
    """Keeps the fused order. The control arm of the reranker experiment."""

    @property
    def name(self) -> str:
        return "none"

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        # Descending scores preserve the incoming order under a stable sort.
        return [
            RerankResult(index=index, score=float(len(documents) - index))
            for index in range(len(documents))
        ]


class CrossEncoderReranker:
    """ONNX cross-encoder via fastembed, loaded lazily and once."""

    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: object | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self) -> object:
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("reranker_loading", model=self._model_name)
                self._model = TextCrossEncoder(
                    model_name=self._model_name, cache_dir=self._cache_dir
                )
                logger.info("reranker_loaded", model=self._model_name)
            return self._model

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        if not documents:
            return []

        def _score() -> list[float]:
            model = self._load()
            return [float(value) for value in model.rerank(query, documents)]  # type: ignore[attr-defined]

        scores = await anyio.to_thread.run_sync(_score)
        ranked = [RerankResult(index=index, score=score) for index, score in enumerate(scores)]
        return sorted(ranked, key=lambda item: -item.score)


def build_reranker(settings: Settings) -> Reranker:
    if not settings.enable_reranker:
        return NoopReranker()
    if settings.embedding_provider.value == "hash":
        # The hash embedder exists so tests and CI avoid model downloads; a
        # reranker download here would defeat that.
        return NoopReranker()
    return CrossEncoderReranker(settings.reranker_model, cache_dir=str(settings.model_cache_dir))
