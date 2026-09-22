"""Dense embeddings.

Two implementations behind one protocol:

* `FastEmbedProvider` - the real one. fastembed runs ONNX models on
  onnxruntime rather than torch, which keeps the image around a tenth of the
  size of a sentence-transformers install for the same `bge-small` weights.
* `HashEmbeddingProvider` - deterministic, dependency-free, no download. The
  test suite and CI run the *real* retrieval path (Qdrant, fusion, filters,
  reranking) against it, so the only thing not exercised is embedding quality.

A query and a document are embedded differently. `bge` models are trained with
an instruction prefix on the query side only; omitting it measurably degrades
recall, which is the kind of detail that silently costs you a few points and
never shows up as an error.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Protocol

import anyio.to_thread

from app.core.config import EmbeddingProviderName, Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# bge-family query instruction. Applied to queries, never to documents.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Deterministic pseudo-embeddings from hashed token n-grams.

    Not semantic - two paraphrases share no dimensions unless they share
    tokens. That is fine for what it is for: proving the plumbing (indexing,
    filtering, fusion, reranking, citations) without a model download. It is
    close to a random-projection bag of words, so it does behave like a vector
    index rather than degenerating to exact match.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return f"hash-{self._dimensions}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            # Sign from a separate byte, so a token can cancel as well as add.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return _normalise(vector)


class FastEmbedProvider:
    """ONNX embeddings via fastembed.

    The model loads lazily and once. Loading is ~1s of CPU and the first call
    downloads ~130MB, so doing it at import time would make the API slow to
    start and would couple container start-up to network availability.
    """

    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: object | None = None
        self._dimensions: int | None = None
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load()
        assert self._dimensions is not None
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> object:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                logger.info("embedding_model_loading", model=self._model_name)
                self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
                description = next(
                    entry
                    for entry in TextEmbedding.list_supported_models()
                    if entry["model"] == self._model_name
                )
                self._dimensions = int(description["dim"])
                logger.info(
                    "embedding_model_loaded",
                    model=self._model_name,
                    dimensions=self._dimensions,
                )
            return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await anyio.to_thread.run_sync(lambda: self._encode(texts))

    async def embed_query(self, text: str) -> list[float]:
        prefixed = f"{BGE_QUERY_PREFIX}{text}" if self._is_bge else text
        vectors = await anyio.to_thread.run_sync(lambda: self._encode([prefixed]))
        return vectors[0]

    @property
    def _is_bge(self) -> bool:
        return "bge" in self._model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [list(map(float, vector)) for vector in model.embed(texts)]  # type: ignore[attr-defined]


def _normalise(vector: list[float]) -> list[float]:
    """Unit length, so Qdrant's cosine distance behaves as a dot product."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector
    return [value / magnitude for value in vector]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider is EmbeddingProviderName.HASH:
        return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    return FastEmbedProvider(settings.embedding_model, cache_dir=str(settings.model_cache_dir))
