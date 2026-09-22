"""BM25 sparse vectors.

The keyword half of hybrid retrieval. Written out rather than pulled from a
library for two reasons: it is about sixty lines and being able to explain
exactly what the keyword scorer does is worth more than a dependency, and every
off-the-shelf option (fastembed's `Qdrant/bm25`) needs a model download, which
would mean CI could not run the keyword path at all.

**How BM25 is split between here and Qdrant.** A sparse vector is
`{term_hash: weight}`. This module computes the *term-frequency* half of BM25:

    tf_weight = f * (k1 + 1) / (f + k1 * (1 - b + b * |d| / avgdl))

Qdrant supplies the other half. A collection whose sparse vector is configured
with `modifier: idf` multiplies each term's stored weight by the corpus IDF it
maintains itself at query time. That division matters: IDF depends on the whole
corpus, so computing it here would mean every new document invalidated every
stored vector. Qdrant keeps it current as documents are added.

The document length normalisation (`b`) does need an average document length,
which is a corpus statistic. It is taken from a running estimate rather than a
full pass: being slightly stale changes scores marginally, whereas re-embedding
the corpus on every insert is not viable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Standard BM25 constants. k1 controls term-frequency saturation, b how much
# document length is penalised.
BM25_K1 = 1.2
BM25_B = 0.75

# Assumed average chunk length in tokens, used for the length normalisation.
# The chunker targets 512 but section-strict chunking produces much shorter
# chunks in practice (see decisions entry 16), so this is measured from the
# corpus rather than assumed from the target.
DEFAULT_AVERAGE_LENGTH = 120.0

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Words that carry no retrieval signal but appear in nearly every clinical
# document. Deliberately short: over-aggressive stopword removal loses "no" in
# "no known drug allergies", which inverts the meaning of the chunk.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "there",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
    ]
)


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Qdrant's sparse format: parallel index and value arrays."""

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping decimals intact.

    "HbA1c 7.2%" -> ["hba1c", "7.2"]. Splitting the decimal would make a query
    for a specific lab value match every chunk containing a 7.
    """
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


def term_index(term: str) -> int:
    """Stable 32-bit index for a term.

    Qdrant sparse vectors are keyed by integer. Hashing avoids maintaining a
    vocabulary table that would have to stay in step with the index, at the
    cost of collisions - at 2^31 slots and a corpus of thousands of terms, the
    collision probability is negligible.
    """
    # Python's hash() is salted per process; this has to be stable across runs.
    digest = hashlib.blake2b(term.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def encode_document(text: str, *, average_length: float = DEFAULT_AVERAGE_LENGTH) -> SparseVector:
    """Term-frequency half of BM25. Qdrant's `idf` modifier supplies the rest."""
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts: dict[int, int] = {}
    for token in tokens:
        index = term_index(token)
        counts[index] = counts.get(index, 0) + 1

    length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / max(average_length, 1.0))
    indices, values = [], []
    for index, frequency in counts.items():
        indices.append(index)
        values.append(frequency * (BM25_K1 + 1) / (frequency + length_norm))
    return SparseVector(indices=indices, values=values)


def encode_query(text: str) -> SparseVector:
    """Queries are not length-normalised - only documents are.

    BM25 scores a document against a query; the query's own length does not
    change how well a document matches it. Each query term contributes 1.0,
    scaled by the IDF Qdrant applies.
    """
    seen: dict[int, float] = {}
    for token in tokenize(text):
        seen[term_index(token)] = 1.0
    return SparseVector(indices=list(seen), values=list(seen.values()))
