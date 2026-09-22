"""Retrieval and system metrics.

Pure functions over ranked lists. Nothing here touches a database, a model or
the clock, so the arithmetic can be checked against worked examples rather than
against whatever the pipeline happened to return that day - which is the only
way to know that a metric moving means the *system* moved.

Two definitions are worth stating plainly, because both are places where
evaluation harnesses quietly differ and then report incomparable numbers.

**Relevance is judged at document level, not chunk level.** A chunk id is
assigned at ingestion and changes every time the corpus is re-chunked, so a
dataset that named chunk ids would be invalidated by the very experiment it
exists to measure (project plan section 14 sweeps chunk size). Expected sources
name a filename, optionally narrowed to a section, and a retrieved chunk is
relevant when it came from one of them.

**Recall is over expected sources, not over retrieved chunks.** A document
splits into several chunks, so counting relevant *chunks* would let one
document retrieved three times score 3/1 recall. `recall_at_k` asks how many of
the expected sources are covered by the top K, which is the question the metric
is meant to answer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A source at the granularity the dataset can name stably.

    `section` is optional and narrows the match: a case about a lab value wants
    the RESULTS table specifically, while a case about a whole discharge
    summary is satisfied by any part of it.
    """

    filename: str
    section: str | None = None

    def matches(self, other: SourceRef) -> bool:
        """True when `other` (a retrieved chunk) satisfies this expectation."""
        if self.filename != other.filename:
            return False
        if self.section is None:
            return True
        return _normalise(self.section) == _normalise(other.section)


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Everything project plan section 13 asks of retrieval, for one case."""

    k: int
    retrieved: int
    relevant_retrieved: int
    expected: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    first_relevant_rank: int | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "k": self.k,
            "retrieved": self.retrieved,
            "relevant_retrieved": self.relevant_retrieved,
            "expected": self.expected,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
            "first_relevant_rank": self.first_relevant_rank,
        }


# --- primitives -----------------------------------------------------------


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Fraction of the top K that is relevant.

    The denominator is K, not `len(relevance)`. Returning fewer than K results
    is a retrieval failure like any other, and dividing by what was returned
    would reward a system that returns one good result and gives up.
    """
    if k <= 0:
        return 0.0
    return sum(1 for hit in relevance[:k] if hit) / k


def recall_at_k(covered: int, expected: int) -> float:
    """Fraction of the expected sources covered by the top K.

    A case with no expected sources - "the context does not support this, say
    so" - has nothing to recall. Scoring it 1.0 would inflate the corpus
    average with cases that never retrieved anything, so it is 0.0 here and
    excluded from the aggregate by `aggregate_ranking`.
    """
    if expected <= 0:
        return 0.0
    return min(covered, expected) / expected


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1 / rank of the first relevant result, 0 if there is none."""
    for position, hit in enumerate(relevance, start=1):
        if hit:
            return 1.0 / position
    return 0.0


def ndcg_at_k(relevance: Sequence[bool], k: int, expected: int) -> float:
    """Binary-gain NDCG at K, over a ranking of *distinct sources*.

    Binary rather than graded because the dataset judges a source relevant or
    not - there is no honest way to say one expected source is twice as
    relevant as another, and inventing grades would make the number look more
    precise than the judgements behind it.

    The ideal ranking puts every expected source in the first positions, so the
    denominator is `min(expected, k)` rather than the number of relevant results
    that happened to be retrieved. Using the latter is a common bug: a run that
    finds one of three expected sources and ranks it first would score 1.0.

    **`relevance` must be collapsed to one entry per distinct source before it
    reaches here**, which is why `score_ranking` does that rather than passing
    its chunk-level judgements straight in. Mixing the two granularities is the
    other common bug, and it is the one that does not announce itself: three
    chunks of one expected document contribute three gains against an ideal
    built from one, and the metric quietly returns more than 1.0.
    """
    if k <= 0 or expected <= 0:
        return 0.0
    gains = sum(
        1.0 / math.log2(position + 1) for position, hit in enumerate(relevance[:k], 1) if hit
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(expected, k) + 1))
    return gains / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; `q` in [0, 100].

    Written out rather than pulled from numpy: the harness is the only consumer,
    and a 70MB dependency for one function is not a trade worth making in an
    image that already carries onnxruntime.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * max(0.0, min(100.0, q)) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


# --- adapters -------------------------------------------------------------


def judge(retrieved: Sequence[SourceRef], expected: Sequence[SourceRef]) -> list[bool]:
    """Relevance of each retrieved chunk, in rank order."""
    return [any(want.matches(got) for want in expected) for got in retrieved]


def covered_sources(retrieved: Sequence[SourceRef], expected: Sequence[SourceRef]) -> int:
    """How many distinct expected sources appear anywhere in `retrieved`."""
    return sum(1 for want in expected if any(want.matches(got) for got in retrieved))


def collapse_to_documents(
    retrieved: Sequence[SourceRef], expected: Sequence[SourceRef]
) -> list[bool]:
    """Relevance of each *distinct document* in the ranking, in first-seen order.

    Retrieval returns chunks, and several of them routinely come from the same
    document. NDCG compares a ranking against the ideal ordering of the
    expected sources, so its ranking has to be counted in the same units the
    expectations are - documents - or the two sides of the ratio are not
    comparable.
    """
    relevance: dict[str, bool] = {}
    for got in retrieved:
        hit = any(want.matches(got) for want in expected)
        # `or` rather than assignment: a later chunk of the same document may
        # match a section-qualified expectation that the first one missed.
        relevance[got.filename] = relevance.get(got.filename, False) or hit
    return list(relevance.values())


def score_ranking(
    retrieved: Sequence[SourceRef], expected: Sequence[SourceRef], *, k: int
) -> RankingMetrics:
    """Every retrieval metric for one case, from one pass over the ranking.

    Precision and MRR are measured over chunks, because they answer chunk-level
    questions: how much of the context the model was given was useful, and how
    far down the context the first useful block sat. Recall and NDCG are
    measured over distinct documents, because that is the granularity the
    dataset can state expectations at. The mix is deliberate, and `RankingMetrics`
    carries both counts so a reader can see which is which.
    """
    window = list(retrieved)[:k]
    relevance = judge(window, expected)
    covered = covered_sources(window, expected)
    first = next((position for position, hit in enumerate(relevance, 1) if hit), None)

    return RankingMetrics(
        k=k,
        retrieved=len(window),
        relevant_retrieved=sum(relevance),
        expected=len(expected),
        precision_at_k=precision_at_k(relevance, k),
        recall_at_k=recall_at_k(covered, len(expected)),
        reciprocal_rank=reciprocal_rank(relevance),
        ndcg_at_k=ndcg_at_k(collapse_to_documents(window, expected), k, len(expected)),
        first_relevant_rank=first,
    )


def aggregate_ranking(cases: Iterable[RankingMetrics]) -> dict[str, float]:
    """Corpus-level retrieval metrics.

    Cases with no expected sources are excluded rather than scored zero. They
    exist to check that the system *declines* to answer, and averaging their
    unavoidable zeros into recall would make the retrieval numbers a function
    of how many refusal cases the dataset happens to contain.
    """
    scored = [case for case in cases if case.expected > 0]
    if not scored:
        return {}
    count = len(scored)
    return {
        "cases": count,
        "precision_at_k": _mean(case.precision_at_k for case in scored),
        "recall_at_k": _mean(case.recall_at_k for case in scored),
        "mrr": _mean(case.reciprocal_rank for case in scored),
        "ndcg_at_k": _mean(case.ndcg_at_k for case in scored),
        "hit_rate": _mean(1.0 if case.relevant_retrieved else 0.0 for case in scored),
    }


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    """P50/P95 and the mean, as project plan section 13 asks for."""
    if not latencies_ms:
        return {}
    return {
        "p50_ms": round(percentile(latencies_ms, 50), 1),
        "p95_ms": round(percentile(latencies_ms, 95), 1),
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 1),
        "max_ms": round(max(latencies_ms), 1),
    }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return round(sum(collected) / len(collected), 4) if collected else 0.0


def _normalise(section: str | None) -> str:
    return " ".join((section or "").split()).casefold()
