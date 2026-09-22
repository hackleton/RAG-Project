"""The retrieval arithmetic, against worked examples.

These are the tests that have to be right by inspection. Every other number in
Sprint 5 is computed from these functions, so a quiet error here would not
surface as a failure - it would surface as a metric that looks plausible and
is wrong, which is worse than no metric at all.

Each case therefore states the expected value *and* how it is arrived at.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    SourceRef,
    aggregate_ranking,
    collapse_to_documents,
    covered_sources,
    judge,
    latency_summary,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)


def refs(*names: str) -> list[SourceRef]:
    return [SourceRef(filename=name) for name in names]


# --- matching -------------------------------------------------------------


def test_an_expectation_without_a_section_matches_any_part_of_the_document() -> None:
    expected = SourceRef(filename="note.txt")
    assert expected.matches(SourceRef(filename="note.txt", section="VITAL SIGNS"))
    assert expected.matches(SourceRef(filename="note.txt", section=None))


def test_an_expectation_with_a_section_matches_only_that_section() -> None:
    expected = SourceRef(filename="note.txt", section="Vital Signs")

    # Case and surrounding whitespace are a formatting detail of the heading
    # detector, not a difference in which section was retrieved.
    assert expected.matches(SourceRef(filename="note.txt", section="  VITAL   SIGNS "))
    assert not expected.matches(SourceRef(filename="note.txt", section="ALLERGIES"))
    assert not expected.matches(SourceRef(filename="other.txt", section="Vital Signs"))


# --- precision ------------------------------------------------------------


def test_precision_divides_by_k_not_by_what_was_returned() -> None:
    """Returning one good result and stopping is not perfect precision."""
    assert precision_at_k([True], k=5) == pytest.approx(0.2)
    assert precision_at_k([True, False, True, False, False], k=5) == pytest.approx(0.4)


def test_precision_ignores_everything_past_k() -> None:
    assert precision_at_k([True, True, False, False, False, True], k=3) == pytest.approx(2 / 3)


# --- recall ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("covered", "expected", "value"),
    [(0, 3, 0.0), (1, 3, 1 / 3), (3, 3, 1.0), (0, 0, 0.0)],
)
def test_recall_is_over_expected_sources(covered: int, expected: int, value: float) -> None:
    assert recall_at_k(covered, expected) == pytest.approx(value)


def test_recall_cannot_exceed_one_when_a_document_is_retrieved_twice() -> None:
    """The bug this guards: counting chunks instead of documents.

    Three chunks of the one expected document would otherwise give 3/1.
    """
    retrieved = refs("a.txt", "a.txt", "a.txt")
    assert covered_sources(retrieved, refs("a.txt")) == 1
    assert recall_at_k(covered_sources(retrieved, refs("a.txt")), 1) == 1.0


# --- MRR ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relevance", "value"),
    [
        ([True, False, False], 1.0),
        ([False, True, False], 0.5),
        ([False, False, True], 1 / 3),
        ([False, False, False], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank_is_one_over_the_first_hit(relevance: list[bool], value: float) -> None:
    assert reciprocal_rank(relevance) == pytest.approx(value)


# --- NDCG -----------------------------------------------------------------


def test_ndcg_is_one_when_the_ranking_is_ideal() -> None:
    assert ndcg_at_k([True, True, False, False, False], k=5, expected=2) == pytest.approx(1.0)


def test_ndcg_falls_when_a_relevant_result_is_ranked_lower() -> None:
    """Worked by hand: one relevant document at rank 3, one expected.

    DCG  = 1/log2(4) = 0.5
    IDCG = 1/log2(2) = 1.0
    """
    assert ndcg_at_k([False, False, True], k=3, expected=1) == pytest.approx(0.5)


def test_ndcg_does_not_reward_finding_one_of_several_expected_sources() -> None:
    """The ideal is every expected source, not the ones that turned up.

    Denominator built from the retrieved hits instead would score this 1.0 -
    a system that found one document in three looking perfect.
    """
    assert ndcg_at_k([True, False, False], k=3, expected=3) == pytest.approx(
        1.0 / (1 + 1 / math.log2(3) + 0.5)
    )


def test_ndcg_never_exceeds_one_for_a_document_collapsed_ranking() -> None:
    """The bug the first full run exposed: NDCG came back at 1.0073.

    Three chunks of one expected document contributed three gains against an
    ideal built from one expected source. Collapsing to distinct documents
    first is what keeps the two sides of the ratio in the same units.
    """
    retrieved = refs("a.txt", "a.txt", "a.txt", "b.txt", "c.txt")
    expected = refs("a.txt")

    # The uncollapsed judgement is what used to be passed in.
    assert sum(judge(retrieved, expected)) == 3
    assert collapse_to_documents(retrieved, expected) == [True, False, False]
    assert score_ranking(retrieved, expected, k=5).ndcg_at_k == pytest.approx(1.0)


# --- percentiles ----------------------------------------------------------


def test_percentile_interpolates_between_neighbours() -> None:
    # p50 of [1,2,3,4] sits halfway between 2 and 3.
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == 1.0
    assert percentile([1, 2, 3, 4], 100) == 4.0


def test_percentile_handles_degenerate_input() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([7], 95) == 7.0


def test_latency_summary_reports_the_tail_not_only_the_middle() -> None:
    summary = latency_summary([100, 100, 100, 100, 5000])
    assert summary["p50_ms"] == 100.0
    assert summary["p95_ms"] > summary["p50_ms"]
    assert summary["max_ms"] == 5000.0


# --- the whole scoring pass -----------------------------------------------


def test_score_ranking_on_a_realistic_ranking() -> None:
    """Two expected documents; one found at rank 1, the other at rank 4."""
    retrieved = refs("wanted-a.txt", "noise.txt", "noise2.txt", "wanted-b.txt", "noise3.txt")
    metrics = score_ranking(retrieved, refs("wanted-a.txt", "wanted-b.txt"), k=5)

    assert metrics.relevant_retrieved == 2
    assert metrics.precision_at_k == pytest.approx(0.4)  # 2 of 5
    assert metrics.recall_at_k == pytest.approx(1.0)  # both documents covered
    assert metrics.reciprocal_rank == pytest.approx(1.0)  # first result was relevant
    assert metrics.first_relevant_rank == 1
    assert 0.0 < metrics.ndcg_at_k < 1.0  # ideal would have both at ranks 1-2


def test_score_ranking_truncates_to_k_before_judging() -> None:
    """A document found at rank 6 has not been found at all, at k=5."""
    retrieved = refs("a", "b", "c", "d", "e", "wanted.txt")
    metrics = score_ranking(retrieved, refs("wanted.txt"), k=5)

    assert metrics.retrieved == 5
    assert metrics.relevant_retrieved == 0
    assert metrics.recall_at_k == 0.0
    assert metrics.first_relevant_rank is None


def test_aggregate_excludes_cases_that_expect_nothing() -> None:
    """Refusal cases have nothing to recall, and must not dilute the average.

    Without the exclusion, adding refusal cases to the dataset would lower
    recall - making the retrieval metrics a function of the dataset's shape
    rather than of the system.
    """
    found = score_ranking(refs("a.txt"), refs("a.txt"), k=5)
    refusal = score_ranking(refs("a.txt"), [], k=5)

    assert aggregate_ranking([found, refusal]) == aggregate_ranking([found])
    assert aggregate_ranking([found])["recall_at_k"] == 1.0
    assert aggregate_ranking([found])["cases"] == 1


def test_aggregate_of_nothing_is_empty_not_zero() -> None:
    """An empty result set and a result set of zeros are different facts."""
    assert aggregate_ranking([]) == {}
