"""Grouped citation markers: `[2,3]` as well as `[2]`.

Found by rehearsing the demo against phi3, which wrote "Cholecalciferol 1000 IU
once daily [2,3]". The single-number pattern then did three different wrong
things in three places, each tested here. The stub model only ever writes
`[n]`, which is why nothing caught it before.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from app.agents.synthesis import parse_citations as agent_citations
from app.evaluation.grading import hallucinated_citations, unsupported_numbers
from app.rag.citations import citation_numbers, strip_citations
from app.rag.prompts import parse_citations as chat_citations

FIVE: list[Any] = [object()] * 5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dose [2]", [2]),
        ("dose [2,3]", [2, 3]),
        ("dose [2, 3]", [2, 3]),
        ("dose [1,2,5] and [4]", [1, 2, 5, 4]),
        ("dose [2][3]", [2, 3]),
        # Not citations: a range, a word, an empty pair, and prose brackets.
        ("dose [2-3]", []),
        ("see [above]", []),
        ("[]", []),
        ("HbA1c 7.4% [reference 4.0-5.6]", []),
    ],
)
def test_every_cited_number_is_read(text: str, expected: list[int]) -> None:
    assert citation_numbers(text) == expected


def test_chat_keeps_every_source_of_a_grouped_citation() -> None:
    """Before: [3] vanished from the citation list the user sees."""
    answer = "Atorvastatin 20 mg [2], and cholecalciferol 1000 IU daily [2,3]."
    assert chat_citations(answer, cast(Any, FIVE)) == [2, 3]


def test_the_agent_keeps_every_source_of_a_grouped_citation() -> None:
    answer = "HbA1c fell from 8.1% to 7.4% [1, 4]."
    assert agent_citations(answer, cast(Any, FIVE)) == [1, 4]


def test_an_out_of_range_number_inside_a_group_is_still_dropped() -> None:
    """Grouping must not smuggle a hallucinated citation past the range check."""
    assert chat_citations("dose [2,9]", cast(Any, FIVE)) == [2]


def test_a_grouped_citation_is_not_read_as_an_invented_number() -> None:
    """Before: "[2,3]" was left in the prose and graded as the number 2.3.

    That made the grader report a hallucinated number the model never wrote,
    inflating `unsupported_number_rate`, which is the headline answer-quality
    figure for a real model.
    """
    context = "CURRENT MEDICATIONS\n- Cholecalciferol 1000 IU once daily"
    answer = "Cholecalciferol 1000 IU once daily [2,3]."
    assert unsupported_numbers(answer, context) == []


def test_a_real_invented_number_is_still_caught() -> None:
    context = "Haemoglobin A1c 7.4%"
    assert unsupported_numbers("HbA1c was 8.2% [1,2].", context) == ["8.2"]


def test_hallucinated_citations_are_counted_per_number_not_per_bracket() -> None:
    assert hallucinated_citations("claim [1,7] and [9]", supplied_blocks=5) == 2


def test_stripping_leaves_the_prose_alone() -> None:
    assert " ".join(strip_citations("dose [2, 3] daily").split()) == "dose daily"
