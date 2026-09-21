"""Chunking, section detection and normalisation.

These are pure functions, so the tests state the rules from
`app/documents/chunking.py` directly rather than going through the API.
"""

from __future__ import annotations

import pytest

from app.documents.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_blocks,
    estimate_tokens,
    split_sentences,
)
from app.documents.normalization import find_boilerplate, normalise_blocks, normalise_text
from app.documents.sections import heading_text
from app.providers.extraction import Block


def body(text: str, page: int = 1) -> Block:
    return Block(text=text, page_number=page, is_heading=False)


def sentences(count: int, word: str = "Observation") -> str:
    return " ".join(f"{word} number {index} was recorded today." for index in range(count))


# --- heading detection ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "CHIEF COMPLAINT",
        "ASSESSMENT AND PLAN",
        "Discharge Instructions:",
        "Laboratory Results",
        "# Findings",
        "Medications",
    ],
)
def test_headings_are_detected(text: str) -> None:
    assert heading_text(Block(text=text, page_number=1)) is not None


@pytest.mark.parametrize(
    "text",
    [
        "The patient returns for scheduled review and reports no new symptoms.",
        "Blood pressure 148/92 mmHg. Heart rate 76 bpm.",
        "MRI",  # a lone acronym is not a section
        "HBA1C 7.2",  # a value, despite being upper case
    ],
)
def test_body_text_is_not_a_heading(text: str) -> None:
    assert heading_text(Block(text=text, page_number=1)) is None


def test_extractor_structure_outranks_the_heuristics() -> None:
    """DOCX tells us the style; a guess must never override that."""
    known_heading = Block(text="this is not shouty", page_number=1, is_heading=True)
    known_body = Block(text="ASSESSMENT AND PLAN", page_number=1, is_heading=False)

    assert heading_text(known_heading) == "this is not shouty"
    assert heading_text(known_body) is None


# --- normalisation ----------------------------------------------------------


def test_ligatures_and_smart_punctuation_are_folded() -> None:
    assert normalise_text("classiﬁcation") == "classification"
    assert normalise_text("the patient’s note") == "the patient's note"
    assert normalise_text("range 4–5") == "range 4-5"


def test_hyphenation_across_a_line_break_is_rejoined() -> None:
    assert normalise_text("hyperten-\nsion") == "hypertension"


def test_a_real_hyphen_survives() -> None:
    """COVID-\\n19 and Follow-\\nUp are hyphens, not line-break artefacts."""
    assert "COVID-19" in normalise_text("COVID-\n19 exposure")


def test_soft_wrapped_lines_are_rejoined_but_sentences_are_not() -> None:
    assert normalise_text("the patient reports\nno new symptoms") == (
        "the patient reports no new symptoms"
    )
    # A sentence boundary is a real break and must survive.
    assert "\n" in normalise_text("First sentence.\nSecond sentence.")


def test_repeated_page_furniture_is_dropped() -> None:
    header = "SYNTHETIC HOSPITAL - CONFIDENTIAL"
    blocks = [
        Block(text=f"{header}\nContent for page {page}.", page_number=page) for page in range(1, 6)
    ]

    assert header in find_boilerplate(blocks)
    assert all(header not in block.text for block in normalise_blocks(blocks))


def test_furniture_detection_needs_enough_pages() -> None:
    """On a two-page document a repeated line is more likely to be content."""
    blocks = [Block(text="Repeated line", page_number=page) for page in (1, 2)]

    assert find_boilerplate(blocks) == set()


def test_page_numbers_are_dropped() -> None:
    blocks = [
        Block(text="Page 3 of 40", page_number=3),
        Block(text="Real clinical content.", page_number=3),
    ]

    kept = [block.text for block in normalise_blocks(blocks)]

    assert kept == ["Real clinical content."]


# --- chunking ---------------------------------------------------------------


def test_sentences_split_without_breaking_abbreviations() -> None:
    parts = split_sentences("Seen by Dr. Lee today. Plan unchanged.")

    assert parts == ["Seen by Dr. Lee today.", "Plan unchanged."]


def test_a_chunk_never_spans_two_sections() -> None:
    """Rule 1: the most important property of the whole chunker."""
    blocks = [
        Block(text="MEDICATIONS", page_number=1),
        body("Metformin 1000 mg twice daily."),
        Block(text="ALLERGIES", page_number=1),
        body("No known drug allergies."),
    ]

    chunks = chunk_blocks(blocks)

    assert [chunk.section for chunk in chunks] == ["MEDICATIONS", "ALLERGIES"]
    assert "Metformin" not in _text_of(chunks, "ALLERGIES")
    assert "allergies" not in _text_of(chunks, "MEDICATIONS").lower()


def test_small_blocks_in_one_section_are_packed_together() -> None:
    """Rule 2: one chunk per paragraph would waste the context window."""
    blocks = [
        Block(text="HISTORY", page_number=1),
        body("First paragraph about the patient."),
        body("Second paragraph about the patient."),
        body("Third paragraph about the patient."),
    ]

    chunks = chunk_blocks(blocks)

    assert len(chunks) == 1
    assert "First" in chunks[0].text and "Third" in chunks[0].text


def test_oversized_sections_are_split_and_overlap_whole_sentences() -> None:
    """Rule 3: overlap exists so a boundary fact appears whole somewhere."""
    config = ChunkingConfig(target_tokens=60, overlap_tokens=20, min_tokens=5)
    blocks = [Block(text="FINDINGS", page_number=1), body(sentences(30))]

    chunks = chunk_blocks(blocks, config)

    assert len(chunks) > 1
    for chunk in chunks:
        # Overlap is taken in whole sentences, so no chunk starts mid-sentence.
        assert chunk.text[0].isupper() or chunk.text[0].isdigit()
    # Consecutive chunks share text - that is the overlap doing its job.
    assert any(
        set(split_sentences(first.text)) & set(split_sentences(second.text))
        for first, second in zip(chunks, chunks[1:], strict=False)
    )


def test_an_unpunctuated_run_is_hard_split_as_a_last_resort() -> None:
    """Rule 4: a table row has no sentences to split on."""
    config = ChunkingConfig(target_tokens=30, overlap_tokens=5, min_tokens=1)
    row = " ".join(f"Cell{index} | {index * 3}" for index in range(60))

    chunks = chunk_blocks([body(row)], config)

    assert len(chunks) > 1
    assert all(chunk.token_count <= config.target_tokens * 2 for chunk in chunks)


def test_tiny_trailing_chunks_are_merged_back() -> None:
    config = ChunkingConfig(target_tokens=200, overlap_tokens=20, min_tokens=30)
    blocks = [
        Block(text="SUMMARY", page_number=1),
        body(sentences(8)),
        body("Signed."),
    ]

    chunks = chunk_blocks(blocks, config)

    assert all(chunk.token_count >= config.min_tokens for chunk in chunks)
    assert "Signed." in chunks[-1].text


def test_chunk_indexes_are_contiguous_after_merging() -> None:
    """(document_id, chunk_index) is a unique key and a citation target."""
    config = ChunkingConfig(target_tokens=50, overlap_tokens=10, min_tokens=20)
    blocks = [
        Block(text="NOTES", page_number=1),
        body(sentences(20)),
        body("Short."),
    ]

    chunks = chunk_blocks(blocks, config)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_page_numbers_are_carried_onto_chunks() -> None:
    """Citations need the page, so provenance has to survive chunking."""
    blocks = [
        Block(text="FINDINGS", page_number=4),
        body("Content from page four.", page=4),
    ]

    chunks = chunk_blocks(blocks)

    assert chunks[0].page_number == 4


def test_a_heading_with_no_body_produces_no_chunk() -> None:
    chunks = chunk_blocks([Block(text="EMPTY SECTION", page_number=1)])

    assert chunks == []


def test_overlap_must_be_smaller_than_the_target() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(target_tokens=100, overlap_tokens=100)


def test_token_estimate_is_monotonic_and_non_zero() -> None:
    """The estimate need not be exact, but it must order text consistently."""
    assert estimate_tokens("") >= 1
    assert estimate_tokens("short text") < estimate_tokens(sentences(10))


def test_chunking_is_deterministic() -> None:
    """Experiments compare configurations; the chunker must not add variance."""
    blocks = [Block(text="HISTORY", page_number=1), body(sentences(40))]
    config = ChunkingConfig(target_tokens=80, overlap_tokens=16)

    first = [chunk.text for chunk in chunk_blocks(blocks, config)]
    second = [chunk.text for chunk in chunk_blocks(blocks, config)]

    assert first == second


def _text_of(chunks: list[Chunk], section: str) -> str:
    return " ".join(chunk.text for chunk in chunks if chunk.section == section)
