"""Section-aware chunking.

The rules, in priority order:

1. **Never merge across a section boundary.** A chunk that straddles
   "MEDICATIONS" and "ALLERGIES" answers neither question well and cites a
   section it is only half from.
2. **Fill to the target size within a section.** One-block chunks would waste
   the context window and strip the surrounding sentences a reader needs.
3. **Overlap by whole sentences, never mid-sentence.** Overlap exists so a fact
   sitting on a boundary appears whole in at least one chunk; splitting a
   sentence to hit an exact token count defeats that.
4. **Split an oversized single block by sentence**, and only fall back to a hard
   character split if one "sentence" is itself larger than the target - which in
   practice means a table row or an unpunctuated list.

Chunk size, overlap and the token counter are all injected rather than fixed:
project plan section 14 treats them as experiment variables, and the real
tokenizer only arrives with the embedding model in Sprint 3.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from app.documents.sections import DEFAULT_SECTION, heading_text
from app.providers.extraction import Block

# Split after ., ! or ? followed by whitespace and a capital/digit. The negative
# lookbehinds keep common clinical abbreviations intact. They are evaluated at
# the position *after* the period, so each one has to include the period itself.
_ABBREVIATIONS = (
    r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)"
    r"(?<!\bvs\.)(?<!\bNo\.)(?<!\bapprox\.)(?<!\bal\.)"
)
_SENTENCE_END = re.compile(rf"{_ABBREVIATIONS}(?<=[.!?])\s+(?=[A-Z0-9])")

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough token count, deliberately model-independent.

    Sprint 3 replaces this with the embedding model's own tokenizer - the
    chunker takes the counter as a parameter precisely so that swap is a
    one-line change. Until then, chunk boundaries only need to be *consistent*,
    not exact: what matters is that the same document always chunks the same
    way so an experiment measures the chunk size and not the counter.

    The 0.75 words-per-token ratio is the usual English approximation, with
    punctuation and digits contributing separately because clinical text is
    dense with both ("HbA1c 7.2%", "2026-09-22").
    """
    words = len(text.split())
    symbols = sum(1 for character in text if not character.isalnum() and not character.isspace())
    return max(1, int(words / 0.75) + symbols // 3)


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]


@dataclass(slots=True)
class Chunk:
    text: str
    index: int
    page_number: int | None
    section: str | None
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkingConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    # Below this a chunk is not worth indexing on its own; it is merged back
    # into the previous chunk of the same section instead.
    min_tokens: int = 40
    count_tokens: TokenCounter = estimate_tokens

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")


@dataclass(slots=True)
class _Pending:
    """A chunk under construction within one section."""

    section: str
    page_number: int
    pieces: list[str] = field(default_factory=list)
    tokens: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.pieces).strip()


def chunk_blocks(blocks: Iterable[Block], config: ChunkingConfig | None = None) -> list[Chunk]:
    """Turn normalised blocks into retrievable chunks."""
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    section = DEFAULT_SECTION
    pending: _Pending | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        text = pending.text
        if text:
            _append(chunks, text, pending.page_number, pending.section, config)
        pending = None

    for block in blocks:
        if (heading := heading_text(block)) is not None:
            # Rule 1: a heading closes the previous section's chunk.
            flush()
            section = heading
            continue

        for piece in _fit_pieces(block.text, config):
            piece_tokens = config.count_tokens(piece)

            if pending is not None and pending.section != section:
                flush()

            if pending is None:
                pending = _Pending(section=section, page_number=block.page_number)

            if pending.tokens and pending.tokens + piece_tokens > config.target_tokens:
                carry = _overlap_pieces(pending.pieces, config)
                flush()
                pending = _Pending(
                    section=section,
                    page_number=block.page_number,
                    pieces=list(carry),
                    tokens=sum(config.count_tokens(item) for item in carry),
                )

            pending.pieces.append(piece)
            pending.tokens += piece_tokens

    flush()
    return _merge_runts(chunks, config)


def _fit_pieces(text: str, config: ChunkingConfig) -> list[str]:
    """Break one block into units no larger than the target size.

    A block that already fits is a single unit - splitting it into sentences
    unconditionally would lose paragraph cohesion for no benefit.
    """
    if config.count_tokens(text) <= config.target_tokens:
        return [text]

    pieces: list[str] = []
    for sentence in split_sentences(text):
        if config.count_tokens(sentence) <= config.target_tokens:
            pieces.append(sentence)
        else:
            # Rule 4: an unpunctuated run (a table row, a long list). Hard split
            # is the only option left, but it stays a last resort.
            pieces.extend(_hard_split(sentence, config))
    return pieces or [text]


def _hard_split(text: str, config: ChunkingConfig) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if config.count_tokens(" ".join(current)) >= config.target_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_pieces(pieces: list[str], config: ChunkingConfig) -> list[str]:
    """Take whole trailing pieces up to the overlap budget (rule 3)."""
    if not config.overlap_tokens:
        return []
    carried: list[str] = []
    budget = config.overlap_tokens
    for piece in reversed(pieces):
        cost = config.count_tokens(piece)
        if cost > budget:
            break
        carried.insert(0, piece)
        budget -= cost
    return carried


def _append(
    chunks: list[Chunk],
    text: str,
    page_number: int,
    section: str,
    config: ChunkingConfig,
) -> None:
    chunks.append(
        Chunk(
            text=text,
            index=len(chunks),
            page_number=page_number,
            section=section,
            token_count=config.count_tokens(text),
        )
    )


def _merge_runts(chunks: list[Chunk], config: ChunkingConfig) -> list[Chunk]:
    """Fold tiny trailing chunks back into the previous chunk of the same section.

    A three-word chunk ("Signed: Dr Lee") is noise in a retrieval index: it
    matches nothing useful and dilutes the scores of chunks that do. It is only
    merged within its own section, so rule 1 still holds.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        too_small = chunk.token_count < config.min_tokens
        same_section = previous is not None and previous.section == chunk.section
        fits = (
            previous is not None
            and previous.token_count + chunk.token_count <= config.target_tokens
        )
        if too_small and same_section and fits and previous is not None:
            previous.text = f"{previous.text}\n{chunk.text}"
            previous.token_count = config.count_tokens(previous.text)
            continue
        merged.append(chunk)

    # Indexes must be contiguous: they are half of the (document_id, chunk_index)
    # unique key and the stable identifier a citation points at.
    for position, chunk in enumerate(merged):
        chunk.index = position
    return merged
