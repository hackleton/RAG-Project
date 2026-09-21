"""Text normalisation, applied between extraction and chunking.

Every transformation here exists because of a specific way PDF text extraction
degrades retrieval quality:

* **Ligatures and smart punctuation.** A PDF renders "fi" as U+FB01. A query for
  "classification" then fails to match "classi<ligature>cation" under BM25.
* **Hyphenation across line breaks.** "hyperten-\\ntension" is two tokens to a
  keyword index and one concept to a human.
* **Soft line wrapping.** PDF text carries the *visual* line breaks. Left alone,
  a sentence spanning three rendered lines becomes three fragments.
* **Repeated headers and footers.** A page header repeated on all forty pages is
  forty near-identical chunks competing with the real content.

Normalisation is deliberately conservative: it never drops content it is not
confident about, because an over-eager cleaner silently deletes the lab value
someone is asking about.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from app.providers.extraction import Block

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

PUNCTUATION = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "…": "...",
    " ": " ",  # non-breaking space
    "​": "",  # zero-width space
}

# "hyperten-\ntension" -> "hypertension": a word split across a line break, so
# the hyphen is an artefact and goes. Only when the next line starts lowercase.
_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")
# "COVID-\n19", "Follow-\nUp": a real hyphen that happens to fall at the wrap.
# Join the lines but keep the hyphen.
_HYPHEN_WRAP = re.compile(r"(\w)-\n([A-Z0-9])")
# A line break that is not a paragraph break: no sentence-ending punctuation
# before it and the next line continues in lower case.
_SOFT_WRAP = re.compile(r"(?<![.!?:;])\n(?=[a-z(])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# A page number alone on a line, or "Page 3 of 40".
_PAGE_MARKER = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", re.IGNORECASE)

# A line has to repeat on at least this share of pages to count as furniture,
# and the document needs at least this many pages for the test to mean anything.
BOILERPLATE_PAGE_SHARE = 0.6
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MAX_LENGTH = 120


def normalise_text(text: str) -> str:
    """Clean one run of text. Safe to apply to any extractor's output."""
    # NFKC folds full-width and compatibility forms; ligatures are handled
    # explicitly because NFKC leaves some of them alone.
    text = unicodedata.normalize("NFKC", text)
    for source, replacement in {**LIGATURES, **PUNCTUATION}.items():
        text = text.replace(source, replacement)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _HYPHEN_WRAP.sub(r"\1-\2", text)
    text = _SOFT_WRAP.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def find_boilerplate(blocks: Iterable[Block]) -> set[str]:
    """Lines that repeat across most pages: running headers and footers.

    Frequency-based rather than position-based, because pypdf gives no
    coordinates. The thresholds are conservative - a line has to be short and
    appear on most pages before it is treated as furniture.
    """
    blocks = list(blocks)
    pages = {block.page_number for block in blocks}
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return set()

    pages_per_line: Counter[str] = Counter()
    for line, _page in {
        (line.strip(), block.page_number)
        for block in blocks
        for line in block.text.split("\n")
        if line.strip()
    }:
        pages_per_line[line] += 1

    threshold = max(2, int(len(pages) * BOILERPLATE_PAGE_SHARE))
    return {
        line
        for line, count in pages_per_line.items()
        if count >= threshold and len(line) <= BOILERPLATE_MAX_LENGTH
    }


def normalise_blocks(blocks: Iterable[Block]) -> list[Block]:
    """Normalise every block and drop page furniture.

    Returns new blocks; the extractor's output is left untouched so a failure
    here can be diagnosed against the raw extraction.
    """
    blocks = list(blocks)
    boilerplate = find_boilerplate(blocks)

    cleaned: list[Block] = []
    for block in blocks:
        kept_lines = [
            line
            for line in block.text.split("\n")
            if line.strip() and line.strip() not in boilerplate and not _PAGE_MARKER.match(line)
        ]
        if not kept_lines:
            continue
        text = normalise_text("\n".join(kept_lines))
        if text:
            cleaned.append(
                Block(text=text, page_number=block.page_number, is_heading=block.is_heading)
            )
    return cleaned
