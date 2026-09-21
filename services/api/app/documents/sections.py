"""Heading detection.

Clinical documents are unusually regular: notes and reports are organised under
a small, repeated vocabulary of section headings - CHIEF COMPLAINT, ASSESSMENT
AND PLAN, MEDICATIONS, RESULTS. That regularity is worth exploiting, because
knowing the section a chunk came from does two things:

1. It lets the chunker split on semantic boundaries rather than at a fixed
   character offset, so a medication list is not cut in half.
2. It gives every chunk a `section` value, which is both a citation field and a
   retrieval filter ("what did the assessment say?").

Detection is heuristic and ordered from most to least reliable. `is_heading`
already set by the extractor (DOCX heading styles) always wins - that is
structural fact, not inference.
"""

from __future__ import annotations

import re

from app.providers.extraction import Block

# A heading is short. Anything longer is a sentence that happens to look shouty.
MAX_HEADING_WORDS = 12
MAX_HEADING_CHARS = 90

# "CHIEF COMPLAINT:", "Assessment and Plan:", "MEDICATIONS"
_TRAILING_COLON = re.compile(r"^[A-Z][A-Za-z0-9 /&'()-]{2,}:$")
# "1. History", "2.1 Findings", "IV. Discussion"
_NUMBERED = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S.*$")
# "# Heading" in Markdown.
_MARKDOWN = re.compile(r"^#{1,6}\s+\S")
# A bare number, decimal or percentage: a measured value, not a section name.
_VALUE_TOKEN = re.compile(r"^[<>]?\d+(?:[.,]\d+)?%?$")

# Headings seen in the synthetic corpus and in real clinical note templates.
# Matching one is strong evidence even when the casing is inconsistent.
KNOWN_SECTIONS = frozenset(
    {
        "chief complaint",
        "history of present illness",
        "past medical history",
        "family history",
        "social history",
        "review of systems",
        "physical examination",
        "vital signs",
        "medications",
        "current medications",
        "allergies",
        "laboratory results",
        "lab results",
        "results",
        "imaging",
        "assessment",
        "plan",
        "assessment and plan",
        "impression",
        "diagnosis",
        "diagnoses",
        "procedures",
        "follow-up",
        "discharge summary",
        "discharge instructions",
        "recommendations",
        "summary",
        "indication",
        "technique",
        "findings",
        "comparison",
    }
)

DEFAULT_SECTION = "Document"


def heading_text(block: Block) -> str | None:
    """Return the block's heading title, or None if it is body text."""
    # Structural truth from the extractor outranks every heuristic below.
    if block.is_heading is True:
        return _clean(block.text)
    if block.is_heading is False:
        return None

    text = block.text.strip()
    if not text or "\n" in text:
        return None
    if len(text) > MAX_HEADING_CHARS or len(text.split()) > MAX_HEADING_WORDS:
        return None

    stripped = _clean(text)
    if not stripped:
        return None

    if _MARKDOWN.match(text):
        return stripped
    if stripped.lower() in KNOWN_SECTIONS:
        return stripped
    if _TRAILING_COLON.match(text):
        return stripped
    # ALL CAPS, but not an acronym on its own ("MRI") and not a measurement
    # ("HBA1C 7.2") - a section name never carries a value.
    words = text.split()
    letters = [character for character in text if character.isalpha()]
    if (
        letters
        and all(character.isupper() for character in letters)
        and len(words) >= 2
        and not any(_VALUE_TOKEN.match(word) for word in words)
    ):
        return stripped
    if _NUMBERED.match(text) and text.endswith((":", "")) and len(text.split()) <= 8:
        return stripped
    return None


def _clean(text: str) -> str:
    return text.lstrip("#").strip().rstrip(":").strip()
