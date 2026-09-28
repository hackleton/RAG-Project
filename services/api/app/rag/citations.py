"""Reading citation markers out of generated text.

One definition, used by chat, by the agent's synthesis step and by the
evaluation grader. There used to be three copies of the same pattern, which
matched `[2]` and nothing else. A real model writes `[2,3]` and `[2, 3]` as
readily as `[2][3]`, and the single-number pattern cost three different things
at once:

* **chat and the agent** kept the first source of a grouped citation and
  silently dropped the rest, so a claim's supporting chunk vanished from the
  citation list the user sees;
* **the grader** failed to strip the marker before looking for numbers, so
  `[2,3]` was read as the number 2.3 and counted as a number the model had
  invented.

The deterministic stub only ever writes `[n]`, so no test could see any of it.
It was found by rehearsing the demo against phi3 (docs/guide/3-when-it-breaks.md).

Kept free of imports beyond `re`, so anything can depend on it without an
import cycle.
"""

from __future__ import annotations

import re

# "[3]", "[2,3]", "[2, 3]", "[1,2,5]". Two digits at most, which is more than
# the number of blocks any prompt supplies. Ranges like "[2-3]" are not
# accepted: no model has written one here, and guessing what a range means
# would put words in the model's mouth.
CITATION_MARKER = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")


def citation_numbers(text: str) -> list[int]:
    """Every number cited in `text`, in order of appearance, repeats included.

    Range checking is the caller's job. Only the caller knows how many blocks
    were supplied, and the caller decides what an out-of-range number means:
    dropped from the answer, or counted as a hallucinated citation.
    """
    return [
        int(part) for match in CITATION_MARKER.finditer(text) for part in match.group(1).split(",")
    ]


def strip_citations(text: str) -> str:
    """Remove every citation marker, so its digits cannot be read as data."""
    return CITATION_MARKER.sub(" ", text)
