"""The evaluation dataset: its schema, its file format, and its database sync.

The dataset is a versioned JSONL file under `evaluation/datasets/questions/`,
one case per line, committed to the repository. Three consequences follow from
that choice and each is deliberate.

**Versioned by filename.** `v1.jsonl` is dataset version `v1`. Every run records
the version it was scored against, because a metric compared across two
different datasets is meaningless and the only defence is making the version
impossible to omit. A new question is a new version, not an edit to an old one -
`v1` never changes once a run has been recorded against it.

**JSONL rather than JSON.** A hundred cases in one JSON array produce a diff
that reindents on every insertion; one case per line means a changed question is
a one-line diff and a reviewer can see what moved.

**Loaded, validated, then synced.** The file is the source of truth and the
`evaluation_cases` table is a materialisation of it, keyed on a stable
`case_key` so per-case results stay comparable across runs. The sync is
idempotent and refuses to silently rewrite a case that a run has already been
scored against.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.evaluation.metrics import SourceRef
from app.models.evaluation import EvaluationCase

logger = get_logger(__name__)

DEFAULT_DATASET_DIR = Path("evaluation/datasets/questions")

Difficulty = Literal["easy", "medium", "hard"]
Persona = Literal["admin", "clinician", "analyst"]


class CaseCategory(StrEnum):
    """What a case is testing.

    Categories are not decoration: the harness reports metrics per category,
    because a corpus average hides the thing worth knowing. A system that
    answers every lab question and no guideline question has the same mean as
    one that is mediocre at both.
    """

    LAB_VALUE = "lab_value"
    MEDICATION = "medication"
    VITALS = "vitals"
    DIAGNOSIS = "diagnosis"
    DISCHARGE = "discharge"
    GUIDELINE = "guideline"
    DRUG_REFERENCE = "drug_reference"
    TEMPORAL = "temporal"
    # The context genuinely does not contain the answer. Scored on refusal, not
    # on retrieval - a dataset with no unanswerable cases cannot tell a
    # grounded system from a confident one.
    UNANSWERABLE = "unanswerable"
    # Project plan section 12, measured rather than asserted.
    INJECTION = "injection"
    # The caller asks about material outside their scope.
    ACCESS_CONTROL = "access_control"


class ExpectedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    # Optional: narrows the expectation to one section of the document.
    section: str | None = None

    def as_ref(self) -> SourceRef:
        return SourceRef(filename=self.filename, section=self.section)


class CaseSpec(BaseModel):
    """One evaluation case as it appears on disk."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1, max_length=2000)
    category: CaseCategory
    difficulty: Difficulty
    # Who asks. Access-control cases are only meaningful with a persona whose
    # scope excludes the material, so this is part of the case, not the runner.
    persona: Persona = "clinician"
    # Which patient the question is about, if any. Narrows retrieval exactly as
    # a real caller would; `null` means the whole of the caller's scope.
    patient_external_id: str | None = Field(default=None, max_length=64)
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)

    @field_validator("expected_facts")
    @classmethod
    def _facts_are_substantive(cls, value: list[str]) -> list[str]:
        # A one-character fact matches almost any answer and would quietly
        # inflate correctness.
        if any(len(fact.strip()) < 2 for fact in value):
            raise ValueError("expected_facts entries must be at least two characters")
        return value

    @property
    def source_refs(self) -> list[SourceRef]:
        return [source.as_ref() for source in self.expected_sources]

    @property
    def expects_refusal(self) -> bool:
        """Cases where the right answer is to decline."""
        return self.category in (
            CaseCategory.UNANSWERABLE,
            CaseCategory.ACCESS_CONTROL,
        )


@dataclass(frozen=True, slots=True)
class Dataset:
    version: str
    cases: tuple[CaseSpec, ...]
    path: Path

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[CaseSpec]:
        return iter(self.cases)

    @property
    def counts_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category.value] = counts.get(case.category.value, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def counts_by_difficulty(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.difficulty] = counts.get(case.difficulty, 0) + 1
        return counts


class DatasetError(ValueError):
    """The dataset file is malformed. Always fatal - a run on a broken dataset
    would produce numbers that look fine."""


def load_dataset(path: Path) -> Dataset:
    """Read and validate a dataset file. The version is the filename stem."""
    if not path.is_file():
        raise DatasetError(f"No dataset at {path}. Run scripts/generate_evaluation_dataset.py.")

    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        try:
            case = CaseSpec.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetError(f"{path.name} line {number}: {exc}") from exc
        if case.key in seen:
            # Keys correlate results across runs; a duplicate would silently
            # overwrite one case's history with another's.
            raise DatasetError(f"{path.name} line {number}: duplicate case key '{case.key}'")
        seen.add(case.key)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{path.name} contains no cases.")
    return Dataset(version=path.stem, cases=tuple(cases), path=path)


def write_dataset(path: Path, cases: list[CaseSpec]) -> int:
    """Write cases as JSONL, one per line, with stable key ordering.

    Sorted keys and an explicit newline so regenerating the dataset on another
    machine produces the same bytes - the same reason the corpus generator does
    it. CI asserts on that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(case.model_dump(mode="json", exclude_none=False), sort_keys=True)
        for case in cases
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


async def sync_cases(session: AsyncSession, dataset: Dataset) -> dict[str, int]:
    """Materialise the dataset into `evaluation_cases`.

    Idempotent: a case whose content is unchanged is left alone, so its row id -
    and therefore every historical result that points at it - survives. A
    changed case is updated in place, which is correct precisely because the
    dataset version is immutable by convention: if the question changed, the
    version should have.
    """
    existing = {
        case.case_key: case
        for case in await session.scalars(
            select(EvaluationCase).where(EvaluationCase.dataset_version == dataset.version)
        )
    }

    added = updated = unchanged = 0
    for spec in dataset:
        payload = _row_payload(spec)
        row = existing.pop(spec.key, None)
        if row is None:
            session.add(EvaluationCase(dataset_version=dataset.version, **payload))
            added += 1
            continue
        if all(getattr(row, field) == value for field, value in payload.items()):
            unchanged += 1
            continue
        for field, value in payload.items():
            setattr(row, field, value)
        updated += 1

    # Cases removed from the file are left in place. Deleting them would cascade
    # to every result ever recorded against them, and a dataset's history is the
    # point of keeping it in a database at all.
    orphaned = len(existing)
    await session.flush()

    summary = {
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "orphaned": orphaned,
    }
    logger.info("evaluation_dataset_synced", version=dataset.version, **summary)
    return summary


def _row_payload(spec: CaseSpec) -> dict[str, object]:
    return {
        "case_key": spec.key,
        "question": spec.question,
        "expected_answer": spec.expected_answer,
        "expected_facts": list(spec.expected_facts),
        "expected_sources": [source.model_dump(mode="json") for source in spec.expected_sources],
        "category": spec.category.value,
        "difficulty": spec.difficulty,
        "persona": spec.persona,
        "patient_external_id": spec.patient_external_id,
    }
