"""Experiment sweeps and the regression gate.

Two jobs that share the same machinery.

**Sweeps** answer project plan section 14: run the same dataset under several
retrieval configurations and record what changed. Every experiment here is a
`RetrievalConfig` difference, because those are the variables a running system
can sweep without re-ingesting the corpus. The two that cannot - chunk size and
embedding model - are listed in `REQUIRES_REINGEST` with the command that
performs them, rather than quietly omitted, because the omission is a property
of how the index is built and not of what is worth measuring.

**The regression gate** answers the other half of Sprint 5: a metric is only
useful if something fails when it drops. `check_regression` compares a run
against a stored baseline and returns the violations, and the caller turns that
into an exit code. Which metrics are gated is a deliberate, narrow list -
gating everything would make the build fail on noise, and a gate people
routinely override is not a gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.rag.retriever import RetrievalConfig


def environment_payload(settings: Settings) -> dict[str, Any]:
    """What a run has to record to be comparable with another one.

    Retrieval numbers are only meaningful next to the embedding model that
    produced them, and generation numbers only next to the model that wrote
    them. Recording both on the run means two runs can always be told apart
    when they should not be compared.
    """
    return {
        "embedding_provider": settings.embedding_provider.value,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model if settings.enable_reranker else None,
        "llm_provider": settings.llm_provider.value,
        "llm_model": settings.llm_model,
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "cache_backend": settings.cache_backend.value,
        "cache_rankings": settings.cache_rankings,
    }


@dataclass(frozen=True, slots=True)
class Experiment:
    """One configuration under test, with the question it is meant to answer."""

    name: str
    hypothesis: str
    overrides: dict[str, Any]

    def config(self, base: RetrievalConfig) -> RetrievalConfig:
        return replace(base, **self.overrides)


# Project plan section 14, restricted to what a sweep can vary at query time.
EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="baseline",
        hypothesis="The deployed configuration. Every other row is read against this one.",
        overrides={},
    ),
    Experiment(
        name="dense-only",
        hypothesis=(
            "Vector search alone loses the exact-match cases - identifiers, drug "
            "names, ICD-10 codes - that BM25 is good at."
        ),
        overrides={"mode": "dense"},
    ),
    Experiment(
        name="sparse-only",
        hypothesis=(
            "BM25 alone loses the paraphrased cases, where the question shares no "
            "vocabulary with the document."
        ),
        overrides={"mode": "sparse"},
    ),
    Experiment(
        name="hybrid-no-rerank",
        hypothesis=(
            "The cross-encoder is the dominant cost in a request. This measures "
            "what accuracy that cost buys."
        ),
        overrides={"rerank": False},
    ),
    Experiment(
        name="top-k-3",
        hypothesis="A tighter context is cheaper and less distracting, at the cost of recall.",
        overrides={"top_k": 3},
    ),
    Experiment(
        name="top-k-10",
        hypothesis=(
            "More context should lift recall on the multi-document cases; precision "
            "must fall, since the denominator grows."
        ),
        overrides={"top_k": 10},
    ),
    Experiment(
        name="pool-20",
        hypothesis=(
            "A small candidate pool is faster. If the metrics hold, the deployed "
            "pool of 50 is doing nothing."
        ),
        overrides={"candidate_pool": 20},
    ),
    Experiment(
        name="pool-100",
        hypothesis="A larger pool gives fusion and the reranker more to work with.",
        overrides={"candidate_pool": 100},
    ),
)

# Variables that change the index rather than the query, so they cannot be
# swept in one process. Recorded here so the gap is explicit.
REQUIRES_REINGEST: tuple[tuple[str, str, str], ...] = (
    (
        "chunk-size",
        "Smaller chunks raise precision and lose the context a claim needs.",
        "CHUNK_TARGET_TOKENS=256 docker compose run --rm api python scripts/ingest_corpus.py "
        "--reset, then run the evaluation again",
    ),
    (
        "embedding-model",
        "A larger embedding model should help the paraphrased cases most.",
        "EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 EMBEDDING_DIMENSIONS=768 docker compose run "
        "--rm api python scripts/reindex.py --all, then run the evaluation again",
    ),
)


# --- regression gate ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gate:
    """One gated metric: where to find it, and how far it may move."""

    path: tuple[str, ...]
    tolerance: float
    higher_is_better: bool = True

    @property
    def label(self) -> str:
        return ".".join(self.path)


# Deliberately short. These are the metrics that are reproducible with the
# deterministic stack (no model sampling), and each one failing means something
# a user would notice.
GATES: tuple[Gate, ...] = (
    Gate(("retrieval", "recall_at_k"), tolerance=0.05),
    Gate(("retrieval", "mrr"), tolerance=0.05),
    Gate(("retrieval", "ndcg_at_k"), tolerance=0.05),
    Gate(("security", "injection_blocked_rate"), tolerance=0.0),
    Gate(("security", "access_contained_rate"), tolerance=0.0),
    Gate(("system", "error_rate"), tolerance=0.02, higher_is_better=False),
)

# A security metric that must never regress at all, whatever the tolerance says.
ABSOLUTE = frozenset({"security.injection_blocked_rate", "security.access_contained_rate"})


@dataclass(frozen=True, slots=True)
class Violation:
    metric: str
    baseline: float
    current: float
    tolerance: float

    @property
    def delta(self) -> float:
        return round(self.current - self.baseline, 4)

    def __str__(self) -> str:
        return (
            f"{self.metric}: {self.baseline:.4f} -> {self.current:.4f} "
            f"({self.delta:+.4f}, tolerance {self.tolerance})"
        )


def check_regression(
    current: dict[str, Any], baseline: dict[str, Any], gates: tuple[Gate, ...] = GATES
) -> list[Violation]:
    """Which gated metrics moved the wrong way by more than their tolerance.

    A metric missing from either side is skipped rather than treated as zero. A
    baseline recorded before a metric existed is a normal state, and scoring it
    as a catastrophic regression would train people to pass `--force`.
    """
    violations = []
    for gate in gates:
        before, after = _dig(baseline, gate.path), _dig(current, gate.path)
        if before is None or after is None:
            continue
        drop = (before - after) if gate.higher_is_better else (after - before)
        tolerance = 0.0 if gate.label in ABSOLUTE else gate.tolerance
        if drop > tolerance:
            violations.append(
                Violation(metric=gate.label, baseline=before, current=after, tolerance=tolerance)
            )
    return violations


# What has to match before two runs are worth comparing at all. Every one of
# these determines what is *in the index*: a different embedding model or a
# different chunk size produces a different set of vectors, and a baseline
# taken over those describes a different system.
#
# The retrieval configuration - top_k, candidate_pool, mode, rerank - is
# deliberately NOT on this list, even though it is recorded on every baseline.
# Those are the settings a developer actually changes, so a drop caused by
# changing one is precisely the regression the gate exists to catch. Treating a
# retrieval change as "incomparable" would skip the check exactly when it
# matters: dropping `RETRIEVAL_TOP_K` to 3 costs 0.12 recall on this dataset,
# and an earlier version of this list reported that as "cannot compare" rather
# than as the regression it is.
COMPARABLE_KEYS: tuple[str, ...] = (
    "embedding_provider",
    "embedding_model",
    "chunk_target_tokens",
    "chunk_overlap_tokens",
)


def incomparable(
    baseline: dict[str, Any], *, dataset_version: str, config: dict[str, Any]
) -> list[str]:
    """Why this run cannot be judged against this baseline, if it cannot.

    Returned as reasons rather than a boolean so the caller can print them: a
    refusal to compare is only useful if it says what to fix.
    """
    reasons = []
    if baseline.get("dataset_version") != dataset_version:
        reasons.append(
            f"dataset version {baseline.get('dataset_version')!r} != {dataset_version!r}"
        )
    recorded = baseline.get("config", {})
    for key in COMPARABLE_KEYS:
        if recorded.get(key) != config.get(key):
            reasons.append(f"{key} {recorded.get(key)!r} != {config.get(key)!r}")
    return reasons


def retrieval_changes(baseline: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Retrieval settings that differ from the baseline's.

    Not a reason to skip the comparison - a reason to read it. A regression
    alongside "top_k 5 -> 3" is a tuning decision to accept or revert; the same
    regression with no config change is a bug.
    """
    before = baseline.get("config", {}).get("retrieval", {}) or {}
    after = config.get("retrieval", {}) or {}
    return [
        f"{key} {before.get(key)!r} -> {after[key]!r}"
        for key in sorted(after)
        if before.get(key) != after[key]
    ]


# Settings that change how answers are *written* rather than what is retrieved.
# Deliberately not in COMPARABLE_KEYS, for the same reason the retrieval config
# is not: every gated metric is either retrieval (upstream of the model) or
# containment (guaranteed by the access filter whatever writes the answer), so
# swapping models must not make the gate skip itself. It must make the reader
# read the generation numbers differently, which is what this is for.
GENERATION_KEYS: tuple[str, ...] = ("llm_provider", "llm_model")


def generation_changes(baseline: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Model differences from the baseline's - context, not a verdict.

    Without this a run can report "no regression" while every generation metric
    in it was produced by a different model. The gated metrics really are
    unaffected, so the verdict is honest; the numbers underneath it are not
    comparable, and silence implied they were.
    """
    recorded = baseline.get("config", {})
    return [
        f"{key} {recorded.get(key)!r} -> {config[key]!r}"
        for key in GENERATION_KEYS
        if key in config and recorded.get(key) != config[key]
    ]


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def write_baseline(
    path: Path, *, dataset_version: str, config: dict[str, Any], metrics: dict[str, Any]
) -> None:
    """Record the accepted numbers.

    Written explicitly with `--accept`, never automatically at the end of a run.
    A baseline that updates itself can only ever pass, which is the one thing a
    baseline must not do.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_version": dataset_version,
        "config": config,
        **metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- reporting ------------------------------------------------------------

REPORT_METRICS: tuple[tuple[str, ...], ...] = (
    ("retrieval", "recall_at_k"),
    ("retrieval", "precision_at_k"),
    ("retrieval", "mrr"),
    ("retrieval", "ndcg_at_k"),
    ("retrieval", "hit_rate"),
    ("generation", "fact_coverage"),
    ("generation", "citation_precision"),
    ("security", "injection_blocked_rate"),
    ("system", "p50_ms"),
    ("system", "p95_ms"),
)


def sweep_report(
    *,
    dataset_version: str,
    environment: dict[str, Any],
    results: list[tuple[Experiment, dict[str, Any]]],
) -> str:
    """The section 14 experiment record, as Markdown.

    Hypothesis, configuration, dataset version and metrics are filled in
    mechanically. *Interpretation* and *decision* are left blank on purpose:
    they are the judgement the experiment exists to produce, and a generated
    sentence there would be a plausible-looking guess in the one field a reader
    is entitled to trust.
    """
    baseline = next(
        (metrics for experiment, metrics in results if experiment.name == "baseline"), {}
    )
    lines = [
        f"# Retrieval sweep - dataset `{dataset_version}`",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}.",
        "",
        "| setting | value |",
        "| --- | --- |",
        *(f"| {key} | `{value}` |" for key, value in sorted(environment.items())),
        "",
        "## Results",
        "",
        "| experiment | " + " | ".join(".".join(path) for path in REPORT_METRICS) + " |",
        "| --- |" + " --- |" * len(REPORT_METRICS),
    ]

    for experiment, metrics in results:
        cells = []
        for path in REPORT_METRICS:
            value = _dig(metrics, path)
            cells.append("-" if value is None else _cell(value, _dig(baseline, path), path))
        lines.append(f"| {experiment.name} | " + " | ".join(cells) + " |")

    lines += ["", "## Hypotheses", ""]
    for experiment, _ in results:
        lines += [f"**{experiment.name}** - {experiment.hypothesis}", ""]

    lines += [
        "## Not swept here",
        "",
        "These change the index rather than the query, so they need a re-ingest:",
        "",
    ]
    for name, hypothesis, command in REQUIRES_REINGEST:
        lines += [f"**{name}** - {hypothesis}", "", f"    {command}", ""]

    lines += [
        "## Interpretation",
        "",
        "_To be written by whoever ran this._",
        "",
        "## Decision",
        "",
        "_Adopt, reject, or investigate further - and why._",
        "",
    ]
    return "\n".join(lines)


def _cell(value: float, baseline: float | None, path: tuple[str, ...]) -> str:
    text = f"{value:.4f}" if value < 100 else f"{value:.0f}"
    if baseline is None or path[0] == "system" or abs(value - baseline) < 1e-9:
        return text
    return f"{text} ({value - baseline:+.4f})"


def _dig(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if isinstance(current, int | float) else None


# Settings that change how *fast* a run is and nothing else. A cached ranking is
# the same ranking - that is the property the cache key exists to guarantee - so
# every retrieval and generation metric must come out identical with it on and
# off, and only the latency block may move. Reported for the same reason as
# GENERATION_KEYS: the verdict stays honest, the latency numbers underneath it
# are not comparable, and silence would imply they were.
LATENCY_KEYS: tuple[str, ...] = ("cache_backend", "cache_rankings")


def latency_changes(baseline: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Caching differences from the baseline's - latency context, not a verdict."""
    recorded = baseline.get("config", {})
    return [
        f"{key} {recorded.get(key)!r} -> {config[key]!r}"
        for key in LATENCY_KEYS
        if key in config and recorded.get(key) != config[key]
    ]
