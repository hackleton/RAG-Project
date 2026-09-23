"""Run the evaluation dataset against the deployed configuration.

    docker compose run --rm api python scripts/run_evaluation.py
    docker compose run --rm api python scripts/run_evaluation.py --gate
    docker compose run --rm api python scripts/run_evaluation.py --accept

`--gate` turns a regression into a non-zero exit, which is how CI uses it.
`--accept` records the current numbers as the new baseline, which is a
deliberate human act - a baseline that updated itself at the end of every run
could only ever pass.

The run goes through the same `RagService` the `/chat` endpoint uses, as the
seeded users, with their real patient grants. Nothing here is a shortcut for
the benefit of the measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api.deps import get_retrieval_config
from app.core.config import LLMProvider as LLMProviderName
from app.core.config import Settings, get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.evaluation.dataset import Dataset, load_dataset, sync_cases
from app.evaluation.experiments import (
    check_regression,
    environment_payload,
    generation_changes,
    incomparable,
    load_baseline,
    retrieval_changes,
    write_baseline,
)
from app.evaluation.grading import LlmJudge
from app.evaluation.harness import EvaluationHarness
from app.evaluation.paths import BASELINE, REPORT_DIR, dataset_path
from app.providers.embedding import build_embedding_provider
from app.providers.llm import build_llm_provider
from app.providers.reranker import build_reranker
from app.providers.vectorstore import QdrantVectorStore
from app.rag.retriever import HybridRetriever, RetrievalConfig
from app.rag.service import RagService


def build_rag(settings: Settings, config: RetrievalConfig) -> tuple[RagService, QdrantVectorStore]:
    store = QdrantVectorStore(settings)
    return (
        RagService(
            retriever=HybridRetriever(
                store=store,
                embedder=build_embedding_provider(settings),
                reranker=build_reranker(settings),
                config=config,
            ),
            llm=build_llm_provider(settings),
        ),
        store,
    )


def print_summary(metrics: dict[str, Any]) -> None:
    def block(title: str, key: str, fmt: str = "{:.4f}") -> None:
        values = metrics.get(key) or {}
        if not values:
            return
        print(f"\n  {title}")
        for name, value in values.items():
            rendered = fmt.format(value) if isinstance(value, float) else str(value)
            print(f"    {name:<28} {rendered}")

    print(f"\n{metrics['passed']}/{metrics['cases']} cases passed ({metrics['pass_rate']:.1%})")
    block("Retrieval", "retrieval")
    block("Generation", "generation")
    block("System", "system")
    block("Security", "security")

    print("\n  By category")
    for name, values in metrics.get("by_category", {}).items():
        recall = values.get("recall_at_k")
        recall_text = f"recall@k {recall:.3f}" if recall is not None else "no retrieval cases"
        print(
            f"    {name:<18} {values['cases']:>3} cases  "
            f"pass {values['pass_rate']:.0%}  {recall_text}"
        )


async def execute(
    arguments: argparse.Namespace,
    settings: Settings,
    dataset: Dataset,
    config: RetrievalConfig,
) -> tuple[dict[str, Any], str]:
    """The half that needs an event loop. Everything else stays synchronous.

    Split this way so the report and baseline files are written from ordinary
    blocking code - writing them inside the coroutine would stall the loop, and
    a script that models its own I/O badly is not a good example of the thing
    it is measuring.
    """
    init_engine(settings)
    rag, store = build_rag(settings, config)
    try:
        async with get_sessionmaker()() as session:
            if not arguments.no_sync:
                summary = await sync_cases(session, dataset)
                await session.commit()
                print(f"  cases synced: {summary}")

            harness = EvaluationHarness(
                rag=rag,
                config=config,
                judge=LlmJudge(build_llm_provider(settings)) if arguments.judge else None,
            )
            run = await harness.run(
                session,
                dataset,
                environment=environment_payload(settings),
                limit=arguments.limit,
            )
            return dict(run.metrics), str(run.id)
    finally:
        await store.close()
        await dispose_engine()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_path())
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing cases first.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also score with the LLM judge. Requires a real LLM provider.",
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--gate", action="store_true", help="Exit non-zero on a regression.")
    parser.add_argument("--accept", action="store_true", help="Record this run as the baseline.")
    arguments = parser.parse_args()

    settings = get_settings()
    dataset = load_dataset(arguments.dataset)
    config = get_retrieval_config(settings)

    if arguments.judge and settings.llm_provider is LLMProviderName.ECHO:
        print("--judge needs a real LLM provider; the echo stub cannot grade.", file=sys.stderr)
        return 1

    print(f"Dataset '{dataset.version}': {len(dataset)} case(s) from {dataset.path}")
    print(f"  categories:  {dataset.counts_by_category}")
    print(
        f"  retrieval:   mode={config.mode} top_k={config.top_k} "
        f"pool={config.candidate_pool} rerank={config.rerank}"
    )
    print(f"  generation:  {settings.llm_provider.value} ({settings.llm_model})")

    metrics, run_id = asyncio.run(execute(arguments, settings, dataset, config))
    print_summary(metrics)

    environment = environment_payload(settings) | {"retrieval": dataclasses.asdict(config)}
    report_path = write_report(
        dataset_version=dataset.version, run_id=run_id, environment=environment, metrics=metrics
    )
    print(f"\nRun {run_id} written to {report_path}")

    if arguments.accept:
        write_baseline(
            arguments.baseline,
            dataset_version=dataset.version,
            config=environment,
            metrics=metrics,
        )
        print(f"Baseline updated: {arguments.baseline}")
        return 0

    baseline = load_baseline(arguments.baseline)
    if baseline is None:
        print(f"\nNo baseline at {arguments.baseline}. Record one with --accept.")
        return 1 if arguments.gate else 0

    reasons = incomparable(baseline, dataset_version=dataset.version, config=environment)
    if reasons:
        # Comparing across dataset versions or embedding models produces a
        # number that looks like a regression and means nothing. Under --gate
        # this is fatal rather than skipped: a gate that cannot run has not
        # passed, and silently returning 0 is how a check stops checking.
        print(f"\nCannot compare against {arguments.baseline}:", file=sys.stderr)
        for reason in reasons:
            print(f"  {reason}", file=sys.stderr)
        print("  Record a baseline for this configuration with --accept.", file=sys.stderr)
        return 1 if arguments.gate else 0

    changed = retrieval_changes(baseline, environment)
    if changed:
        # Printed before the verdict, because it changes how the verdict reads.
        print(f"\nRetrieval settings differ from {arguments.baseline}:")
        for change in changed:
            print(f"  {change}")

    generation = generation_changes(baseline, environment)
    if generation:
        # The gated metrics are all upstream of the model, or guaranteed by the
        # access filter whatever writes the answer, so the verdict below still
        # holds. The generation block above it does not compare across models,
        # and saying nothing implied that it did.
        print(f"\nGeneration model differs from {arguments.baseline}:")
        for change in generation:
            print(f"  {change}")
        print("  Retrieval and security still compare; generation metrics do not.")

    violations = check_regression(metrics, baseline)
    if not violations:
        print(
            f"\nNo regression against {arguments.baseline} "
            f"(recorded {baseline.get('recorded_at')})."
        )
        return 0

    print(f"\nRegression against {arguments.baseline}:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    # Without --gate a regression is reported and tolerated, so running this by
    # hand tells you what moved without refusing to finish. CI passes --gate.
    return 1 if arguments.gate else 0


def write_report(
    *, dataset_version: str, run_id: str, environment: dict[str, Any], metrics: dict[str, Any]
) -> Path:
    """One JSON file per run, alongside the baseline it will be compared to."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"run-{stamp}-{dataset_version}.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_version": dataset_version,
                "config": environment,
                **metrics,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    sys.exit(main())
