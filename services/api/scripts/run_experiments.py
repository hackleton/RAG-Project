"""Sweep retrieval configurations over the evaluation dataset.

Project plan section 14 asks for vector-only versus hybrid, hybrid with and
without semantic ranking, and top-K / candidate-pool tuning - each recorded
with its hypothesis, configuration, dataset version and metrics. This runs all
of them against the same dataset in one process and writes the comparison to
`evaluation/experiments/`.

The two experiments the plan lists that are *not* here - chunk size and
embedding model - change the index rather than the query, so they cannot be
swept without re-ingesting. The generated report names them and gives the
command, rather than leaving a reader to assume they were considered and found
uninteresting.

    docker compose run --rm api python scripts/run_experiments.py
    docker compose run --rm api python scripts/run_experiments.py --only baseline,dense-only

Expect this to take a while: it is the whole dataset once per experiment, and
the reranker is a cross-encoder call per candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api.deps import get_retrieval_config
from app.core.config import get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.evaluation.dataset import load_dataset, sync_cases
from app.evaluation.experiments import (
    EXPERIMENTS,
    Experiment,
    environment_payload,
    sweep_report,
)
from app.evaluation.harness import EvaluationHarness
from app.evaluation.paths import EXPERIMENT_DIR, dataset_path
from app.providers.embedding import build_embedding_provider
from app.providers.llm import build_llm_provider
from app.providers.reranker import build_reranker
from app.providers.vectorstore import QdrantVectorStore
from app.rag.retriever import HybridRetriever
from app.rag.service import RagService

Results = list[tuple[Experiment, dict[str, Any]]]


async def sweep(
    dataset_file: Path, experiments: list[Experiment], limit: int | None
) -> tuple[Results, dict[str, Any]]:
    settings = get_settings()
    dataset = load_dataset(dataset_file)
    base = get_retrieval_config(settings)
    environment = environment_payload(settings)

    print(f"Dataset '{dataset.version}': {len(dataset)} case(s)")
    print(f"Sweeping {len(experiments)} configuration(s) - this runs the dataset once each.\n")

    init_engine(settings)
    # The vector store, embedder and reranker are built once and shared across
    # every experiment. Rebuilding them per configuration would reload the
    # cross-encoder weights eight times and change nothing about the results.
    store = QdrantVectorStore(settings)
    embedder = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    llm = build_llm_provider(settings)

    results: Results = []
    try:
        async with get_sessionmaker()() as session:
            await sync_cases(session, dataset)
            await session.commit()

            for experiment in experiments:
                config = experiment.config(base)
                rag = RagService(
                    retriever=HybridRetriever(
                        store=store, embedder=embedder, reranker=reranker, config=config
                    ),
                    llm=llm,
                )
                print(f"- {experiment.name:<18} {dataclasses.asdict(config)}")
                run = await EvaluationHarness(rag=rag, config=config).run(
                    session,
                    dataset,
                    environment=environment | {"experiment": experiment.name},
                    limit=limit,
                )
                metrics = dict(run.metrics)
                results.append((experiment, metrics))
                retrieval = metrics.get("retrieval", {})
                print(
                    f"    recall@k {retrieval.get('recall_at_k', 0):.3f}  "
                    f"mrr {retrieval.get('mrr', 0):.3f}  "
                    f"ndcg {retrieval.get('ndcg_at_k', 0):.3f}  "
                    f"p95 {metrics.get('system', {}).get('p95_ms', 0):.0f}ms"
                )
    finally:
        await store.close()
        await dispose_engine()

    return results, environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_path())
    parser.add_argument(
        "--only", default=None, help="Comma-separated experiment names. Default: all of them."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases of each."
    )
    arguments = parser.parse_args()

    wanted = {name.strip() for name in arguments.only.split(",")} if arguments.only else None
    experiments = [item for item in EXPERIMENTS if wanted is None or item.name in wanted]
    if not experiments:
        known = ", ".join(item.name for item in EXPERIMENTS)
        print(f"No experiment matches --only={arguments.only}. Known: {known}", file=sys.stderr)
        return 1

    # The sweep is the only part that needs an event loop; the report is written
    # from ordinary blocking code afterwards.
    results, environment = asyncio.run(sweep(arguments.dataset, experiments, arguments.limit))

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = EXPERIMENT_DIR / f"sweep-{stamp}-{arguments.dataset.stem}.md"
    path.write_text(
        sweep_report(
            dataset_version=arguments.dataset.stem, environment=environment, results=results
        ),
        encoding="utf-8",
        newline="\n",
    )

    print(f"\nWrote {path}")
    print("Interpretation and decision are left blank - they are yours to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
