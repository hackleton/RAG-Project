"""Evaluation: the dataset, the metrics, the runner and the regression gate.

Project plan section 13 in four pieces, each usable on its own:

* `dataset` - the versioned question set, its schema and its database sync.
* `metrics` - pure retrieval arithmetic (Recall@K, Precision@K, MRR, NDCG) and
  latency percentiles. No I/O, so the numbers can be checked against worked
  examples rather than against yesterday's pipeline.
* `grading` - deterministic generation scoring, with an opt-in LLM judge for
  the qualities determinism cannot reach.
* `harness` / `experiments` - running a dataset against a configuration,
  sweeping configurations, and failing a build when a gated metric drops.
"""

from app.evaluation.dataset import CaseCategory, CaseSpec, Dataset, load_dataset, sync_cases
from app.evaluation.harness import EvaluationHarness, summarise
from app.evaluation.metrics import SourceRef, score_ranking

__all__ = [
    "CaseCategory",
    "CaseSpec",
    "Dataset",
    "EvaluationHarness",
    "SourceRef",
    "load_dataset",
    "score_ranking",
    "summarise",
    "sync_cases",
]
