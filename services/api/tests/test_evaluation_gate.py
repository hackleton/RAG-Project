"""The regression gate and the experiment sweep's plumbing.

The gate is the part of Sprint 5 that can fail a build, so its failure modes
matter more than its success ones. Two in particular:

* a gate that cannot run must not report success, and
* a gate must not treat the change it exists to judge as a reason to skip.

The second is not hypothetical. An earlier version listed the retrieval
configuration among the things that had to match before two runs were
comparable, which meant dropping `RETRIEVAL_TOP_K` from 5 to 3 - a change that
costs 0.12 recall on the real dataset - was reported as "cannot compare"
instead of as a regression. `test_a_retrieval_change_is_judged_not_skipped`
is that bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.evaluation.experiments import (
    EXPERIMENTS,
    GATES,
    check_regression,
    environment_payload,
    incomparable,
    load_baseline,
    retrieval_changes,
    sweep_report,
    write_baseline,
)
from app.rag.retriever import RetrievalConfig


def metrics(**overrides: float) -> dict[str, Any]:
    base = {
        "retrieval": {"recall_at_k": 0.9, "mrr": 0.8, "ndcg_at_k": 0.85},
        "security": {"injection_blocked_rate": 1.0, "access_contained_rate": 1.0},
        "system": {"error_rate": 0.0},
    }
    for path, value in overrides.items():
        group, name = path.split("__")
        base[group][name] = value  # type: ignore[index]
    return base


def baseline(**config: Any) -> dict[str, Any]:
    return {
        "dataset_version": "v1",
        "config": {
            "embedding_provider": "hash",
            "embedding_model": "m",
            "chunk_target_tokens": 512,
            "chunk_overlap_tokens": 64,
            "retrieval": {"top_k": 5, "mode": "hybrid"},
            **config,
        },
        **metrics(),
    }


def environment(**overrides: Any) -> dict[str, Any]:
    return {
        "embedding_provider": "hash",
        "embedding_model": "m",
        "chunk_target_tokens": 512,
        "chunk_overlap_tokens": 64,
        "retrieval": {"top_k": 5, "mode": "hybrid"},
        **overrides,
    }


# --- what counts as a regression ------------------------------------------


def test_an_unchanged_run_has_no_violations() -> None:
    assert check_regression(metrics(), baseline()) == []


def test_a_drop_inside_the_tolerance_is_not_a_regression() -> None:
    """Tolerance exists so a reranker tie does not fail a build."""
    assert check_regression(metrics(retrieval__recall_at_k=0.87), baseline()) == []


def test_a_drop_past_the_tolerance_is_reported_with_both_numbers() -> None:
    violations = check_regression(metrics(retrieval__recall_at_k=0.80), baseline())

    assert [violation.metric for violation in violations] == ["retrieval.recall_at_k"]
    assert violations[0].baseline == 0.9
    assert violations[0].current == 0.80
    assert violations[0].delta == pytest.approx(-0.1)
    assert "0.9000 -> 0.8000" in str(violations[0])


def test_an_improvement_is_never_a_regression() -> None:
    assert check_regression(metrics(retrieval__recall_at_k=1.0), baseline()) == []


def test_error_rate_regresses_upwards() -> None:
    """The one gated metric where lower is better."""
    assert check_regression(metrics(system__error_rate=0.10), baseline())
    assert check_regression(metrics(system__error_rate=0.0), baseline()) == []


def test_the_security_rates_have_no_tolerance_at_all() -> None:
    """A single leaked case is a regression, whatever the tolerance says."""
    one_case_in_a_hundred = metrics(security__injection_blocked_rate=0.99)
    violations = check_regression(one_case_in_a_hundred, baseline())

    assert [violation.metric for violation in violations] == ["security.injection_blocked_rate"]
    assert violations[0].tolerance == 0.0


def test_a_metric_missing_from_the_baseline_is_skipped_not_failed() -> None:
    """A baseline predating a metric is normal; scoring it a catastrophe
    teaches people to bypass the gate."""
    older = baseline()
    del older["retrieval"]["ndcg_at_k"]

    assert check_regression(metrics(), older) == []


def test_every_gated_metric_is_present_in_a_real_metrics_payload() -> None:
    """Guards a rename: a gate pointing at a path nothing produces is silent."""
    from app.evaluation.harness import summarise

    payload = summarise([])
    assert payload == {}  # no cases -> nothing to gate, which is its own answer

    for gate in GATES:
        assert len(gate.path) == 2
        assert gate.label.startswith(("retrieval.", "generation.", "system.", "security."))


# --- comparability ---------------------------------------------------------


def test_a_matching_configuration_is_comparable() -> None:
    assert incomparable(baseline(), dataset_version="v1", config=environment()) == []


def test_a_different_dataset_version_is_not_comparable() -> None:
    reasons = incomparable(baseline(), dataset_version="v2", config=environment())
    assert reasons == ["dataset version 'v1' != 'v2'"]


@pytest.mark.parametrize("key", ["embedding_provider", "embedding_model", "chunk_target_tokens"])
def test_anything_that_changes_the_index_is_not_comparable(key: str) -> None:
    """A different index is a different system, not a regressed one."""
    reasons = incomparable(baseline(), dataset_version="v1", config=environment(**{key: "changed"}))
    assert any(reason.startswith(key) for reason in reasons)


def test_a_retrieval_change_is_judged_not_skipped() -> None:
    """The bug this file exists for.

    Changing `top_k` is exactly the change the gate is meant to judge. Listing
    the retrieval config among the comparability keys made the gate skip itself
    precisely when a developer had changed something.
    """
    config = environment(retrieval={"top_k": 3, "mode": "hybrid"})

    assert incomparable(baseline(), dataset_version="v1", config=config) == []
    assert retrieval_changes(baseline(), config) == ["top_k 5 -> 3"]
    # And the numbers still get compared.
    assert check_regression(metrics(retrieval__recall_at_k=0.8), baseline())


def test_an_unchanged_retrieval_configuration_reports_no_changes() -> None:
    assert retrieval_changes(baseline(), environment()) == []


# --- the baseline file -----------------------------------------------------


def test_a_baseline_round_trips_and_records_when_it_was_taken(tmp_path: Any) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(path, dataset_version="v1", config=environment(), metrics=metrics())

    loaded = load_baseline(path)
    assert loaded is not None
    assert loaded["dataset_version"] == "v1"
    assert loaded["config"]["embedding_provider"] == "hash"
    assert loaded["retrieval"]["recall_at_k"] == 0.9
    assert "recorded_at" in loaded  # so a stale baseline is visible as stale


def test_a_missing_baseline_is_none_rather_than_an_error(tmp_path: Any) -> None:
    assert load_baseline(tmp_path / "nothing.json") is None


# --- experiments -----------------------------------------------------------


def test_every_experiment_has_a_hypothesis_and_a_unique_name() -> None:
    """A configuration with no stated hypothesis is a number nobody can read."""
    names = [experiment.name for experiment in EXPERIMENTS]

    assert len(names) == len(set(names))
    assert "baseline" in names
    assert all(len(experiment.hypothesis) > 30 for experiment in EXPERIMENTS)


def test_an_experiment_patches_the_deployed_configuration() -> None:
    base = RetrievalConfig(top_k=5, candidate_pool=50)
    dense = next(item for item in EXPERIMENTS if item.name == "dense-only")

    applied = dense.config(base)
    assert applied.mode == "dense"
    assert applied.top_k == 5  # everything it does not name is left alone
    assert base.mode == "hybrid"  # and the deployed config is not mutated


def test_the_sweep_report_states_deltas_against_the_baseline_row() -> None:
    report = sweep_report(
        dataset_version="v1",
        environment={"embedding_model": "m"},
        results=[
            (EXPERIMENTS[0], metrics()),
            (EXPERIMENTS[1], metrics(retrieval__recall_at_k=0.80)),
        ],
    )

    assert "dataset `v1`" in report
    assert "0.9000" in report
    assert "-0.1000" in report  # the delta, not just the value
    assert EXPERIMENTS[1].hypothesis in report
    # Left blank on purpose: a generated sentence in the one field a reader is
    # entitled to trust would be a plausible-looking guess.
    assert "_To be written by whoever ran this._" in report


def test_the_report_names_the_experiments_it_could_not_run() -> None:
    """Chunk size and embedding model change the index, so a query-time sweep
    cannot reach them. Silence would read as 'considered and uninteresting'."""
    report = sweep_report(dataset_version="v1", environment={}, results=[])

    assert "chunk-size" in report
    assert "embedding-model" in report
    assert "scripts/reindex.py" in report


def test_the_environment_payload_names_both_models(settings: Settings) -> None:
    """Retrieval numbers mean nothing without the embedding model beside them."""
    payload = environment_payload(settings)

    assert payload["embedding_provider"] == settings.embedding_provider.value
    assert payload["llm_provider"] == settings.llm_provider.value
    assert payload["chunk_target_tokens"] == settings.chunk_target_tokens
