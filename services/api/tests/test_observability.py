"""Spans, self time, and the counters the dashboard reads.

The interesting property is `self_ms`: a breakdown whose parts do not add up to
the whole is a breakdown that sends people to optimise the wrong component.
"""

from __future__ import annotations

import time

import pytest

from app.observability.metrics import ProcessMetrics, percentile
from app.observability.trace import MAX_SPANS, clear_trace, current_trace, span, start_trace


@pytest.fixture(autouse=True)
def _isolated_trace() -> object:
    """No trace leaks into the next case, the way the middleware guarantees."""
    clear_trace()
    yield None
    clear_trace()


# --- spans ------------------------------------------------------------------


def test_a_span_outside_a_request_is_a_usable_no_op() -> None:
    """The same instrumented code runs in a script and in the harness."""
    assert current_trace() is None
    with span("retrieval.dense", pool=50) as record:
        record.attributes["candidates"] = 3
    assert current_trace() is None


def test_a_child_s_time_is_subtracted_from_its_parent() -> None:
    """Inclusive totals would report 200ms of work inside a 100ms request."""
    trace = start_trace("test")
    with span("outer"):
        time.sleep(0.01)
        with span("inner"):
            time.sleep(0.03)

    components = trace.self_ms()
    assert set(components) == {"outer", "inner"}
    # The child took most of it, and the parent is charged only its own work.
    assert components["inner"] > components["outer"]
    assert components["outer"] + components["inner"] == pytest.approx(
        trace.spans[0].duration_ms, abs=1.0
    )


def test_two_calls_to_the_same_component_are_summed() -> None:
    trace = start_trace("test")
    for _ in range(3):
        with span("llm.complete"):
            pass
    assert list(trace.self_ms()) == ["llm.complete"]


def test_siblings_are_not_treated_as_children() -> None:
    trace = start_trace("test")
    with span("parent"):
        with span("first"):
            pass
        with span("second"):
            pass

    assert [record.depth for record in trace.spans] == [0, 1, 1]
    assert set(trace.self_ms()) == {"parent", "first", "second"}


def test_a_span_that_raised_is_recorded_as_failed() -> None:
    """The trace you want when something failed says where it was."""
    trace = start_trace("test")
    with pytest.raises(RuntimeError), span("retrieval.dense"):
        raise RuntimeError("qdrant is down")

    assert trace.spans[0].failed is True
    assert trace.to_dict()["spans"][0]["failed"] is True


def test_the_span_list_is_bounded() -> None:
    """A loop must not be able to grow the JSON stored on every message."""
    trace = start_trace("test")
    for index in range(MAX_SPANS + 25):
        with span(f"step.{index}"):
            pass

    assert len(trace.spans) == MAX_SPANS
    assert trace.to_dict()["dropped_spans"] == 25


def test_the_trace_never_carries_content() -> None:
    """Section 15: identifiers, counts and outcomes - never text."""
    trace = start_trace("test")
    with span("llm.complete", model="phi3") as record:
        record.attributes["prompt_chars"] = 4096
    payload = trace.to_dict()["spans"][0]["attributes"]
    assert payload == {"model": "phi3", "prompt_chars": 4096}


# --- counters ---------------------------------------------------------------


def test_the_percentile_is_the_nearest_rank() -> None:
    values = [float(n) for n in range(1, 101)]
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile([], 0.95) == 0.0
    assert percentile([7.0], 0.99) == 7.0


def test_only_server_faults_count_as_errors() -> None:
    """A 401 or a 403 is the system working, not an error rate."""
    metrics = ProcessMetrics()
    for status in (200, 401, 403, 404):
        metrics.record_request(status_code=status, duration_ms=10.0)
    metrics.record_request(status_code=500, duration_ms=10.0)

    snapshot = metrics.snapshot()
    assert snapshot["requests"]["total"] == 5
    assert snapshot["requests"]["by_status_class"] == {"2xx": 1, "4xx": 3, "5xx": 1}
    assert snapshot["requests"]["error_rate"] == 0.2


def test_the_hit_rate_is_none_until_something_was_looked_up() -> None:
    """Zero would read as "the cache never helps", which is a different claim."""
    metrics = ProcessMetrics()
    metrics.record_cache("ranking", "store")
    assert metrics.snapshot()["cache"][0]["hit_rate"] is None

    metrics.record_cache("ranking", "hit")
    metrics.record_cache("ranking", "miss")
    assert metrics.snapshot()["cache"][0]["hit_rate"] == 0.5


def test_a_denied_tool_call_is_counted_as_a_failure() -> None:
    metrics = ProcessMetrics()
    metrics.record_tool("run_sql", ok=True)
    metrics.record_tool("run_sql", ok=False)
    assert metrics.snapshot()["tools"] == [{"name": "run_sql", "calls": 2, "failures": 1}]
