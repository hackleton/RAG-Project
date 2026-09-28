"""Per-request tracing.

    with span("retrieval.dense", pool=50) as record:
        ...
        record.attributes["candidates"] = len(results)

Project plan section 15 wants latency captured per component, not just per
request. A total of 31 seconds tells you the system is slow; a total of 31
seconds made of 180ms retrieval and 30.9s generation tells you where to look,
and is the difference between "add a cache" and "add the *right* cache".

**Not OpenTelemetry, deliberately.** The SDK plus an OTLP exporter plus a
collector is three moving parts to run a local stack, and the span data would
land somewhere this project has no backend for. The span model here carries the
same semantics - name, parent, duration, attributes - so an OTLP exporter is an
adapter over `Trace.spans` rather than a rewrite. What it buys instead is that
the trace is a plain dict, which means it can be *stored on the message row it
describes* and read back later by the evaluation harness and the cost
dashboard. A span shipped to a backend nobody runs would not be.

Two properties worth stating:

**A span is recorded when it starts, not when it ends.** A request that raises
half way through still shows which component it was inside, which is exactly
the trace you want when something failed.

**Tracing is optional and silent.** `span()` outside a request is a no-op that
still yields a usable object, so the same instrumented code runs in a script,
in the evaluation harness and under a test with no trace active.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# A runaway loop must not be able to grow the persisted trace without bound.
# Beyond this the trace counts what it dropped rather than silently truncating.
MAX_SPANS = 250


@dataclass(slots=True)
class Span:
    """One timed component, with its position in the tree kept as a depth."""

    name: str
    start_ms: float
    duration_ms: float = 0.0
    depth: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "start_ms": round(self.start_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "depth": self.depth,
        }
        if self.attributes:
            payload["attributes"] = self.attributes
        if self.failed:
            payload["failed"] = True
        return payload


@dataclass(slots=True)
class Trace:
    """The spans of one request, in the order they started."""

    name: str
    _started: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    dropped: int = 0
    _depth: int = 0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def open(self, name: str, attributes: dict[str, Any]) -> Span:
        record = Span(name=name, start_ms=self.elapsed_ms, depth=self._depth, attributes=attributes)
        if len(self.spans) < MAX_SPANS:
            self.spans.append(record)
        else:
            self.dropped += 1
        self._depth += 1
        return record

    def close(self, record: Span) -> None:
        self._depth = max(0, self._depth - 1)
        record.duration_ms = self.elapsed_ms - record.start_ms

    def self_ms(self) -> dict[str, float]:
        """Time spent *in* each component, with its children subtracted.

        Inclusive totals are the wrong shape for a breakdown: `retrieval` at
        200ms and `retrieval.rerank` at 180ms inside it add up to 380ms of a
        200ms request, and a dashboard drawn from that is simply wrong. Self
        time sums to the traced total, so the share of each component is a real
        share.
        """
        totals: dict[str, float] = {}
        for index, record in enumerate(self.spans):
            children = 0.0
            for other in self.spans[index + 1 :]:
                if other.depth <= record.depth:
                    break
                if other.depth == record.depth + 1:
                    children += other.duration_ms
            totals[record.name] = totals.get(record.name, 0.0) + max(
                0.0, record.duration_ms - children
            )
        return {name: round(value, 2) for name, value in totals.items()}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_ms": round(self.elapsed_ms, 2),
            "components_ms": self.self_ms(),
            "spans": [record.to_dict() for record in self.spans],
        }
        if self.dropped:
            payload["dropped_spans"] = self.dropped
        return payload


_current: ContextVar[Trace | None] = ContextVar("trace", default=None)


def start_trace(name: str) -> Trace:
    """Begin a trace on this context. Replaces any trace already there."""
    trace = Trace(name=name)
    _current.set(trace)
    return trace


def clear_trace() -> None:
    _current.set(None)


def current_trace() -> Trace | None:
    return _current.get()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Time a component, and attach whatever the caller learns about it.

    The yielded span is always real, so `record.attributes[...] = ...` needs no
    guard at the call site even when nothing is collecting.

    Attributes are identifiers, counts and outcomes - never document text, a
    prompt or a query. Project plan section 15, same rule as the audit trail.
    """
    trace = _current.get()
    if trace is None:
        yield Span(name=name, start_ms=0.0, attributes=attributes)
        return

    record = trace.open(name, attributes)
    try:
        yield record
    except BaseException:
        record.failed = True
        raise
    finally:
        trace.close(record)
