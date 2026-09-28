"""Tracing and process metrics.

Project plan section 15 asks for a trace from API entry through retrieval,
reranking, LLM calls and tool calls, with latency captured *per component*
rather than only in total. `trace` provides that; `metrics` counts what a
single row cannot show - request volume, error rate and cache behaviour since
the process started.
"""

from __future__ import annotations

from app.observability.metrics import METRICS, ProcessMetrics
from app.observability.trace import Span, Trace, current_trace, span, start_trace

__all__ = [
    "METRICS",
    "ProcessMetrics",
    "Span",
    "Trace",
    "current_trace",
    "span",
    "start_trace",
]
