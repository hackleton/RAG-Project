"""Process counters: volume, errors, cache behaviour, model calls.

These complement the durable telemetry on `chat_messages`, they do not replace
it. The database knows what every answer cost and how long it took, for as long
as the row exists; it knows nothing about the requests that never produced an
answer - health probes, 404s, a burst of 500s from a dependency outage, a cache
that is missing on everything.

**In the process, and honest about it**, on the same terms as the login
throttle in Sprint 6: with more than one replica each holds its own counts, and
a restart resets them. The dashboard therefore labels this block "since start"
and shows the process start time next to it, rather than presenting counters
that silently mean something different after a deploy. Exporting these to a
real time-series backend is the fix, and is what `snapshot()` is shaped for.

Guarded by a lock because the reranker and the embedder run on worker threads:
`dict[key] += 1` is a read and a write, and losing counts to a race would make
the cache hit rate quietly wrong in the direction that flatters it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal

# Enough to make a percentile meaningful without holding a request's worth of
# history; the durable percentiles come from SQL Server.
LATENCY_SAMPLE = 1024

CacheEvent = Literal["hit", "miss", "store", "error"]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Computed here rather than in SQL: `PERCENTILE_CONT` is a window function in
    T-SQL, which makes a grouped query awkward, and SQLite - which the test
    suite runs on - does not have it at all. Over the bounded samples this
    project reports on, sorting in Python is both simpler and portable.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    # ceil, not round: `round(n + 0.5)` looks like a ceiling and is not, because
    # Python rounds halves to even - p95 of 1..100 came out as 96.
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return round(ordered[rank - 1], 2)


class ProcessMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = datetime.now(UTC)
        self._started = time.monotonic()
        self._requests = 0
        self._by_status: dict[str, int] = {}
        self._latencies: deque[float] = deque(maxlen=LATENCY_SAMPLE)
        self._cache: dict[str, dict[str, int]] = {}
        self._llm_calls = 0
        self._llm_ms = 0.0
        self._llm_prompt_tokens = 0
        self._llm_completion_tokens = 0
        self._tool_calls: dict[str, dict[str, int]] = {}

    # --- recording --------------------------------------------------------

    def record_request(self, *, status_code: int, duration_ms: float) -> None:
        bucket = f"{status_code // 100}xx"
        with self._lock:
            self._requests += 1
            self._by_status[bucket] = self._by_status.get(bucket, 0) + 1
            self._latencies.append(duration_ms)

    def record_cache(self, namespace: str, event: CacheEvent) -> None:
        with self._lock:
            counts = self._cache.setdefault(
                namespace, {"hit": 0, "miss": 0, "store": 0, "error": 0}
            )
            counts[event] += 1

    def record_llm(
        self,
        *,
        duration_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        with self._lock:
            self._llm_calls += 1
            self._llm_ms += duration_ms
            self._llm_prompt_tokens += prompt_tokens or 0
            self._llm_completion_tokens += completion_tokens or 0

    def record_tool(self, name: str, *, ok: bool) -> None:
        with self._lock:
            counts = self._tool_calls.setdefault(name, {"calls": 0, "failures": 0})
            counts["calls"] += 1
            if not ok:
                counts["failures"] += 1

    # --- reading ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            requests = self._requests
            by_status = dict(self._by_status)
            cache = {name: dict(counts) for name, counts in self._cache.items()}
            llm = (self._llm_calls, self._llm_ms, self._llm_prompt_tokens)
            completion_tokens = self._llm_completion_tokens
            tools = {name: dict(counts) for name, counts in self._tool_calls.items()}

        failures = by_status.get("5xx", 0)
        llm_calls, llm_ms, prompt_tokens = llm
        return {
            "started_at": self._started_at.isoformat(),
            "uptime_seconds": round(time.monotonic() - self._started, 1),
            "requests": {
                "total": requests,
                "by_status_class": by_status,
                # Server faults only. A 401 or a 403 is the system working.
                "error_rate": round(failures / requests, 4) if requests else 0.0,
                "latency_ms": {
                    "p50": percentile(latencies, 0.50),
                    "p95": percentile(latencies, 0.95),
                    "max": round(max(latencies), 2) if latencies else 0.0,
                    "sampled": len(latencies),
                },
            },
            "cache": [
                {
                    "namespace": name,
                    **counts,
                    "hit_rate": (
                        round(counts["hit"] / (counts["hit"] + counts["miss"]), 4)
                        if counts["hit"] + counts["miss"]
                        else None
                    ),
                }
                for name, counts in sorted(cache.items())
            ],
            "llm": {
                "calls": llm_calls,
                "total_ms": round(llm_ms, 2),
                "mean_ms": round(llm_ms / llm_calls, 2) if llm_calls else 0.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "tools": [{"name": name, **counts} for name, counts in sorted(tools.items())],
        }

    def reset(self) -> None:
        """Start again. For the test suite; there is no endpoint for it."""
        with self._lock:
            self._started_at = datetime.now(UTC)
            self._started = time.monotonic()
            self._requests = 0
            self._by_status.clear()
            self._latencies.clear()
            self._cache.clear()
            self._llm_calls = 0
            self._llm_ms = 0.0
            self._llm_prompt_tokens = 0
            self._llm_completion_tokens = 0
            self._tool_calls.clear()


# Process-wide by design: these count what this process did, so they cannot be
# per-app-instance without meaning something different in a test than in a
# deployment. `reset()` is what the suite uses to isolate cases.
METRICS = ProcessMetrics()
