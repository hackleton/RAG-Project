"""Cost and latency payloads.

Split into two blocks that mean genuinely different things, because a dashboard
that mixes them is a dashboard that gets misread:

* `answers` and `daily` come from `chat_messages` - durable, survive a restart,
  cover exactly what was served, and go back as far as the retention window.
* `process` comes from in-memory counters - request volume, error rate and
  cache behaviour **since this process started**, which nothing else can see.

The UI labels them separately and shows the process start time, so nobody reads
"12 requests" after a deploy as a quiet afternoon.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class LatencyPercentiles(BaseModel):
    p50: float
    p95: float
    p99: float
    max: float
    count: int


class ModelUsage(BaseModel):
    """Per-model totals. `cost_usd` is zero for local models, by definition.

    The field exists anyway: the accounting it feeds is what an Azure OpenAI
    deployment would populate, and a cost column added the day it is first
    needed is a column nobody has checked.
    """

    model: str
    answers: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: LatencyPercentiles


class ComponentLatency(BaseModel):
    """Where the time went, summed over the sampled answers.

    Self time - a component's own cost with its children subtracted - so the
    shares add up to the traced total rather than double-counting nesting.
    """

    name: str
    calls: int
    total_ms: float
    mean_ms: float
    share: float


class AnswerMetrics(BaseModel):
    answers: int
    latency_ms: LatencyPercentiles
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    models: list[ModelUsage]
    components: list[ComponentLatency]
    # How many rows the component breakdown was computed over. Bounded, so a
    # long window does not mean an unbounded read.
    sampled: int


class DailyPoint(BaseModel):
    day: date
    answers: int
    p95_ms: float
    tokens: int
    cost_usd: float


class CacheNamespaceStats(BaseModel):
    namespace: str
    hit: int
    miss: int
    store: int
    error: int
    hit_rate: float | None


class CacheStatus(BaseModel):
    backend: str
    enabled: bool
    # Whether every replica sees the same cache. False means the ranking cache
    # and the login limiter are per-process, which changes what both promise.
    shared: bool
    reachable: bool | None
    namespaces: list[CacheNamespaceStats]


class RequestMetrics(BaseModel):
    total: int
    by_status_class: dict[str, int]
    # Server faults only. A 401 or a 403 is the system working as designed.
    error_rate: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    sampled: int


class ToolUsage(BaseModel):
    name: str
    calls: int
    failures: int


class ProcessMetricsRead(BaseModel):
    started_at: datetime
    uptime_seconds: float
    requests: RequestMetrics
    llm_calls: int
    llm_mean_ms: float
    tools: list[ToolUsage]


class MetricsOverview(BaseModel):
    generated_at: datetime
    window_days: int
    since: datetime
    answers: AnswerMetrics
    daily: list[DailyPoint]
    cache: CacheStatus
    process: ProcessMetricsRead
