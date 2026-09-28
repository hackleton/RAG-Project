"""Cost and latency, read back out of what was served.

Project plan section 20 wants "latency and token/cost telemetry are visible",
and section 15 wants latency per component rather than only per request. Both
are answered from the telemetry Sprint 3 started storing on every assistant
message, plus the process counters Sprint 7 added for the requests that never
produce a message.

**Admin only.** The aggregates name no patient and no user, so this is less
sensitive than the audit trail - but a per-user latency and cost breakdown is
still a view of what colleagues are doing, and "less sensitive" is not a role.
Unlike `/audit`, reading it is not itself audited: there is nothing here to
disclose, so a row per read would be noise in a trail whose value is that
everything in it matters.

**One bounded query, and the percentiles are computed in Python.** T-SQL's
`PERCENTILE_CONT` is a window function, which makes it awkward to group, and
SQLite - which the test suite runs on - does not have it at all. A percentile
that is computed differently in the suite from in production is a percentile
nobody should trust, so it is computed the same way in both.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select

from app.api.deps import CacheDep, DbSession, SettingsDep, require_roles
from app.models.chat import ChatMessage
from app.models.enums import ChatRole, Role
from app.observability.metrics import METRICS, percentile
from app.schemas.metrics import (
    AnswerMetrics,
    CacheNamespaceStats,
    CacheStatus,
    ComponentLatency,
    DailyPoint,
    LatencyPercentiles,
    MetricsOverview,
    ModelUsage,
    ProcessMetricsRead,
    RequestMetrics,
    ToolUsage,
)

router = APIRouter(prefix="/metrics", tags=["metrics"])

# How many recent answers the detail - percentiles, per-model, per-component
# and the daily series - is computed over. The totals are exact for the whole
# window; this bounds the read, and the response says what it covered.
SAMPLE_LIMIT = 2000

UNKNOWN_MODEL = "unknown"


@router.get(
    "/overview",
    response_model=MetricsOverview,
    summary="Latency, tokens, cost and cache behaviour",
    dependencies=[Depends(require_roles(Role.ADMIN))],
)
async def overview(
    session: DbSession,
    settings: SettingsDep,
    cache: CacheDep,
    days: int = Query(7, ge=1, le=365, description="How far back to aggregate."),
) -> MetricsOverview:
    since = datetime.now(UTC) - timedelta(days=days)

    totals = (
        await session.execute(
            select(
                func.count(ChatMessage.id),
                func.sum(ChatMessage.prompt_tokens),
                func.sum(ChatMessage.completion_tokens),
                func.sum(ChatMessage.estimated_cost_usd),
            ).where(ChatMessage.role == ChatRole.ASSISTANT, ChatMessage.created_at >= since)
        )
    ).one()

    sample = (
        await session.execute(
            select(
                ChatMessage.created_at,
                ChatMessage.latency_ms,
                ChatMessage.prompt_tokens,
                ChatMessage.completion_tokens,
                ChatMessage.estimated_cost_usd,
                ChatMessage.retrieval_trace,
            )
            .where(ChatMessage.role == ChatRole.ASSISTANT, ChatMessage.created_at >= since)
            .order_by(ChatMessage.created_at.desc())
            .limit(SAMPLE_LIMIT)
        )
    ).all()

    return MetricsOverview(
        generated_at=datetime.now(UTC),
        window_days=days,
        since=since,
        answers=AnswerMetrics(
            answers=int(totals[0] or 0),
            latency_ms=_percentiles([row.latency_ms for row in sample]),
            prompt_tokens=int(totals[1] or 0),
            completion_tokens=int(totals[2] or 0),
            cost_usd=round(float(totals[3] or 0.0), 6),
            models=_by_model(sample),
            components=_components(sample),
            sampled=len(sample),
        ),
        daily=_daily(sample),
        cache=await _cache_status(settings, cache),
        process=_process(),
    )


# --- shaping ---------------------------------------------------------------


def _percentiles(values: Sequence[int | None]) -> LatencyPercentiles:
    present = [float(value) for value in values if value is not None]
    return LatencyPercentiles(
        p50=percentile(present, 0.50),
        p95=percentile(present, 0.95),
        p99=percentile(present, 0.99),
        max=round(max(present), 2) if present else 0.0,
        count=len(present),
    )


def _model_of(trace: dict[str, Any] | None) -> str:
    """Which model wrote this answer.

    Stored inside the trace rather than in a column, because it is a property
    of how the answer was produced rather than of the message. A row written
    before the field existed reads as unknown rather than being dropped -
    losing rows from a cost report is worse than labelling them.
    """
    if not isinstance(trace, dict):
        return UNKNOWN_MODEL
    model = trace.get("model")
    return model if isinstance(model, str) and model else UNKNOWN_MODEL


def _by_model(sample: Sequence[Any]) -> list[ModelUsage]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in sample:
        grouped[_model_of(row.retrieval_trace)].append(row)

    usage = [
        ModelUsage(
            model=model,
            answers=len(rows),
            prompt_tokens=sum(row.prompt_tokens or 0 for row in rows),
            completion_tokens=sum(row.completion_tokens or 0 for row in rows),
            cost_usd=round(sum(row.estimated_cost_usd or 0.0 for row in rows), 6),
            latency_ms=_percentiles([row.latency_ms for row in rows]),
        )
        for model, rows in grouped.items()
    ]
    return sorted(usage, key=lambda item: item.answers, reverse=True)


def _components(sample: Sequence[Any]) -> list[ComponentLatency]:
    totals: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)

    for row in sample:
        trace = row.retrieval_trace
        components = trace.get("components_ms") if isinstance(trace, dict) else None
        if not isinstance(components, dict):
            continue
        for name, value in components.items():
            if isinstance(value, int | float):
                totals[name] += float(value)
                calls[name] += 1

    traced = sum(totals.values())
    breakdown = [
        ComponentLatency(
            name=name,
            calls=calls[name],
            total_ms=round(total, 2),
            mean_ms=round(total / calls[name], 2) if calls[name] else 0.0,
            share=round(total / traced, 4) if traced else 0.0,
        )
        for name, total in totals.items()
    ]
    return sorted(breakdown, key=lambda item: item.total_ms, reverse=True)


def _daily(sample: Sequence[Any]) -> list[DailyPoint]:
    """Grouped in Python, not with a date cast.

    `CAST(x AS DATE)` and `date(x)` are different functions on SQL Server and
    SQLite, and the difference would only ever show up as a test that passes
    against a dialect nothing is deployed on.
    """
    grouped: dict[date, list[Any]] = defaultdict(list)
    for row in sample:
        grouped[row.created_at.date()].append(row)

    return [
        DailyPoint(
            day=day,
            answers=len(rows),
            p95_ms=percentile(
                [float(row.latency_ms) for row in rows if row.latency_ms is not None], 0.95
            ),
            tokens=sum((row.prompt_tokens or 0) + (row.completion_tokens or 0) for row in rows),
            cost_usd=round(sum(row.estimated_cost_usd or 0.0 for row in rows), 6),
        )
        for day, rows in sorted(grouped.items())
    ]


async def _cache_status(settings: Any, cache: Any) -> CacheStatus:
    snapshot = METRICS.snapshot()
    return CacheStatus(
        backend=settings.cache_backend.value,
        enabled=settings.cache_enabled,
        shared=settings.shared_cache,
        # A live check, so the dashboard distinguishes "nothing was cached" from
        # "the cache is unreachable and every read is failing open".
        reachable=await cache.ping() if settings.cache_enabled else None,
        namespaces=[CacheNamespaceStats(**row) for row in snapshot["cache"]],
    )


def _process() -> ProcessMetricsRead:
    snapshot = METRICS.snapshot()
    requests = snapshot["requests"]
    return ProcessMetricsRead(
        started_at=datetime.fromisoformat(snapshot["started_at"]),
        uptime_seconds=snapshot["uptime_seconds"],
        requests=RequestMetrics(
            total=requests["total"],
            by_status_class=requests["by_status_class"],
            error_rate=requests["error_rate"],
            p50_ms=requests["latency_ms"]["p50"],
            p95_ms=requests["latency_ms"]["p95"],
            max_ms=requests["latency_ms"]["max"],
            sampled=requests["latency_ms"]["sampled"],
        ),
        llm_calls=snapshot["llm"]["calls"],
        llm_mean_ms=snapshot["llm"]["mean_ms"],
        tools=[ToolUsage(**row) for row in snapshot["tools"]],
    )
