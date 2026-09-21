"""Structured logging.

Every log line carries the request-scoped correlation id (bound by
`RequestContextMiddleware`) so one request can be followed across
API -> retrieval -> LLM -> tool boundaries. This is the foundation the
OpenTelemetry tracing in Sprint 7 hangs off.

Security rule (project plan section 15): never log raw document content,
prompts containing patient data, or credentials. Log identifiers and counts.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)


def _inject_request_context(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    if (request_id := request_id_ctx.get()) is not None:
        event_dict.setdefault("request_id", request_id)
    if (user_id := user_id_ctx.get()) is not None:
        event_dict.setdefault("user_id", user_id)
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Route stdlib logging and structlog through one processor chain.

    Idempotent: safe to call from both the app factory and the test fixtures.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # One handler for everything, so third-party libraries emit the same shape.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Our middleware already emits one structured line per request; uvicorn's
    # plain-text access log would only duplicate it.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
