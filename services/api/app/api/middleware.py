"""Cross-cutting HTTP middleware."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_ctx, user_id_ctx
from app.observability.metrics import METRICS
from app.observability.trace import clear_trace, start_trace

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
HEALTH_PREFIX = "/health"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a correlation id, bind it to the log context, time the request.

    An inbound `X-Request-ID` is honoured so a trace started by the frontend or
    a gateway continues through the API; otherwise one is minted. The id is
    echoed on every response, including error envelopes, which is what makes a
    user-reported failure findable in the logs.

    It also opens the span collector (project plan section 15), so anything
    downstream can time itself without being handed a tracer, and closes it
    afterwards so a span can never leak from one request into the next.

    Health probes are counted out of the process metrics as well as logged at
    debug. A liveness check every ten seconds is most of the traffic on an idle
    deployment, and a p50 that is mostly probe would describe nothing anybody
    wants to know.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        user_token = user_id_ctx.set(None)
        structlog.contextvars.clear_contextvars()

        is_probe = request.url.path.startswith(HEALTH_PREFIX)
        trace = start_trace(f"{request.method} {request.url.path}")

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            components = trace.self_ms()
            if not is_probe:
                METRICS.record_request(status_code=status_code, duration_ms=duration_ms)
            level = logger.debug if is_probe else logger.info
            level(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                # One line per request carrying where the time went, so the
                # breakdown survives even for requests that never persist a row.
                **({"components_ms": components} if components else {}),
            )
            clear_trace()
            request_id_ctx.reset(token)
            user_id_ctx.reset(user_token)
