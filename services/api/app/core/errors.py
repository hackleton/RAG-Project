"""Application error types and their HTTP representation.

Handlers raise domain errors; a single set of exception handlers turns them
into the one error envelope the frontend and the evaluation harness parse:

    {"error": {"code": "...", "message": "...", "details": {...}},
     "request_id": "..."}

Unhandled exceptions never leak a stack trace or driver message to the client
- they are logged with the request id and returned as `internal_error`.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for expected, client-visible failures."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "authentication_failed"
    message = "Could not validate credentials."


class PermissionDeniedError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "permission_denied"
    message = "You do not have access to this resource."


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "Resource not found."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "Resource already exists."


class ValidationError(AppError):
    status_code = 422  # the starlette constant for this is deprecated
    code = "validation_error"
    message = "Request payload failed validation."


class ServiceUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"
    message = "A dependency is unavailable."


class NotImplementedYetError(AppError):
    """Endpoint whose shape is fixed but whose behaviour lands in a later sprint."""

    status_code = status.HTTP_501_NOT_IMPLEMENTED
    code = "not_implemented"
    message = "This endpoint is not implemented yet."

    def __init__(self, sprint: str) -> None:
        super().__init__(
            f"Not implemented yet - scheduled for {sprint}.", details={"sprint": sprint}
        )


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    request: Request | None = None,
) -> JSONResponse:
    # Prefer request.state: Starlette's ServerErrorMiddleware wraps our own, so
    # for an unhandled exception the context var has already been reset by the
    # time this runs - and that is precisely the response whose id someone will
    # quote in a bug report.
    request_id = getattr(request.state, "request_id", None) if request else None
    body: dict[str, Any] = {
        "error": {"code": code, "message": message, "details": details or {}},
        "request_id": request_id or request_id_ctx.get(),
    }
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("app_error", code=exc.code, message=exc.message)
        else:
            logger.info("app_error", code=exc.code, message=exc.message)
        return error_response(exc.status_code, exc.code, exc.message, exc.details, request)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "authentication_failed", 403: "permission_denied", 404: "not_found"}.get(
            exc.status_code, "http_error"
        )
        return error_response(exc.status_code, code, str(exc.detail), None, request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422,
            "validation_error",
            "Request payload failed validation.",
            {"errors": exc.errors()},
            request,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # exc_info goes to the log, never to the client.
        logger.exception("unhandled_exception", exc_type=type(exc).__name__)
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected error occurred.",
            None,
            request,
        )
