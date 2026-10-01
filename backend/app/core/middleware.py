"""RAGOps self-instrumentation: request timing, error capture, request ids."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, new_request_id, request_id_ctx, trace_id_ctx

logger = get_logger("ragops.api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id and time every HTTP request.

    This is RAGOps observing *itself*: latency and status of the collector
    itself end up in the same log stream as everything else it processes.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_ctx.set(request_id)
        try:
            start = time.perf_counter()
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "http.request.error",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        finally:
            request_id_ctx.reset(token)


class TraceContextMiddleware(BaseHTTPMiddleware):
    """Propagate an inbound ``X-Trace-ID`` so SDK and collector logs correlate."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        inbound = request.headers.get("X-Trace-ID")
        token = trace_id_ctx.set(inbound or "-")
        try:
            return await call_next(request)
        finally:
            trace_id_ctx.reset(token)


class APIErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Return structured JSON for unhandled errors.

    Clients of a telemetry platform need a parseable failure, not an HTML
    traceback, and the body must not leak internals.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:
            logger.exception(
                "http.unhandled_error",
                method=request.method,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "code": "internal_error",
                },
            )
