"""Structured JSON logging plus request/DB/evaluation instrumentation.

RAGOps observes LLM systems, so it has to hold itself to the same standard:
every log line is JSON so it can be shipped to Loki/Datadog/CloudWatch, and
the middleware below records RAGOps' own latency and error rate rather than
leaving the platform unmeasured.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

from app.config import settings

# Correlation id for the in-flight request, attached to every log line emitted
# while handling it.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="-")

_configured = False


def _add_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    event_dict.setdefault("request_id", request_id_ctx.get())
    if trace_id_ctx.get() != "-":
        event_dict.setdefault("trace_id", trace_id_ctx.get())
    return event_dict


def configure_logging() -> None:
    """Install structlog processors. Idempotent; called from the app lifespan."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # uvicorn installs its own handlers; route them through structlog instead so
    # access logs come out as JSON too.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(noisy)
        lg.handlers.clear()
        lg.propagate = True

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any
    if settings.environment == "development" and settings.debug:
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class Timer:
    """Context manager returning elapsed wall time in milliseconds.

    Used instead of a decorator everywhere latency is recorded, because the
    value is usually needed inline in the emitted payload.
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def log_db_query(
    operation: str, duration_ms: float, *, error: str | None = None
) -> None:
    """Record a database timing so slow queries are visible in the log stream."""
    payload: dict[str, Any] = {
        "operation": operation,
        "duration_ms": round(duration_ms, 3),
    }
    # Anything over 500ms is worth surfacing; below that the volume is noise.
    #
    # `BoundLogger.log` takes an *int* level (`logging.WARNING`), not the method
    # name. Passing "warning" makes structlog compare a str against an int and
    # raise -- which would turn every logged query into a 500. The named
    # methods are used instead, and the event name is the positional argument
    # rather than a payload key so the two cannot collide.
    logger = get_logger("ragops.db")
    if error:
        logger.warning("db.query.error", error=error, **payload)
    elif duration_ms > 500:
        logger.warning("db.query.slow", **payload)
    else:
        logger.debug("db.query", **payload)


def log_evaluation(
    run_id: str,
    evaluation_type: str,
    duration_ms: float,
    num_items: int,
    *,
    error: str | None = None,
) -> None:
    """Record evaluation execution time and volume (section 36)."""
    get_logger("ragops.evaluation").info(
        "evaluation.completed",
        run_id=run_id,
        evaluation_type=evaluation_type,
        duration_ms=round(duration_ms, 2),
        num_items=num_items,
        error=error,
    )
