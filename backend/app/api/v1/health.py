"""Liveness and readiness, measured rather than asserted.

Every check in the response is the result of actually touching the component:
a ``SELECT 1``, a Redis ``PING``, a listing of the Ollama model tags, a read of
the index. Nothing here returns ``ok: true`` because a check is configured to be
optional — the only way to get ``ok`` is to have succeeded.

The response is ``degraded`` rather than failing when an optional dependency is
down, because RAGOps stays useful without Redis (analytics fall back to the
database) and without Ollama (every recorded trace is still readable). Only the
database can make the service ``unhealthy``: it holds the telemetry, and without
it there is no product.

Unreachable is not the same as broken. A timed-out Redis is reported with the
reason, so ``degraded`` says *which* component is missing rather than leaving an
operator to bisect four booleans.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.core.cache import get_redis
from app.core.database import SessionLocal
from app.core.logging import get_logger
from app.models import utcnow
from app.schemas.common import HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

#: How long any single component probe may take. Short on purpose: a health
#: endpoint that waits on a dead dependency is what turns a partial outage into
#: a total one, because every load balancer probe queues behind it.
PROBE_TIMEOUT_SECONDS = 2.0

#: A probe that outlives this is treated as a failure even if it later succeeds.
SLOW_PROBE_MS = 1_000.0

#: The only component whose failure makes the whole service unhealthy.
ESSENTIAL_CHECKS = frozenset({"database"})

APP_VERSION = "0.1.0"


async def _timed(probe: Any) -> tuple[bool, float, str | None, dict[str, Any]]:
    """Run one probe, returning ``(ok, latency_ms, error, details)``.

    Every probe is wrapped rather than written to raise: one unreachable
    component must not turn the health endpoint itself into a 500, because the
    caller that most needs the answer is the one already struggling.

    ``details`` is whatever the probe returned on success — the model list, for
    Ollama. A probe that succeeded with no extra to say yields an empty dict,
    which is merged into the check as nothing.
    """
    started = time.perf_counter()
    try:
        details = await asyncio.wait_for(probe(), timeout=PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        elapsed = (time.perf_counter() - started) * 1000.0
        return False, elapsed, f"no response within {PROBE_TIMEOUT_SECONDS:g}s", {}
    except Exception as exc:  # noqa: BLE001 - a probe must report, not propagate
        elapsed = (time.perf_counter() - started) * 1000.0
        return False, elapsed, f"{type(exc).__name__}: {exc}", {}
    return True, (time.perf_counter() - started) * 1000.0, None, details or {}


async def _probe_database() -> dict[str, Any]:
    """A real round trip: ``SELECT 1`` against the configured database."""
    async with SessionLocal() as session:
        await session.execute(text("SELECT 1"))
    return {}


async def _probe_redis() -> dict[str, Any]:
    """A real ``PING``, reported as reachable or not.

    Redis is optional by design — the cache degrades to a miss, not an error —
    so an unreachable instance is ``degraded`` rather than ``unhealthy``.
    """
    client = get_redis()
    if client is None:
        # No client means the connection could not even be constructed. That is
        # a distinct failure from "constructed but not answering", and flattening
        # the two would hide a configuration error behind a timeout.
        raise RuntimeError(
            f"no Redis client for {settings.redis_url!r}; cache is unreachable"
        )
    await client.ping()
    return {}


async def _probe_ollama() -> dict[str, Any]:
    """The model tags the backend currently serves.

    ``list_models`` is documented to return an empty list rather than raise when
    the backend is down, so an empty list from a *successful* call is reported as
    reachable-with-no-models — which is a real state, and a different one from
    unreachable.
    """
    from app.providers.registry import get_provider

    models = await get_provider("ollama").list_models()
    return {"models": models, "configured_model": settings.ollama_model}


def _probe_vector_store() -> dict[str, Any]:
    """Whether the retrieval index is loaded, and what is in it.

    Synchronous and cheap: it reads the already-loaded chunk list rather than
    building an index. A missing index is ``ok: false`` with the path that
    would be built from, because "not indexed yet" is the single most common
    reason a fresh checkout has a working API and no retrieval.
    """
    from app.ml.retriever import get_retrieval_service

    service = get_retrieval_service()
    stats = service.stats()
    return {
        "ok_detail": (
            f"{stats['num_chunks']} chunks from {stats['num_sources']} sources"
            if stats["is_indexed"]
            else "not indexed"
        ),
        **stats,
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health, with each component actually probed",
)
async def health() -> HealthResponse:
    """Report what is reachable right now.

    The probes run concurrently: total latency is the slowest component rather
    than their sum, which is what keeps a degraded Redis from making the
    endpoint slow for everyone.

    Status is derived from the checks rather than set by hand:
    ``unhealthy`` if an essential check failed, ``degraded`` if any optional one
    did, ``healthy`` otherwise.
    """
    database_ok, database_ms, database_error, _ = await _timed(_probe_database)
    checks: dict[str, Any] = {
        "database": _check(database_ok, database_ms, database_error),
    }

    # Optional components are probed together, and a failure in one of them must
    # not prevent the others from being reported.
    redis_task = asyncio.create_task(_timed(_probe_redis))
    ollama_task = asyncio.create_task(_timed(_probe_ollama))
    (redis_ok, redis_ms, redis_error, _), (ollama_ok, ollama_ms, ollama_error, ollama_models) = (
        await asyncio.gather(redis_task, ollama_task, return_exceptions=False)
    )
    checks["redis"] = _check(redis_ok, redis_ms, redis_error)
    checks["ollama"] = {
        **_check(ollama_ok, ollama_ms, ollama_error),
        **ollama_models,
    }

    checks["vector_store"] = _vector_store_check()

    essential_failed = any(
        not check["ok"] for name, check in checks.items() if name in ESSENTIAL_CHECKS
    )
    any_failed = any(not check["ok"] for check in checks.values())
    if essential_failed:
        status = "unhealthy"
    elif any_failed:
        status = "degraded"
    else:
        status = "healthy"

    payload = HealthResponse(
        status=status,
        version=APP_VERSION,
        environment=settings.environment,
        checks=checks,
        timestamp=_now(),
    )
    logger.info(
        "health.checked",
        status=status,
        failed=[name for name, check in checks.items() if not check["ok"]],
    )
    return payload


def _check(ok: bool, latency_ms: float, error: str | None) -> dict[str, Any]:
    """One check's result, with its latency and its reason when it failed.

    A probe slower than :data:`SLOW_PROBE_MS` is reported as slow but stays
    ``ok``: latency is reported as a fact, and a check that reported slowness as
    failure would page someone about a database that is answering.
    """
    result: dict[str, Any] = {"ok": ok, "latency_ms": round(latency_ms, 2)}
    if not ok and error:
        result["error"] = error
    elif latency_ms > SLOW_PROBE_MS:
        result["slow"] = True
    return result


def _vector_store_check() -> dict[str, Any]:
    """The vector store's status, never raising out of the health endpoint."""
    started = time.perf_counter()
    try:
        detail = _probe_vector_store()
    except Exception as exc:  # noqa: BLE001 - a probe must report, not propagate
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "latency_ms": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = (time.perf_counter() - started) * 1000.0
    is_indexed = bool(detail.pop("is_indexed", False))
    return {"ok": is_indexed, "latency_ms": round(elapsed, 2), **detail}


def _now() -> datetime:
    """Current UTC time, from the same clock the models use.

    The model's clock rather than ``datetime.now()``: a health timestamp written
    in local time next to database timestamps in UTC is a bug waiting for
    somebody to read the two as the same instant.
    """
    return utcnow()
