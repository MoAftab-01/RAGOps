"""Collector observability and lifecycle control.

The buffered collector in :mod:`app.telemetry.collector` already owns the queue,
the retry budget, and the flush loop; this router only exposes what it reports and
lets an operator start or stop it. The counters it returns are monotonic since
process start, which is what makes ``dropped`` meaningful — a collector that has
dropped nothing since boot has genuinely dropped nothing, as opposed to one that
was restarted to hide the number.

``GET`` is open so the dashboard renders with zero setup. ``POST`` and
``DELETE`` are writes and require the API key: stopping the collector drops
whatever is buffered, and that decision should not be reachable by anyone who
can load a page.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import WriteGuard
from app.core.logging import get_logger
from app.telemetry.collector import CollectorStats, get_collector, start_collector, stop_collector

logger = get_logger(__name__)

router = APIRouter(prefix="/collector", tags=["collector"])


class CollectorStatus(BaseModel):
    """Collector counters plus the current queue depth.

    ``queue_depth`` against ``queue_capacity`` is the pair to watch: depth pinned
    at capacity means writes are arriving faster than the flush loop persists
    them, and the gap is going to ``dropped`` rather than to a slower ingest.
    """

    enabled: bool
    running: bool
    queue_depth: int
    queue_capacity: int
    enqueued: int
    flushed: int
    dropped: int
    failed_flushes: int
    retries: int
    flush_count: int
    last_flush_at: str | None = Field(
        default=None, description="ISO-8601 time of the last flush attempt"
    )
    last_error: str | None = None

    @classmethod
    def from_stats(cls, stats: CollectorStats) -> "CollectorStatus":
        return cls(
            enabled=stats.enabled,
            running=stats.running,
            queue_depth=stats.queue_depth,
            queue_capacity=stats.queue_capacity,
            enqueued=stats.enqueued,
            flushed=stats.flushed,
            dropped=stats.dropped,
            failed_flushes=stats.failed_flushes,
            retries=stats.retries,
            flush_count=stats.flush_count,
            last_flush_at=(
                stats.last_flush_at.isoformat() if stats.last_flush_at is not None else None
            ),
            last_error=stats.last_error,
        )


@router.get(
    "/stats",
    response_model=CollectorStatus,
    summary="Collector counters",
)
async def collector_stats() -> CollectorStatus:
    """Current collector counters.

    ``stats()`` is documented never to raise, which is what lets this be a plain
    dependency-free endpoint: a collector that is wedged still reports its
    counters rather than turning the endpoint into a second failure.
    """
    return CollectorStatus.from_stats(get_collector().stats())


@router.post(
    "/start",
    response_model=CollectorStatus,
    summary="Start the buffered collector",
    dependencies=[WriteGuard],
)
async def start_buffered_collector() -> CollectorStatus:
    """Start the background flush task. Idempotent.

    A collector that was disabled in settings is not started by this call: the
    disable is a configuration decision, and an API call quietly overriding it
    would make the setting meaningless.
    """
    collector = await start_collector()
    stats = collector.stats()
    logger.info("collector.started", running=stats.running, enabled=stats.enabled)
    return CollectorStatus.from_stats(stats)


@router.post(
    "/stop",
    response_model=CollectorStatus,
    summary="Stop the buffered collector",
    dependencies=[WriteGuard],
)
async def stop_buffered_collector() -> CollectorStatus:
    """Stop the flush loop, draining what is buffered first.

    The drain is what makes this safe to call during a deploy: the collector
    flushes its queue before its task is cancelled. Returns the post-drain
    counters, so a caller can see whether the drain succeeded — ``last_error``
    is non-null if it did not.
    """
    collector = await stop_collector()
    stats = collector.stats()
    logger.info(
        "collector.stopped",
        flushed=stats.flushed,
        dropped=stats.dropped,
        failed_flushes=stats.failed_flushes,
    )
    return CollectorStatus.from_stats(stats)
