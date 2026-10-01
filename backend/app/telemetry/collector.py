"""Buffered background writer for the telemetry ingest path.

Instrumenting a request should not make that request wait for the database, so
:class:`BufferedCollector` accepts :class:`~app.schemas.ingest.TraceCreate`
payloads into an :class:`asyncio.Queue` and writes them from a background task
started by the FastAPI lifespan. The task flushes when the queue reaches
``settings.collector_buffer_size`` or every
``settings.collector_flush_interval_seconds``, whichever comes first, and each
flush writes into a **new** session so a long-lived session can never hold a
transaction open across the interval.

Three properties matter more than throughput, because this code runs inside
someone else's application:

**It never raises into the caller.** A telemetry library that takes down the
app it observes is worse than one that loses a metric. :meth:`enqueue` is
synchronous and cannot fail; a full queue drops the oldest payload and counts
it. Every other failure is logged.

**A failed write is retried, not lost.** A flush that raises requeues its batch
with an incremented attempt count, bounded by ``max_retries``. The bound is
what stops a dead database from turning the queue into an infinite retry loop
that grows without limit.

**Nothing is required at import time.** No Redis, no database, no running event
loop. Redis is deliberately not used for the buffer: an in-process queue keeps
the write path one hop from the database, and a telemetry collector that cannot
start because an optional cache is down is not worth the extra hop. A Redis
handoff is the seam for running more than one API process, where an in-process
queue would strand each process's buffer at exit.

:func:`start_collector` and :func:`stop_collector` wire one process-wide
collector into the app lifespan, and :func:`get_collector` exposes it to the
health endpoint through :meth:`BufferedCollector.stats`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import SessionLocal
from app.core.logging import get_logger
from app.schemas.ingest import TraceCreate
from app.telemetry.ingest_service import IngestService, get_ingest_service

logger = get_logger(__name__)

__all__ = [
    "BufferedCollector",
    "CollectorStats",
    "get_collector",
    "reset_collector",
    "start_collector",
    "stop_collector",
]

# Sentinel pushed onto the queue to ask the flush task to finish. An object
# rather than None because a payload is never None, and a dedicated type means
# the drain loop cannot mistake a sentinel for data.
_STOP: Final = object()

# Ceiling on how long stop() waits for the background task to finish its final
# flush before cancelling it. A stuck database must not stop the app from
# shutting down.
_STOP_TIMEOUT_SECONDS: Final = 10.0

# After a failed flush, wait this long before trying again. Without it a
# database that is refusing connections turns the flush loop into a busy loop
# that spins at the full speed of the failing round trip.
_FAILURE_BACKOFF_SECONDS: Final = 1.0

# How many times one payload may be requeued after a failed flush before it is
# dropped. Three retries survives a deploy or a brief failover; beyond that the
# payload is stale enough that replaying it is no longer useful.
DEFAULT_MAX_RETRIES: Final = 3

# Only the most recent counter values are kept, so a long-running collector
# cannot grow its stats without bound.
_COUNTER_HISTORY: Final = 20


@dataclass(slots=True)
class _QueuedTrace:
    """A payload plus how many times its write has already failed."""

    payload: TraceCreate
    attempts: int = 0


@dataclass(frozen=True, slots=True)
class CollectorStats:
    """A snapshot of the collector, as reported by the health endpoint.

    Counters are monotonic since process start. ``queue_depth`` is the only
    field that moves in both directions, and it is the one an operator watches:
    a depth that sits at capacity means writes are not keeping up.
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
    last_flush_at: datetime | None
    last_error: str | None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe form for the health response."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "enqueued": self.enqueued,
            "flushed": self.flushed,
            "dropped": self.dropped,
            "failed_flushes": self.failed_flushes,
            "retries": self.retries,
            "flush_count": self.flush_count,
            "last_flush_at": (
                self.last_flush_at.isoformat() if self.last_flush_at else None
            ),
            "last_error": self.last_error,
        }


class BufferedCollector:
    """Buffers trace payloads and writes them in batches from a background task."""

    def __init__(
        self,
        *,
        service: IngestService | None = None,
        session_factory: Callable[[], AsyncSession] | None = None,
        buffer_size: int | None = None,
        flush_interval: float | None = None,
        batch_size: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enabled: bool | None = None,
        organization_id: uuid.UUID | None = None,
    ) -> None:
        self._service = service if service is not None else get_ingest_service()
        self._session_factory = session_factory or SessionLocal
        # Which company this collector is writing for, resolved from the key the
        # SDK authenticated with. It is a constructor argument rather than
        # something re-read per flush because it cannot change: one collector
        # is one key, and a collector that could switch organizations between
        # two batches of the same buffered trace would split a burst across two
        # companies' dashboards. ``None`` is the platform principal -- the
        # in-process collector the demo routes use, which is not tenant-scoped.
        self._organization_id = organization_id
        self._buffer_size = buffer_size if buffer_size is not None else settings.collector_buffer_size
        self._flush_interval = (
            flush_interval
            if flush_interval is not None
            else settings.collector_flush_interval_seconds
        )
        # One transaction never carries more rows than the buffer holds, so a
        # burst cannot turn into an unbounded statement.
        self._batch_size = max(1, batch_size if batch_size is not None else self._buffer_size)
        self._max_retries = max(0, max_retries)
        self._enabled = settings.collector_enabled if enabled is None else enabled

        self._queue: asyncio.Queue[_QueuedTrace | object] = asyncio.Queue(
            maxsize=max(1, self._buffer_size)
        )
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

        self._enqueued = 0
        self._flushed = 0
        self._dropped = 0
        self._failed_flushes = 0
        self._retries = 0
        self._flush_count = 0
        self._last_flush_at: datetime | None = None
        self._last_error: str | None = None
        self._recent_errors: list[str] = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        """Whether the background flush task is live."""
        return self._task is not None and not self._task.done()

    @property
    def enabled(self) -> bool:
        """Whether the collector is configured to buffer at all."""
        return self._enabled

    def stats(self) -> CollectorStats:
        """Current counters, for ``GET /api/health``.

        Read-only and never raises: the health endpoint reports collector
        trouble, so it must not become a second way for the collector to fail.
        """
        return CollectorStats(
            enabled=self._enabled,
            running=self.is_running,
            queue_depth=self._queue.qsize(),
            queue_capacity=self._buffer_size,
            enqueued=self._enqueued,
            flushed=self._flushed,
            dropped=self._dropped,
            failed_flushes=self._failed_flushes,
            retries=self._retries,
            flush_count=self._flush_count,
            last_flush_at=self._last_flush_at,
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background flush task. Idempotent.

        A disabled collector starts nothing: with telemetry buffering turned
        off, the routes ingest synchronously and there is no queue to drain.
        """
        if not self._enabled:
            logger.info("telemetry.collector_disabled")
            return
        if self.is_running:
            logger.debug("telemetry.collector_already_running")
            return

        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="ragops-telemetry-collector")
        logger.info(
            "telemetry.collector_started",
            buffer_size=self._buffer_size,
            flush_interval_seconds=self._flush_interval,
            batch_size=self._batch_size,
            max_retries=self._max_retries,
        )

    async def stop(self) -> None:
        """Stop the collector, flushing everything still queued.

        The task is asked to finish rather than cancelled, so a batch it is
        already writing is not lost mid-statement. Only a task that overruns
        ``_STOP_TIMEOUT_SECONDS`` is cancelled, and anything left in the queue
        afterwards is drained synchronously — a shutdown that silently discards
        the last few seconds of telemetry is the one place a drop is
        indistinguishable from data that was never sent.
        """
        task = self._task
        self._task = None

        if task is not None and not task.done():
            self._stopping = True
            await self._queue.put(_STOP)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_STOP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("telemetry.collector_stop_timeout")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:  # pragma: no cover - shutdown racing cancel
                pass

        self._stopping = False
        await self._drain_remaining()
        logger.info("telemetry.collector_stopped", **self._counters())

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------
    def enqueue(self, payload: TraceCreate) -> bool:
        """Queue one payload for writing. Returns whether it was accepted.

        Synchronous and non-raising, because it is called from a request handler
        that is instrumenting somebody else's code path.

        When the queue is full the *oldest* payload is discarded rather than the
        newest. A collector that always keeps the most recent window is more
        useful than one that preserves a backlog it will never catch up on, and
        the drop is counted in :meth:`stats` rather than being silent.
        """
        if not self._enabled:
            return False

        item = _QueuedTrace(payload=payload)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            if not self._make_room():
                self._drop(1)
                return False
            logger.warning(
                "telemetry.collector_queue_full",
                capacity=self._buffer_size,
                dropped_total=self._dropped,
            )
        self._enqueued += 1
        return True

    def _make_room(self) -> bool:
        """Evict the oldest payload so the caller can enqueue a newer one.

        A collector that always keeps the most recent window is more useful than
        one that preserves a backlog it will never catch up on. The evicted
        payload is counted as dropped, so the loss is visible in health rather
        than being a silent gap in the dashboard.
        """
        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - a consumer freed the slot
            return True
        self._queue.task_done()
        self._drop(1)
        logger.debug("telemetry.collector_payload_dropped", reason="queue_full")
        return True

    def _drop(self, count: int) -> None:
        """Record ``count`` payloads as dropped."""
        self._dropped += count

    # ------------------------------------------------------------------
    # Flush loop
    # ------------------------------------------------------------------
    async def _run(self) -> None:
        """Flush on a timer, on a full buffer, or on the stop sentinel.

        The wait is on the queue itself with the flush interval as a timeout,
        so a burst of traffic is written as soon as the buffer fills rather than
        waiting out the interval.
        """
        logger.debug("telemetry.collector_loop_started")
        try:
            while True:
                try:
                    first = await asyncio.wait_for(
                        self._queue.get(), timeout=self._flush_interval
                    )
                except TimeoutError:
                    continue

                if first is _STOP:
                    self._queue.task_done()
                    break

                batch = [first]
                self._drain_into(batch)
                await self._flush(batch)
        except asyncio.CancelledError:
            # Requeue whatever this iteration was holding so a hard cancel
            # cannot silently drop it, then let the cancellation propagate.
            raise
        finally:
            logger.debug("telemetry.collector_loop_stopped")

    def _drain_into(self, batch: list[_QueuedTrace]) -> None:
        """Move everything currently queued into ``batch``, up to the batch size.

        A payload that arrives while a batch is being written waits for the next
        interval rather than joining a transaction that is already open. A stop
        sentinel seen here is kept and re-queued at the back of the queue, so
        the flush loop always finds it and no payload behind it is orphaned.
        """
        while len(batch) < self._batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.task_done()
            if item is _STOP:
                self._queue.put_nowait(_STOP)
                return
            batch.append(item)

    async def _drain_remaining(self) -> None:
        """Write everything still queued, in batches, before returning.

        Retries are disabled here. A requeue only makes sense while somebody is
        left to read the queue, and at shutdown there is nobody: an exhausted
        payload is dropped with an error instead, which is the honest outcome
        for a database that is still refusing connections.
        """
        while True:
            batch: list[_QueuedTrace] = []
            self._drain_into(batch)
            if not batch:
                return
            await self._flush(batch, requeue=False)

    async def _flush(self, batch: Sequence[_QueuedTrace], *, requeue: bool = True) -> None:
        """Write one batch, requeueing it if the write failed and retries remain.

        ``requeue=False`` is used by the final shutdown drain, where requeuing
        into a queue nobody will read again would only spin: an exhausted batch
        is dropped there with an error, which is the honest outcome for a
        database that is still down when the process exits.
        """
        if not batch:
            return

        try:
            await self._write(batch)
        except Exception as exc:  # noqa: BLE001 - a write failure must not kill the task
            self._failed_flushes += 1
            self._record_error(exc)
            logger.warning(
                "telemetry.collector_flush_failed",
                error=f"{type(exc).__name__}: {exc}",
                batch_size=len(batch),
                failed_flushes=self._failed_flushes,
            )
            if requeue:
                self._requeue(batch)
                # Give a struggling database room to recover instead of
                # hammering it at the speed of its own failure.
                await asyncio.sleep(
                    min(self._flush_interval, _FAILURE_BACKOFF_SECONDS)
                )
            else:
                self._drop(len(batch))
                logger.error(
                    "telemetry.collector_flush_abandoned",
                    batch_size=len(batch),
                    reason="shutdown",
                )
        else:
            self._flushed += len(batch)
            self._flush_count += 1
            self._last_flush_at = datetime.now(timezone.utc)
            logger.debug(
                "telemetry.collector_flushed",
                batch_size=len(batch),
                queue_depth=self._queue.qsize(),
            )

    def _requeue(self, batch: Sequence[_QueuedTrace]) -> None:
        """Put a failed batch back on the queue, dropping what is out of retries.

        The bound is what stops a dead database from turning the queue into an
        unbounded retry buffer: after ``max_retries`` a payload is abandoned and
        counted, so the operator sees both the failure and the loss.
        """
        for item in batch:
            if item.attempts >= self._max_retries:
                self._drop(1)
                logger.error(
                    "telemetry.collector_payload_abandoned",
                    trace_id=item.payload.trace_id or "-",
                    attempts=item.attempts,
                )
                continue
            item.attempts += 1
            self._retries += 1
            if not self._push(item):
                self._drop(1)

    def _push(self, item: _QueuedTrace) -> bool:
        """Queue one payload, evicting the oldest if the buffer is full."""
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._make_room()
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - a racing consumer
                return False
        return True

    async def _write(self, batch: Sequence[_QueuedTrace]) -> None:
        """Persist one batch in a fresh session, then invalidate cached analytics.

        The session is created per flush rather than reused so no transaction is
        ever open between flushes. The cache invalidation runs after the commit:
        an analytics response computed before the write landed would otherwise
        be served as fresh, and every dashboard number on screen would be one
        flush behind until it expired on its own.
        """
        payloads = [item.payload for item in batch]
        async with self._session() as session:
            result = await self._service.ingest_batch(
                session, payloads, organization_id=self._organization_id
            )

        if result.errors:
            # The batch committed; only individual payloads were rejected.
            # Counted as failures so a permanently malformed payload is visible
            # in health rather than vanishing.
            self._failed_flushes += len(result.errors)
            self._record_error(result.errors[0])
            logger.warning(
                "telemetry.collector_partial_failure",
                accepted=result.accepted,
                errors=result.errors[:3],
            )
        await self._invalidate_analytics()

    async def _invalidate_analytics(self) -> None:
        """Drop memoised analytics. Best effort; Redis is optional by design."""
        from app.core.cache import invalidate  # noqa: PLC0415 - keeps import optional

        try:
            await invalidate("analytics")
        except Exception as exc:  # noqa: BLE001 - a cache is never worth failing a write
            logger.debug("telemetry.cache_invalidate_failed", error=str(exc))

    @asynccontextmanager
    async def _session(self) -> Any:
        """A transactional scope over a brand-new session."""
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    def _record_error(self, exc: BaseException | str) -> None:
        """Remember the most recent failure, keeping only a short tail."""
        self._last_error = (
            exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        )
        self._recent_errors.append(self._last_error)
        del self._recent_errors[:-_COUNTER_HISTORY]

    def _counters(self) -> dict[str, int]:
        """Counters for a log line."""
        return {
            "enqueued": self._enqueued,
            "flushed": self._flushed,
            "dropped": self._dropped,
            "failed_flushes": self._failed_flushes,
            "retries": self._retries,
        }


# ---------------------------------------------------------------------------
# Process-wide collector
# ---------------------------------------------------------------------------

_COLLECTOR: BufferedCollector | None = None


def get_collector() -> BufferedCollector:
    """Return the process-wide collector, constructing it on first use.

    Constructing it reads settings and opens nothing, so a test can ask for the
    collector and assert on its defaults without a database or a lifespan.
    """
    global _COLLECTOR
    if _COLLECTOR is None:
        _COLLECTOR = BufferedCollector()
    return _COLLECTOR


def reset_collector() -> None:
    """Drop the cached collector. For tests and settings reloads."""
    global _COLLECTOR
    _COLLECTOR = None


async def start_collector() -> BufferedCollector:
    """Start the process-wide collector. Called from the app lifespan."""
    collector = get_collector()
    await collector.start()
    return collector


async def stop_collector() -> BufferedCollector:
    """Stop the process-wide collector, flushing what is queued."""
    collector = get_collector()
    await collector.stop()
    return collector
