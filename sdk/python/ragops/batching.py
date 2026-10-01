"""Client-side buffering for trace uploads.

Instrumenting a request must not add a synchronous HTTP round trip to the host
application's hot path, so the SDK accumulates finished traces in memory and
POSTs them to ``/api/traces/batch`` in the background.

Two rules govern everything here:

1. **Never raise into the caller.** A telemetry library that can take down the
   application it observes is worse than no telemetry. Network failures,
   timeouts and server errors are logged and the batch is dropped.
2. **Never block interpreter exit.** The worker is a daemon thread, so an
   in-flight flush at shutdown is abandoned rather than delaying ``atexit``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from ._logging import get_logger

logger = get_logger(__name__)

# Sentinel pushed to wake the worker on a demand flush or on close.
_STOP = object()


class BufferedBatcher:
    """Accumulate payloads in a queue and flush them on a background thread.

    The queue is the only shared mutable state, and ``queue.Queue`` is
    thread-safe, so ``add`` is safe to call from any number of request threads.

    Args:
        send: Callable that performs the actual POST for one batch. It receives
            the list of buffered payloads. Exceptions from it are caught and
            logged, never propagated.
        endpoint: URL posted to, used only for log context.
        flush_interval: Seconds between timer-driven flushes.
        batch_size: Number of buffered items that triggers an immediate flush.
        max_queue_size: Hard cap. Beyond ``10 * batch_size`` the oldest pending
            items are dropped with a warning, because an unbounded buffer in a
            long-lived app is an out-of-memory bug, not a feature.
    """

    def __init__(
        self,
        send: Callable[[list[dict[str, Any]]], None],
        *,
        endpoint: str,
        flush_interval: float = 5.0,
        batch_size: int = 50,
        max_queue_size: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be > 0")

        self._send = send
        self._endpoint = endpoint
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._max_queue_size = max_queue_size or (10 * batch_size)

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._max_queue_size)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._dropped = 0

        # Daemon: an in-flight flush must not keep the interpreter alive.
        self._thread = threading.Thread(
            target=self._run,
            name="ragops-batcher",
            daemon=True,
        )
        self._thread.start()

    # -- public API ---------------------------------------------------------

    @property
    def dropped(self) -> int:
        """How many payloads have been discarded because the buffer was full."""
        return self._dropped

    def add(self, payload: dict[str, Any]) -> None:
        """Enqueue one payload. Never raises and never blocks the caller.

        ``put_nowait`` is deliberate: a telemetry call must not stall a request
        thread waiting on a buffer to drain. If the queue is full the item is
        dropped and a warning is emitted (at most once per flush cycle, to
        avoid turning a full buffer into a log flood).
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped += 1
            logger.warning(
                "sdk.batch.drop",
                endpoint=self._endpoint,
                max_queue_size=self._max_queue_size,
                dropped_total=self._dropped,
            )

    def flush(self) -> None:
        """Flush buffered payloads on the calling thread, synchronously.

        Used at trace completion and on shutdown, where the caller has asked for
        the data to actually land. The background thread is left alone: it owns
        the worker loop and the buffer lock arbitrates between them.
        """
        self._drain_into_buffer()
        self._flush_buffer()

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop the worker, flush what is left, and release resources.

        Safe to call more than once. The worker is asked to exit and joined with
        a bounded timeout; if it is stuck in a network call the join gives up
        rather than hanging the host application, and the daemon flag means the
        thread still cannot block interpreter exit.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            # The worker is already backed up; the stop signal is best-effort.
            pass
        self._thread.join(timeout=timeout)
        # Whatever the worker did not get to is flushed here.
        self.flush()

    # -- worker -------------------------------------------------------------

    def _run(self) -> None:
        """Drain the queue, flushing on a timer, on a full batch, or on stop."""
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval)
            except queue.Empty:
                # Timer tick: ship whatever is buffered even if the batch is not
                # full, so a low-traffic app still delivers promptly.
                self._flush_buffer()
                continue

            if item is _STOP:
                # Anything still queued gets flushed before the thread exits.
                self._drain_into_buffer()
                self._flush_buffer()
                logger.debug("sdk.batch.closed", endpoint=self._endpoint)
                return

            with self._lock:
                self._buffer.append(item)
                should_flush = len(self._buffer) >= self._batch_size
            if should_flush:
                self._flush_buffer()

    def _drain_into_buffer(self) -> None:
        """Move everything currently queued into the pending buffer."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                # Consume but do not buffer the sentinel.
                continue
            with self._lock:
                self._buffer.append(item)

    def _flush_buffer(self) -> None:
        """POST and clear the pending buffer, swallowing every failure."""
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []

        try:
            self._send(batch)
        except Exception as exc:  # noqa: BLE001 - telemetry must never propagate
            logger.warning(
                "sdk.batch.flush_failed",
                endpoint=self._endpoint,
                batch_size=len(batch),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return
        logger.debug("sdk.batch.flushed", endpoint=self._endpoint, count=len(batch))

    def __enter__(self) -> BufferedBatcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["BufferedBatcher"]
