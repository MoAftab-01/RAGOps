"""The RAGOps client — a drop-in observability layer for RAG applications.

Instrument a request by wrapping it in a trace. Retrieval hits, prompts,
generations, token counts and latency are recorded automatically and shipped to
the RAGOps server in the background.

Usage::

    from ragops import RAGOps

    client = RAGOps(endpoint="http://localhost:8000", application="my-app")

    with client.trace(application="customer-support", user_id="demo-user") as trace:
        documents = retriever.search(query)
        trace.log_retrieval(query=query, documents=documents)
        answer = llm.generate(...)
        trace.log_generation(answer=answer, input_tokens=..., output_tokens=...)

    client.close()

Design rules, in priority order:

1. **Never take down the application being observed.** Every network call is
   wrapped; failures are logged and swallowed. The SDK is safe to import and
   use with no RAGOps server running at all — telemetry degrades to a no-op.
2. **Never block the hot path.** Completed traces are handed to a background
   batcher, so instrumenting a request costs a dict append, not a round trip.
3. **Never invent a number.** Token counts the caller omits are derived with
   the same deterministic regex counter the server uses — never an LLM.
"""

from __future__ import annotations

import atexit
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import httpx

from ._logging import get_logger
from .batching import BufferedBatcher
from .tokens import estimate_tokens
from .tracer import Trace

logger = get_logger(__name__)

DEFAULT_ENDPOINT = "http://localhost:8000"
BATCH_PATH = "/api/traces/batch"
HEALTH_PATH = "/api/health"


class RAGOps:
    """Client for the RAGOps telemetry API.

    Args:
        endpoint: Base URL of the RAGOps server.
        api_key: Optional API key, sent as ``X-API-Key``. Required only when the
            server runs with authentication enabled.
        application: Default application name, used by any method that does not
            override it.
        flush_interval: Seconds between background flushes.
        batch_size: Traces buffered before an immediate flush.
        timeout: Per-request HTTP timeout in seconds.
        enabled: Set ``False`` to construct a fully inert client (useful in
            tests and in local development where telemetry is unwanted).
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str | None = None,
        application: str | None = None,
        flush_interval: float = 5.0,
        batch_size: int = 50,
        timeout: float = 10.0,
        *,
        enabled: bool = True,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.application = application
        self.timeout = timeout
        self.enabled = enabled
        self._closed = False
        self._lock = threading.Lock()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.Client(
            base_url=self.endpoint,
            headers=headers,
            timeout=timeout,
        )

        self._batcher = BufferedBatcher(
            self._post_batch,
            endpoint=f"{self.endpoint}{BATCH_PATH}",
            flush_interval=flush_interval,
            batch_size=batch_size,
        )
        # Best-effort flush at interpreter exit so a script that never calls
        # close() still delivers what it recorded.
        atexit.register(self.shutdown)
        logger.debug("sdk.client.initialized", endpoint=self.endpoint, enabled=enabled)

    # -- health -------------------------------------------------------------

    def health(self) -> dict[str, Any] | None:
        """Return the server's health payload, or ``None`` if unreachable.

        This is the one method whose failure the caller usually wants to see:
        it is a diagnostic, not instrumentation, so it returns ``None`` instead
        of raising and leaves the decision to the caller.
        """
        if not self.enabled:
            return None
        try:
            response = self._client.get(HEALTH_PATH)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - diagnostic, never raises
            logger.warning(
                "sdk.health.failed",
                endpoint=self.endpoint,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
        try:
            return dict(response.json())
        except Exception as exc:  # noqa: BLE001
            logger.warning("sdk.health.bad_payload", error=str(exc))
            return None

    # -- tracing ------------------------------------------------------------

    def trace(
        self,
        application: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        kind: str = "rag",
        name: str | None = None,
        trace_id: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trace:
        """Start a new trace. Use it as a context manager.

        On a clean ``with`` exit the trace is completed with status ``success``
        and flushed; if the block raises, the error is recorded, the status
        becomes ``error``, and the exception propagates unchanged.
        """
        resolved_application = application or self.application
        if not resolved_application:
            raise ValueError(
                "An application name is required: pass application= to the client "
                "constructor or to trace()."
            )
        return Trace(
            application=resolved_application,
            on_complete=self._enqueue,
            user_id=user_id,
            session_id=session_id,
            kind=kind,
            name=name,
            trace_id=trace_id,
            agent_name=agent_name,
            metadata=metadata,
        )

    def create_trace(
        self,
        application: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        kind: str = "rag",
        name: str | None = None,
        trace_id: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Iterable[str] | None = None,
    ) -> Trace:
        """Create a trace without a ``with`` block, for manual control.

        Call ``trace.complete()`` when finished. Prefer :meth:`trace` unless you
        genuinely need to start and finish at different points.
        """
        created = self.trace(
            application=application,
            user_id=user_id,
            session_id=session_id,
            kind=kind,
            name=name,
            trace_id=trace_id,
            agent_name=agent_name,
            metadata=metadata,
        )
        if tags:
            created.set_tags(tags)
        return created

    # -- one-shot recording -------------------------------------------------

    def log_retrieval(
        self,
        query: str,
        documents: Iterable[Any] | None = None,
        retriever: str = "hybrid",
        top_k: int = 5,
        duration_ms: float | None = None,
        configuration: Mapping[str, Any] | None = None,
        application: str | None = None,
        user_id: str | None = None,
    ) -> Trace:
        """Record a standalone retrieval and complete that trace immediately.

        A convenience for scripts and batch jobs that retrieve without a full
        RAG round trip; inside a request, use ``trace.log_retrieval(...)``.
        """
        trace = self.create_trace(application=application, user_id=user_id)
        trace.log_retrieval(
            query=query,
            documents=documents,
            retriever=retriever,
            top_k=top_k,
            duration_ms=duration_ms,
            configuration=configuration,
        )
        trace.complete()
        return trace

    def log_generation(
        self,
        model: str,
        answer: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        prompt: str | None = None,
        duration_ms: float | None = None,
        provider: str = "ollama",
        application: str | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> Trace:
        """Record a standalone generation and complete that trace immediately.

        Omitted token counts are derived with the shared deterministic counter,
        exactly as they are inside a ``with`` block.
        """
        trace = self.create_trace(application=application, user_id=user_id)
        trace.log_generation(
            model=model,
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt,
            duration_ms=duration_ms,
            provider=provider,
            **kwargs,
        )
        trace.complete()
        return trace

    def complete_trace(
        self,
        trace: Trace,
        status: str | None = None,
        output: str | None = None,
        error: str | None = None,
    ) -> None:
        """Finalise an already-started trace and queue it for upload.

        Idempotent: completing the same trace twice ships it once.
        """
        if output is not None:
            trace.set_output(output)
        if error is not None:
            trace.set_error(error)
        if status is not None:
            trace.status = status
        trace.complete()

    # -- lifecycle ----------------------------------------------------------

    def flush(self) -> None:
        """Push everything buffered to the server now, synchronously.

        Call before exiting a short-lived script, or whenever a test needs the
        data to have landed. A failure is logged, not raised.
        """
        if self._closed:
            return
        self._batcher.flush()

    def close(self, *, timeout: float = 5.0) -> None:
        """Flush, stop the background thread and release the HTTP connection.

        Idempotent and safe to call from a ``finally`` block.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._batcher.close(timeout=timeout)
        finally:
            self._client.close()
        logger.debug("sdk.client.closed", endpoint=self.endpoint)

    def shutdown(self) -> None:
        """Synchronous best-effort shutdown hook, registered with ``atexit``.

        Wrapped in a bare ``except``: an exception here would print during
        interpreter teardown, which is the worst possible time to raise.
        """
        try:
            self.close(timeout=2.0)
        except Exception:  # noqa: BLE001, S110 - teardown must be silent
            pass

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> RAGOps:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _enqueue(self, payload: dict[str, Any]) -> None:
        """Hand a completed trace to the batcher. Never raises."""
        if not self.enabled or self._closed:
            return
        self._batcher.add(payload)

    def _post_batch(self, batch: list[dict[str, Any]]) -> None:
        """POST one batch to the ingest endpoint.

        Runs on the batcher thread, which already catches anything raised here.
        A non-2xx response is treated as a failure and logged: the server
        rejects malformed traces, and silently retrying them forever would be
        worse than dropping the batch.
        """
        response = self._client.post(
            BATCH_PATH,
            json={"traces": batch},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"RAGOps ingest returned {response.status_code}: {response.text[:200]}"
            )
        logger.debug("sdk.batch.accepted", count=len(batch), status=response.status_code)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("enabled" if self.enabled else "disabled")
        return f"RAGOps(endpoint={self.endpoint!r}, application={self.application!r}, {state})"


__all__ = ["RAGOps", "DEFAULT_ENDPOINT", "BATCH_PATH", "HEALTH_PATH", "estimate_tokens"]
