"""The ``Trace`` context manager — the ergonomic core of the SDK.

A trace accumulates spans, retrievals and generations while the host application
does its work, then ships the whole thing to the server in one buffered payload.
The object is deliberately forgiving: documents may be dicts or arbitrary
objects, token counts may be omitted (they are derived deterministically), and
nothing raised by telemetry ever reaches the caller.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from ._logging import get_logger
from .tokens import estimate_tokens

logger = get_logger(__name__)

# Maps the SDK's friendlier ``kind`` values onto the server's span-kind enum.
_SPAN_KINDS = frozenset(
    {"retrieval", "embedding", "rerank", "llm", "tool", "agent_step", "other"}
)

# Cap on documents forwarded from a single retrieval. The server truncates at
# 100 too; matching it here keeps the payload small and the intent explicit.
_MAX_DOCUMENTS = 100


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_document(doc: Any, rank: int) -> dict[str, Any] | None:
    """Normalise one retrieval hit into the server's document shape.

    Accepts a mapping (``{"document_id": ..., "score": ...}``) or any object
    exposing ``document_id``/``title``/``content``/``score`` attributes, so a
    host's retriever can hand back its own result objects untouched. Returns
    ``None`` for a hit with no usable identity or score, which the server would
    reject.
    """
    if isinstance(doc, Mapping):
        get = doc.get
        document_id = get("document_id") or get("id") or get("chunk_id")
        title = get("title")
        content = get("content") or get("text") or get("page_content")
        score = get("score")
        if score is None:
            score = get("final_score")
        if score is None:
            score = get("similarity")
    else:
        document_id = getattr(doc, "document_id", None) or getattr(doc, "id", None)
        title = getattr(doc, "title", None)
        content = (
            getattr(doc, "content", None)
            or getattr(doc, "text", None)
            or getattr(doc, "page_content", None)
        )
        score = getattr(doc, "score", None)

    if not document_id:
        logger.warning("sdk.retrieval.document_missing_id", rank=rank)
        return None
    if score is None:
        # Without any score the server's RetrievedDocumentIn validator rejects
        # the document; default to 0.0 rather than silently dropping the hit.
        score = 0.0

    document: dict[str, Any] = {
        "document_id": str(document_id),
        "rank": rank,
        "final_score": float(score),
    }
    if title is not None:
        document["title"] = str(title)[:512]
    if content is not None:
        text = str(content)
        document["content"] = text
        document["content_preview"] = text[:280]
    return document


def _coerce_documents(documents: Iterable[Any]) -> list[dict[str, Any]]:
    coerced: list[dict[str, Any]] = []
    for index, doc in enumerate(documents or (), start=1):
        normalised = _coerce_document(doc, index)
        if normalised is not None:
            coerced.append(normalised)
        if len(coerced) >= _MAX_DOCUMENTS:
            break
    return coerced


class Trace:
    """One end-to-end RAG request, accumulated and then shipped as a unit.

    Use it as a context manager. On a clean exit the trace is completed with
    status ``success`` and handed to the batcher; on an exception the error is
    recorded, the status becomes ``error``, and the exception is re-raised
    untouched so the host's own error handling still runs.
    """

    def __init__(
        self,
        *,
        application: str,
        on_complete: Callable[[dict[str, Any]], None],
        user_id: str | None = None,
        session_id: str | None = None,
        kind: str = "rag",
        name: str | None = None,
        trace_id: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        start_time: datetime | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.application = application
        self.user_id = user_id
        self.session_id = session_id
        self.kind = kind
        self.name = name
        self.agent_name = agent_name
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.tags: list[str] = []

        self.start_time = start_time or _utcnow()
        self.end_time: datetime | None = None
        self.duration_ms: float | None = None
        self.input_text: str | None = None
        self.output_text: str | None = None
        self.error: str | None = None
        self.status: str = "running"
        self.agent_iterations: int = 0

        self._spans: list[dict[str, Any]] = []
        self._retrievals: list[dict[str, Any]] = []
        self._generations: list[dict[str, Any]] = []
        self._on_complete = on_complete
        self._closed = False
        self._monotonic_start = time.perf_counter()

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> bool:
        if exc is not None:
            self.set_error(f"{type(exc).__name__}: {exc}")
            self.status = "error"
        else:
            self.status = "success"
        self.complete()
        # Never swallow the host application's exception.
        return False

    # -- recording ----------------------------------------------------------

    def log_span(
        self,
        name: str,
        kind: str = "other",
        duration_ms: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a timed stage (embedding, rerank, tool call, ...).

        Unknown ``kind`` values are coerced to ``"other"`` rather than raising,
        so a typo in a stage name cannot break the traced request.
        """
        span_kind = kind if kind in _SPAN_KINDS else "other"
        span: dict[str, Any] = {
            "name": name,
            "kind": span_kind,
            "status": "success",
            "attributes": dict(attributes or {}),
        }
        if duration_ms is not None:
            span["duration_ms"] = max(0.0, float(duration_ms))
        self._spans.append(span)
        return span

    def log_retrieval(
        self,
        query: str,
        documents: Iterable[Any] | None = None,
        retriever: str = "hybrid",
        top_k: int = 5,
        duration_ms: float | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a retrieval stage and the documents it returned.

        ``documents`` may be dicts or objects exposing ``document_id``,
        ``title``, ``content`` and ``score``. Ranks are assigned from the
        iteration order: the first document is rank 1, which is what every
        downstream metric (MRR, NDCG, precision@k) assumes.
        """
        if not query or not str(query).strip():
            logger.warning("sdk.retrieval.empty_query", trace_id=self.trace_id)
            return {}

        retrieval: dict[str, Any] = {
            "query": str(query),
            "retriever": retriever,
            "top_k": int(top_k),
            "documents": _coerce_documents(documents or ()),
            "configuration": dict(configuration or {}),
        }
        if duration_ms is not None:
            retrieval["duration_ms"] = max(0.0, float(duration_ms))
        self._retrievals.append(retrieval)
        return retrieval

    def log_generation(
        self,
        model: str,
        answer: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        prompt: str | None = None,
        duration_ms: float | None = None,
        provider: str = "ollama",
        temperature: float | None = None,
        max_tokens: int | None = None,
        time_to_first_token_ms: float | None = None,
        status: str = "success",
        metadata: Mapping[str, Any] | None = None,
        span_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an LLM generation.

        When ``input_tokens``/``output_tokens`` are omitted they are derived with
        the same deterministic counter the server uses, applied to ``prompt`` and
        ``answer``. This is a regex, not a model call: token counting is exact
        math and must never be delegated to an LLM. Callers that have real
        counts (Ollama reports them) should pass them and they will be used
        verbatim.
        """
        resolved_input = (
            int(input_tokens) if input_tokens is not None else estimate_tokens(prompt)
        )
        resolved_output = (
            int(output_tokens) if output_tokens is not None else estimate_tokens(answer)
        )

        generation: dict[str, Any] = {
            "model": model,
            "provider": provider,
            "input_tokens": max(0, resolved_input),
            "output_tokens": max(0, resolved_output),
            "status": status,
            "metadata": dict(metadata or {}),
        }
        if prompt is not None:
            generation["prompt"] = prompt
        if answer is not None:
            generation["completion"] = answer
        for key, value in (
            ("duration_ms", duration_ms),
            ("time_to_first_token_ms", time_to_first_token_ms),
            ("temperature", temperature),
            ("max_tokens", max_tokens),
        ):
            if value is not None:
                generation[key] = float(value) if key.endswith("_ms") else value
        if span_id is not None:
            generation["span_id"] = span_id

        self._generations.append(generation)
        if answer and not self.output_text:
            self.output_text = answer
        return generation

    def log_agent_step(
        self,
        name: str,
        action: str | None = None,
        observation: str | None = None,
        duration_ms: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one step of an agent loop and bump the iteration counter."""
        self.agent_iterations += 1
        span_attributes: dict[str, Any] = dict(attributes or {})
        if action is not None:
            span_attributes.setdefault("action", action)
        if observation is not None:
            span_attributes.setdefault("observation", observation)
        return self.log_span(
            name, kind="agent_step", duration_ms=duration_ms, attributes=span_attributes
        )

    def set_output(self, text: str | None) -> None:
        """Set the trace's final output text (what the user ultimately saw)."""
        if text is not None:
            self.output_text = text

    def set_error(self, msg: str) -> None:
        """Record an error message and mark the trace failed."""
        self.error = msg
        self.status = "error"

    def set_tags(self, tags: Iterable[str]) -> None:
        """Attach free-form tags used to slice traces in the dashboard."""
        self.tags = [str(tag) for tag in (tags or ())]

    def set_input(self, text: str | None) -> None:
        """Set the trace's input text (the original user query)."""
        if text is not None:
            self.input_text = text

    def set_metadata(self, key: str, value: Any) -> None:
        """Merge one key into the trace's free-form metadata dict."""
        self.metadata[key] = value

    def add_generation_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate denormalised token totals, mirroring the server's maths.

        The server recomputes a trace's totals from its ``LLMCall`` rows, so the
        client does the same here to keep an in-flight trace's numbers honest.
        """
        if input_tokens:
            self.metadata["input_tokens"] = int(
                self.metadata.get("input_tokens", 0)
            ) + int(input_tokens)
        if output_tokens:
            self.metadata["output_tokens"] = int(
                self.metadata.get("output_tokens", 0)
            ) + int(output_tokens)

    # -- completion ---------------------------------------------------------

    def complete(self) -> None:
        """Finalise the trace and hand its payload to the batcher exactly once."""
        if self._closed:
            return
        self._closed = True

        self.end_time = _utcnow()
        if self.duration_ms is None:
            self.duration_ms = (time.perf_counter() - self._monotonic_start) * 1000.0

        payload = self.to_payload()
        try:
            self._on_complete(payload)
        except Exception as exc:  # noqa: BLE001 - telemetry must never propagate
            logger.warning(
                "sdk.trace.dispatch_failed",
                trace_id=self.trace_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def to_payload(self) -> dict[str, Any]:
        """Render the wire payload matching the server's ``TraceCreate`` schema."""
        return {
            "trace_id": self.trace_id,
            "application": self.application,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms or 0.0, 3),
            "input_text": self.input_text,
            "output_text": self.output_text,
            "error": self.error,
            "status": self.status,
            "agent_name": self.agent_name,
            "agent_iterations": self.agent_iterations,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "spans": list(self._spans),
            "retrievals": list(self._retrievals),
            "generations": list(self._generations),
        }


__all__ = ["Trace"]
