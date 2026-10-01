"""Telemetry write-path schemas (what the SDK posts).

These are the contract with ``sdk/python/ragops``. They accept either exact
token counts reported by the caller or enough text for the server to compute
them deterministically — the server never asks a model to count tokens.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator



# ---------------------------------------------------------------------------
# Spans and retrieval
# ---------------------------------------------------------------------------


class SpanIn(BaseModel):
    """A timed stage inside a trace."""

    name: str = Field(max_length=128)
    kind: Literal[
        "retrieval", "embedding", "rerank", "llm", "tool", "agent_step", "other"
    ] = "other"
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    status: Literal["success", "error", "running", "cancelled"] = "success"
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocumentIn(BaseModel):
    """One ranked retrieval hit.

    Per-stage scores are optional but recommended: they are what lets the trace
    view explain a ranking, and what the duplicate-context detector keys on.
    """

    document_id: str = Field(max_length=255)
    title: str | None = Field(default=None, max_length=512)
    rank: int = Field(ge=1)
    final_score: float = 0.0
    bm25_score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    content: str | None = Field(
        default=None, description="Full text; truncated server-side for storage"
    )
    content_preview: str | None = None

    @model_validator(mode="after")
    def _at_least_one_score(self) -> RetrievedDocumentIn:
        if (
            self.final_score == 0.0
            and self.bm25_score is None
            and self.vector_score is None
            and self.rerank_score is None
        ):
            raise ValueError(
                "Provide at least one of final_score, bm25_score, vector_score "
                "or rerank_score for a retrieved document."
            )
        return self


class RetrievalIn(BaseModel):
    """A retrieval stage attached to a trace."""

    query: str = Field(min_length=1, max_length=8000)
    retriever: str = Field(default="hybrid", max_length=64)
    top_k: int = Field(default=5, ge=1, le=100)
    documents: list[RetrievedDocumentIn] = Field(default_factory=list)
    duration_ms: float | None = Field(default=None, ge=0)
    timestamp: datetime | None = Field(
        default=None,
        description=(
            "When the retrieval ran. Defaults to the server's insert time, which "
            "is correct for live traffic but collapses a backfilled or replayed "
            "batch into one instant. State it when replaying history."
        ),
    )
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("documents")
    @classmethod
    def _cap_documents(cls, docs: list[RetrievedDocumentIn]) -> list[RetrievedDocumentIn]:
        # One trace storing thousands of hits is a client bug, not telemetry.
        return docs[:100]


class GenerationIn(BaseModel):
    """An LLM generation attached to a trace.

    Token counts may be supplied by the caller (Ollama returns exact counts) or
    omitted, in which case the server derives them from ``prompt``/``completion``
    with the deterministic counter. The server never invents them.
    """

    model: str = Field(max_length=128)
    provider: str = Field(default="ollama", max_length=64)
    prompt: str | None = None
    completion: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_ms: float | None = Field(default=None, ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    status: Literal["success", "error", "running", "cancelled"] = "success"
    span_id: uuid.UUID | None = None
    timestamp: datetime | None = Field(
        default=None,
        description=(
            "When the generation ran. Defaults to the server's insert time; "
            "state it when replaying history so token and cost analytics land in "
            "the right window."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceCreate(BaseModel):
    """Create (or upsert) a trace.

    ``trace_id`` is caller-supplied so an application can correlate its own
    request id with the RAGOps trace without a second round trip.
    """

    trace_id: str | None = Field(default=None, max_length=64)
    application: str = Field(max_length=128, description="Application name")
    user_id: str | None = Field(default=None, max_length=255)
    session_id: str | None = Field(default=None, max_length=128)
    kind: Literal["chat", "rag", "agent", "evaluation"] = "rag"
    name: str | None = Field(default=None, max_length=255)
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    input_text: str | None = Field(default=None, max_length=100_000)
    output_text: str | None = Field(default=None, max_length=100_000)
    error: str | None = None
    status: Literal["success", "error", "running", "cancelled"] = "running"
    agent_name: str | None = Field(default=None, max_length=128)
    agent_iterations: int = Field(default=0, ge=0)
    context_tokens: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[SpanIn] = Field(default_factory=list)
    retrievals: list[RetrievalIn] = Field(default_factory=list)
    generations: list[GenerationIn] = Field(default_factory=list)


class BatchIngest(BaseModel):
    """Buffered batch flush from the SDK.

    The SDK buffers client-side so instrumenting a request does not add a
    synchronous HTTP round trip to the application's hot path.
    """

    traces: list[TraceCreate] = Field(default_factory=list, max_length=500)

    @field_validator("traces")
    @classmethod
    def _require_content(cls, traces: list[TraceCreate]) -> list[TraceCreate]:
        if not traces:
            raise ValueError("Batch must contain at least one trace.")
        return traces


class IngestResponse(BaseModel):
    accepted: int
    trace_ids: list[str]
    errors: list[str] = Field(default_factory=list)
