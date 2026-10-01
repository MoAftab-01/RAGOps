"""Read schemas for the trace explorer."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import ORMModel


class RetrievedDocumentOut(ORMModel):
    id: uuid.UUID
    document_id: str
    title: str | None
    rank: int
    final_score: float
    bm25_score: float | None
    vector_score: float | None
    rerank_score: float | None
    token_count: int
    content_preview: str | None


class RetrievalOut(ORMModel):
    id: uuid.UUID
    query: str
    retriever: str
    top_k: int
    num_results: int
    latency_ms: float | None
    context_tokens: int
    top_score: float | None
    configuration: dict[str, Any] | None
    created_at: datetime
    documents: list[RetrievedDocumentOut] = Field(default_factory=list)


class SpanOut(ORMModel):
    id: uuid.UUID
    name: str
    kind: str
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    status: str
    error: str | None
    attributes: dict[str, Any] | None


class LLMCallOut(ORMModel):
    id: uuid.UUID
    model_name: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float | None
    time_to_first_token_ms: float | None
    estimated_cost: float
    status: str
    prompt: str | None
    completion: str | None
    temperature: float | None
    max_tokens: int | None
    created_at: datetime


class TraceSummary(ORMModel):
    """Row shape in the trace list. Deliberately excludes prompt/completion
    text, which would make listing traces expensive."""

    id: uuid.UUID
    trace_id: str
    application_id: uuid.UUID
    application_name: str | None = None
    user_external_id: str | None
    session_id: str | None
    kind: str
    status: str
    name: str | None
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    total_tokens: int
    estimated_cost: float
    context_tokens: int
    agent_name: str | None
    agent_iterations: int
    input_preview: str | None = None
    has_error: bool = False


class TraceDetail(ORMModel):
    """Full trace including nested spans, retrievals and generations."""

    id: uuid.UUID
    trace_id: str
    application_id: uuid.UUID
    application_name: str | None = None
    user_id: uuid.UUID | None
    user_external_id: str | None
    session_id: str | None
    kind: str
    status: str
    name: str | None
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    input_text: str | None
    output_text: str | None
    error: str | None
    total_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float
    context_tokens: int
    agent_name: str | None
    agent_iterations: int
    tags: list[str] | None
    metadata: dict[str, Any] | None
    created_at: datetime

    spans: list[SpanOut] = Field(default_factory=list)
    retrievals: list[RetrievalOut] = Field(default_factory=list)
    llm_calls: list[LLMCallOut] = Field(default_factory=list)
    # Derived, computed server-side from retrieved document ids and content.
    retrieval_analysis: dict[str, Any] = Field(default_factory=dict)
