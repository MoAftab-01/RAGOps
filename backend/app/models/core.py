"""Core telemetry models: applications, users, model registry, traces, spans.

These tables are the write path of RAGOps. They are deliberately normalised
rather than one wide "events" table: the analytics layer issues grouped
aggregations (``GROUP BY model``, ``GROUP BY user_id``) and per-column
indexes on those dimensions are what keep a 30-day window fast.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TraceStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"
    CANCELLED = "cancelled"


class TraceKind(StrEnum):
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"
    EVALUATION = "evaluation"


class SpanKind(StrEnum):
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    LLM = "llm"
    TOOL = "tool"
    AGENT_STEP = "agent_step"
    OTHER = "other"


class Application(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An instrumented AI application (e.g. ``customer-support-bot``).

    ``name`` is unique **platform-wide**, not per organization. Two companies
    therefore cannot both own an application called ``customer-support-bot``.
    That is a real product limitation, and it is a deliberate one: it means a
    company key that posts an application name belonging to someone else gets a
    loud, unambiguous error instead of silently landing its telemetry in another
    company's data. Relaxing it to ``(organization_id, name)`` is a much larger
    change — every name-to-id resolution path would need the organization
    threaded through it — and it trades a documented limitation for a silent
    data-placement hazard.

    ``organization_id`` is the single tenancy seam. Every other table reaches an
    organization through this column. It is nullable because the local
    platform key and the demo dataset have no organization; such rows are
    invisible to any tenant-scoped read, which makes NULL a safe state rather
    than a hole.
    """

    __tablename__ = "applications"

    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    environment: Mapped[str] = mapped_column(String(32), default="development", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata")

    # `SET NULL`, a deliberate deviation from the CASCADE used by every other
    # foreign key in this schema: deleting an organization must not delete a
    # company's telemetry. Orphaning the applications is recoverable; deleting
    # the traces is not.
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), index=True
    )

    traces: Mapped[list[Trace]] = relationship(back_populates="application")


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An end user of an instrumented application.

    ``external_id`` is what the customer's app knows; the surrogate UUID keeps
    the analytics joins narrow regardless of how gnarly their user ids are.
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("application_id", "external_id", name="uq_users_app_external"),
        Index("ix_users_app_external", "application_id", "external_id"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=False
    )
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata")


class Model(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Registry entry for an LLM plus its *simulated* price list.

    Local inference (Ollama) costs nothing, so ``input_cost_per_1k`` /
    ``output_cost_per_1k`` are a what-if price list used to compare configs
    against metered providers. ``is_local`` marks entries where the number is
    synthetic so the UI can label it.
    """

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("provider", "name", name="uq_models_provider_name"),)

    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    context_window: Mapped[int | None] = mapped_column(Integer)
    is_local: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    input_cost_per_1k: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    output_cost_per_1k: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)


class Trace(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One end-to-end request through an instrumented application."""

    __tablename__ = "traces"
    __table_args__ = (
        UniqueConstraint("application_id", "trace_id", name="uq_traces_app_trace"),
        # The dashboard's primary access pattern is "this app, this window".
        Index("ix_traces_app_start", "application_id", "start_time"),
        Index("ix_traces_start_status", "start_time", "status"),
        Index("ix_traces_user_start", "user_id", "start_time"),
    )

    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    user_external_id: Mapped[str | None] = mapped_column(String(255), index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True)

    kind: Mapped[str] = mapped_column(String(32), default=TraceKind.RAG, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=TraceStatus.SUCCESS, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))

    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)

    input_text: Mapped[str | None] = mapped_column(Text)
    output_text: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    context_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Agent observability
    agent_name: Mapped[str | None] = mapped_column(String(128), index=True)
    agent_iterations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    tags: Mapped[list[str] | None] = mapped_column(JSONB)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata")

    application: Mapped[Application] = relationship(back_populates="traces")
    spans: Mapped[list[Span]] = relationship(
        back_populates="trace", cascade="all, delete-orphan", order_by="Span.start_time"
    )
    llm_calls: Mapped[list[LLMCall]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )
    retrieval_calls: Mapped[list[RetrievalCall]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )


class Span(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A timed stage inside a trace (embedding, retrieval, rerank, LLM, tool)."""

    __tablename__ = "spans"
    __table_args__ = (Index("ix_spans_trace_start", "trace_id", "start_time"),)

    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"), nullable=False
    )
    parent_span_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("spans.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default=SpanKind.OTHER, nullable=False)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default=TraceStatus.SUCCESS, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    trace: Mapped[Trace] = relationship(back_populates="spans")


class LLMCall(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single generation request. The unit of token and cost accounting."""

    __tablename__ = "llm_calls"
    __table_args__ = (
        Index("ix_llm_calls_trace", "trace_id"),
        Index("ix_llm_calls_app_time", "application_id", "created_at"),
        Index("ix_llm_calls_model_time", "model_id", "created_at"),
    )

    trace_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE")
    )
    span_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("spans.id", ondelete="SET NULL"))
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("models.id", ondelete="SET NULL")
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    latency_ms: Mapped[float | None] = mapped_column(Float)
    time_to_first_token_ms: Mapped[float | None] = mapped_column(Float)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    status: Mapped[str] = mapped_column(String(32), default=TraceStatus.SUCCESS, nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text)
    completion: Mapped[str | None] = mapped_column(Text)
    temperature: Mapped[float | None] = mapped_column(Float)
    max_tokens: Mapped[int | None] = mapped_column(Integer)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata")

    trace: Mapped[Trace] = relationship(back_populates="llm_calls")


class RetrievalCall(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A retrieval stage: what was asked, with which strategy, and what came back."""

    __tablename__ = "retrieval_calls"
    __table_args__ = (
        Index("ix_retrieval_trace", "trace_id"),
        Index("ix_retrieval_app_time", "application_id", "created_at"),
    )

    trace_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE")
    )
    span_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("spans.id", ondelete="SET NULL"))
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )

    query: Mapped[str] = mapped_column(Text, nullable=False)
    retriever: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    top_k: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    num_results: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    context_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Top score returned, used as the "retrieval score" feature for anomalies.
    top_score: Mapped[float | None] = mapped_column(Float)
    configuration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    trace: Mapped[Trace] = relationship(back_populates="retrieval_calls")
    documents: Mapped[list[RetrievedDocument]] = relationship(
        back_populates="retrieval_call", cascade="all, delete-orphan"
    )


class RetrievedDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One ranked hit, with the score contributed by each retrieval stage.

    Keeping the per-stage scores lets the trace view explain *why* a document
    ranked where it did, and lets the analyzer measure duplicate context.
    """

    __tablename__ = "retrieved_documents"
    __table_args__ = (
        Index("ix_retrieved_docs_call_rank", "retrieval_call_id", "rank"),
    )

    retrieval_call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("retrieval_calls.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    bm25_score: Mapped[float | None] = mapped_column(Float)
    vector_score: Mapped[float | None] = mapped_column(Float)
    rerank_score: Mapped[float | None] = mapped_column(Float)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content_preview: Mapped[str | None] = mapped_column(Text)

    retrieval_call: Mapped[RetrievalCall] = relationship(back_populates="documents")
