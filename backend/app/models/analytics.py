"""Analytics-facing models: documents, evaluations, anomalies, recommendations.

Separated from :mod:`app.models.core` because these tables have a different
lifecycle — telemetry is append-only and hot, evaluation artifacts are
written in bursts by batch jobs and read by the UI.
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
    Text as TextType,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnomalyType(StrEnum):
    TOKEN_SPIKE = "token_spike"
    LATENCY_SPIKE = "latency_spike"
    RETRIEVAL_SCORE_DROP = "retrieval_score_drop"
    COST_SPIKE = "cost_spike"
    RETRIEVAL_COUNT_ANOMALY = "retrieval_count_anomaly"
    AGENT_LOOP = "agent_loop"
    ERROR_RATE = "error_rate"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An ingested knowledge-base chunk.

    Evaluation datasets reference these by ``external_id`` (a content hash of
    the chunk), so a dataset stays valid across re-ingestions.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("application_id", "external_id", name="uq_documents_app_external"),
        Index("ix_documents_app_source", "application_id", "source"),
    )

    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source: Mapped[str | None] = mapped_column(String(512), index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(TextType, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_external_id: Mapped[str | None] = mapped_column(String(255))
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata")


class EvaluationRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One evaluation execution against one dataset and one configuration.

    A run is the atom that regression detection compares: two runs of the same
    ``dataset_name`` are comparable, and their ``config`` diffs are what lets
    RAGOps explain *why* a metric moved instead of guessing.
    """

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_eval_runs_app_created", "application_id", "created_at"),
        Index("ix_eval_runs_dataset", "dataset_name", "created_at"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    evaluation_type: Mapped[str] = mapped_column(String(32), nullable=False)  # retrieval | answer
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.PENDING, nullable=False)

    k: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    num_queries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Aggregated metrics: recall@5, mrr, ndcg@5, hit_rate@5, precision@5, ...
    metrics: Mapped[dict[str, Any] | None] = mapped_column("metrics")
    # The exact retrieval/model configuration used, for regression diffing.
    config: Mapped[dict[str, Any] | None] = mapped_column("config")

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per-query evaluation output. Aggregation happens in SQL, not in Python."""

    __tablename__ = "evaluation_results"
    __table_args__ = (Index("ix_eval_results_run", "run_id"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    k: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    # Per-query metrics, e.g. {"recall": 0.8, "precision": 0.6, "mrr": 1.0, ...}
    metrics: Mapped[dict[str, Any] | None] = mapped_column("metrics")
    retrieved_document_ids: Mapped[list[str] | None] = mapped_column("retrieved_document_ids")
    relevant_document_ids: Mapped[list[str] | None] = mapped_column("relevant_document_ids")
    duration_ms: Mapped[float | None] = mapped_column(Float)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")


class AnswerEvaluation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Answer-quality evaluation for a single question/answer/context triple.

    Deterministic and judge-based numbers live in separate columns so the UI
    can always show which method produced which figure.
    """

    __tablename__ = "answer_evaluations"
    __table_args__ = (Index("ix_answer_eval_run", "run_id"),)

    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE")
    )
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[list[str] | None] = mapped_column("context")
    reference_answer: Mapped[str | None] = mapped_column(Text)

    # --- deterministic / embedding-based ---
    faithfulness: Mapped[float | None] = mapped_column(Float)
    context_relevance: Mapped[float | None] = mapped_column(Float)
    answer_relevance: Mapped[float | None] = mapped_column(Float)
    citation_coverage: Mapped[float | None] = mapped_column(Float)
    unsupported_claim_ratio: Mapped[float | None] = mapped_column(Float)
    # --- LLM-as-judge (nullable: judge is opt-in) ---
    judge_faithfulness: Mapped[float | None] = mapped_column(Float)
    judge_answer_relevance: Mapped[float | None] = mapped_column(Float)
    judge_model: Mapped[str | None] = mapped_column(String(128))
    judge_rationale: Mapped[str | None] = mapped_column(Text)

    claims: Mapped[list[dict[str, Any]] | None] = mapped_column("claims")
    metrics: Mapped[dict[str, Any] | None] = mapped_column("metrics")
    duration_ms: Mapped[float | None] = mapped_column(Float)


class Anomaly(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A detected outlier, with the evidence that produced it.

    ``evidence`` holds measured facts (expected band, observed value, peer
    window statistics). RAGOps never writes a *cause* into it — causes shown in
    the UI are limited to signals actually present in the trace.
    """

    __tablename__ = "anomalies"
    __table_args__ = (
        Index("ix_anomalies_app_detected", "application_id", "detected_at"),
        Index("ix_anomalies_type_severity", "anomaly_type", "severity"),
    )

    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), default=Severity.MEDIUM, nullable=False)

    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_value: Mapped[float | None] = mapped_column(Float)
    expected_low: Mapped[float | None] = mapped_column(Float)
    expected_high: Mapped[float | None] = mapped_column(Float)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    peer_median: Mapped[float | None] = mapped_column(Float)
    peer_p95: Mapped[float | None] = mapped_column(Float)

    evidence: Mapped[dict[str, Any] | None] = mapped_column("evidence")
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    detection_method: Mapped[str] = mapped_column(String(64), default="isolation_forest")
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class OptimizationRecommendation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An evidence-backed optimisation suggestion.

    Every recommendation must populate ``evidence`` with the metric that
    triggered it; the engine refuses to emit a recommendation it cannot
    justify from recorded data.
    """

    __tablename__ = "optimization_recommendations"
    __table_args__ = (Index("ix_reco_app_created", "application_id", "created_at"),)

    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)

    priority: Mapped[str] = mapped_column(String(16), default="medium", nullable=False, index=True)
    severity_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Quantified bound on the opportunity, e.g. "38% of input tokens are duplicates".
    evidence: Mapped[dict[str, Any] | None] = mapped_column("evidence")
    # Where the evidence came from: metric names + their values.
    metrics: Mapped[dict[str, Any] | None] = mapped_column("metrics")
    status: Mapped[str] = mapped_column(String(32), default="open", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)


class Experiment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An A/B or multi-arm experiment over evaluation runs."""

    __tablename__ = "experiments"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    description: Mapped[str | None] = mapped_column(Text)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.PENDING, nullable=False)
    dataset_name: Mapped[str | None] = mapped_column(String(255))
    baseline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="SET NULL")
    )
    winner_variant: Mapped[str | None] = mapped_column(String(255))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    variants: Mapped[list[ExperimentVariant]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentVariant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One arm of an experiment, bound to the evaluation run that measured it."""

    __tablename__ = "experiment_variants"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    config: Mapped[dict[str, Any] | None] = mapped_column("config")
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="SET NULL")
    )

    experiment: Mapped[Experiment] = relationship(back_populates="variants")
