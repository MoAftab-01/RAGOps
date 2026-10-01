"""Anomaly, recommendation, experiment and application schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------


class AnomalyOut(ORMModel):
    id: uuid.UUID
    application_id: uuid.UUID | None
    application_name: str | None = None
    trace_id: str | None
    anomaly_type: str
    severity: str
    metric: str
    observed_value: float | None
    expected_low: float | None
    expected_high: float | None
    anomaly_score: float | None
    peer_median: float | None
    peer_p95: float | None
    evidence: dict[str, Any] | None
    detected_at: datetime
    detection_method: str
    is_resolved: bool


class AnomalyDetectRequest(BaseModel):
    """Run IsolationForest over recent telemetry.

    The model is fitted per application on a feature matrix built from recorded
    fields only. There is no target label, so the fit is unsupervised and the
    "expected band" comes from the same window's percentiles.
    """

    application: str | None = Field(default=None, max_length=128)
    window: str = Field(default="7d")
    lookback_hours: int = Field(default=168, ge=1, le=2160)
    contamination: float | None = Field(
        default=None, ge=0.001, le=0.2, description="Override the configured rate"
    )
    features: list[str] = Field(
        default_factory=lambda: [
            "total_tokens",
            "duration_ms",
            "context_tokens",
            "num_documents",
            "agent_iterations",
            "estimated_cost",
        ]
    )
    persist: bool = True


class AnomalyDetectResponse(BaseModel):
    application_id: uuid.UUID | None
    application_name: str | None
    num_samples: int
    num_features: int
    contamination: float
    detection_method: str = "isolation_forest"
    num_anomalies: int
    anomaly_rate: float
    feature_importance_proxy: dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature deviation magnitude, not a fitted importance score",
    )
    anomalies: list[AnomalyOut] = Field(default_factory=list)
    duration_ms: float


# ---------------------------------------------------------------------------
# Optimization recommendations
# ---------------------------------------------------------------------------


class RecommendationOut(ORMModel):
    # Null only for a dry run. A generation with ``persist: false`` produces the
    # same recommendation object but writes no row, so it has no id and no
    # creation time; the route reports those as null rather than inventing them,
    # because an id that looks real but resolves to nothing is a broken link.
    id: uuid.UUID | None = None
    application_id: uuid.UUID | None
    application_name: str | None = None
    category: str
    title: str
    rationale: str
    recommendation: str
    priority: str
    severity_score: float
    evidence: dict[str, Any] | None
    metrics: dict[str, Any] | None
    status: str
    confidence: float
    # Server-assigned on write, so null for the same reason ``id`` is.
    created_at: datetime | None = None


class RecommendationGenerateRequest(BaseModel):
    application: str | None = Field(default=None, max_length=128)
    window: str = Field(default="7d")
    persist: bool = True
    min_severity: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class ExperimentVariantIn(BaseModel):
    name: str = Field(max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)
    run_id: uuid.UUID | None = None


class ExperimentCreate(BaseModel):
    name: str = Field(max_length=255)
    application: str | None = Field(default=None, max_length=128)
    description: str | None = None
    hypothesis: str | None = None
    dataset_name: str | None = None
    variants: list[ExperimentVariantIn] = Field(min_length=2, max_length=10)
    baseline_variant: str | None = Field(
        default=None, description="Variant name to treat as baseline"
    )


class ExperimentVariantOut(ORMModel):
    id: uuid.UUID
    name: str
    config: dict[str, Any] | None
    run_id: uuid.UUID | None
    metrics: dict[str, Any] | None = None
    deltas_vs_baseline: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentOut(ORMModel):
    id: uuid.UUID
    name: str
    application_id: uuid.UUID | None
    description: str | None
    hypothesis: str | None
    status: str
    dataset_name: str | None
    baseline_run_id: uuid.UUID | None
    winner_variant: str | None
    completed_at: datetime | None
    created_at: datetime
    variants: list[ExperimentVariantOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Applications and models
# ---------------------------------------------------------------------------


class ApplicationOut(ORMModel):
    id: uuid.UUID
    name: str
    description: str | None
    environment: str
    is_active: bool
    created_at: datetime
    trace_count: int | None = None
    last_seen_at: datetime | None = None
    total_tokens: int | None = None


class ModelOut(ORMModel):
    id: uuid.UUID
    name: str
    provider: str
    context_window: int | None
    is_local: bool
    input_cost_per_1k: float
    output_cost_per_1k: float
    description: str | None


class ModelUpsert(BaseModel):
    name: str = Field(max_length=128)
    provider: str = Field(default="ollama", max_length=64)
    context_window: int | None = None
    is_local: bool = True
    input_cost_per_1k: float = Field(default=0.0, ge=0)
    output_cost_per_1k: float = Field(default=0.0, ge=0)
    description: str | None = None
