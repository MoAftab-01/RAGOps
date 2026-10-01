"""Analytics, dashboard, cost/quality and token-efficiency schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import TimeSeriesPoint


class DashboardOverview(BaseModel):
    """Headline numbers for the dashboard. Every field comes from a query."""

    window: str
    start: datetime
    end: datetime
    application_id: uuid.UUID | None = None
    application_name: str | None = None

    total_calls: int
    total_traces: int
    error_count: int
    error_rate: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost: float
    cost_label: str

    # Null, not 0.0: an empty window has no latency distribution. A zero here
    # would read as "every call returned instantly", which is a claim about a
    # system that was never called.
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None

    avg_tokens_per_request: float
    # Quality aggregates, null when no evaluation data exists in the window.
    retrieval_score: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    token_efficiency: float | None = None

    unique_users: int
    anomaly_count: int
    open_recommendation_count: int

    time_series: list[TimeSeriesPoint] = Field(default_factory=list)
    model_usage: list[dict[str, Any]] = Field(default_factory=list)
    application_usage: list[dict[str, Any]] = Field(default_factory=list)


class TokenAnalytics(BaseModel):
    window: str
    start: datetime
    end: datetime
    total_tokens: int
    input_tokens: int
    output_tokens: int
    avg_tokens_per_request: float
    # Percentiles are NULL upstream for a window with no calls, and that is a
    # different statement from "every call used zero tokens". The UI renders
    # null as an empty percentile rather than a chart of nothing at the origin.
    p50_tokens: float | None = None
    p95_tokens: float | None = None
    p99_tokens: float | None = None
    max_tokens: int
    cost_per_request: float
    total_cost: float
    cost_label: str
    tokens_per_user: list[dict[str, Any]] = Field(default_factory=list)
    tokens_per_application: list[dict[str, Any]] = Field(default_factory=list)
    tokens_per_model: list[dict[str, Any]] = Field(default_factory=list)
    time_series: list[TimeSeriesPoint] = Field(default_factory=list)


class CostAnalytics(BaseModel):
    window: str
    start: datetime
    end: datetime
    total_cost: float
    cost_label: str
    cost_per_request: float
    cost_per_user: float
    cost_per_application: float
    breakdown: list[dict[str, Any]] = Field(default_factory=list)
    time_series: list[TimeSeriesPoint] = Field(default_factory=list)


class LatencyAnalytics(BaseModel):
    window: str
    start: datetime
    end: datetime
    # Same reasoning as ``TokenAnalytics`` percentiles: NULL when no call in the
    # window recorded a duration, never a fabricated 0.0ms.
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    max_latency_ms: float | None = None
    breakdown_by_stage: list[dict[str, Any]] = Field(default_factory=list)
    breakdown_by_model: list[dict[str, Any]] = Field(default_factory=list)
    time_series: list[TimeSeriesPoint] = Field(default_factory=list)


class TokenEfficiencyReport(BaseModel):
    """Aggregate token-waste analysis for a window."""

    window: str
    start: datetime
    end: datetime
    score: float = Field(description="0-100 token efficiency score")
    potential_waste_pct: float
    # `wasted_tokens` and `duplicate_document_count` are means over the traces
    # that carry a recorded efficiency score, so both are per-request averages
    # and routinely fractional. Declaring them as int and rounding would report
    # "0.32 duplicates per request" as zero, i.e. as an absence of waste.
    wasted_tokens: float = Field(
        description="Mean measured wasted tokens per scored request"
    )
    total_input_tokens: int
    duplicate_document_count: float = Field(
        description="Mean duplicated retrieved documents per scored request"
    )
    avg_context_share: float
    avg_output_yield: float
    waste_by_application: list[dict[str, Any]] = Field(default_factory=list)
    # Requests whose context was dominated by duplicates or unreferenced text.
    top_offenders: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


class ModelComparison(BaseModel):
    """One row of the cost/quality scatter plot."""

    model_name: str
    provider: str
    call_count: int
    total_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_latency_ms: float
    estimated_cost: float
    cost_per_1k_tokens: float
    cost_label: str
    # Nullable, and the repository really does return None here: a model seen in
    # a trace but never registered in the pricing table has an unestablished
    # local/metered status. Coercing that to False would assert the model was
    # free, which is a claim nobody made -- so the null propagates to the UI,
    # which renders "unregistered" instead of "$0.00 (local)".
    is_local: bool | None = None
    # Quality columns are populated only where evaluation data exists.
    retrieval_score: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    quality_score: float | None = None


class CostQualityReport(BaseModel):
    window: str
    start: datetime
    end: datetime
    models: list[ModelComparison] = Field(default_factory=list)
    # Explicit statement that the dollar axis is simulated, for the UI banner.
    cost_label: str
    quality_metric_definition: str
    best_value: str | None = None
    best_quality: str | None = None
