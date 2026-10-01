"""Shared response envelopes and pagination models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas serialised directly from ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None


class Page(BaseModel, Generic[T]):
    """Paginated payload. The frontend renders the same envelope everywhere."""

    items: list[T]
    total: int = Field(description="Total rows matching the filter, ignoring pagination")
    page: int
    page_size: int
    has_next: bool

    @classmethod
    def build(
        cls, items: list[T], total: int, page: int, page_size: int
    ) -> Page[T]:
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total,
        )


class TimeSeriesPoint(BaseModel):
    """One bucket of a time series.

    ``bucket`` is ISO-8601 UTC. Metrics are ``None`` rather than 0 for empty
    buckets so the UI can break the line instead of drawing a false drop to zero.
    """

    bucket: datetime
    request_count: int | None = None
    error_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    avg_retrieval_score: float | None = None
    avg_faithfulness: float | None = None
    avg_answer_relevance: float | None = None


class BreakdownItem(BaseModel):
    """A ``GROUP BY`` row. ``extra`` carries dimension-specific fields."""

    key: str
    label: str | None = None
    count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    avg_latency_ms: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class MessageResponse(BaseModel):
    message: str
    detail: dict[str, Any] | None = None
