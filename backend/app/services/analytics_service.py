"""Dashboard and analytics aggregation.

Every number this module returns comes from a query in
:mod:`app.repositories.analytics_repo` over rows that were actually ingested.
Nothing is defaulted, smoothed, or filled in.

Two rules are load-bearing throughout:

* **An empty bucket is ``None``, never ``0``.** ``generate_series`` produces
  every bucket in the requested window whether or not any trace landed in it.
  Filling a gap with ``0`` would assert "zero requests happened", which is a
  different and false claim from "we have no data here".
* **No ``numpy`` over a full table.** Percentiles are computed by PostgreSQL's
  ``percentile_cont`` within-group, which is what :func:`analytics_repo.percentile_cont`
  wraps. Pulling the column into Python to compute it there would both be wrong
  at scale and inconsistent with what the database can do exactly.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import analytics_repo
from app.services.window import AnalyticsScope


def _avg_tokens_per_call(total_tokens: int | None, total_calls: int | None) -> float | None:
    """Mean tokens per LLM call, or ``None`` when no calls were recorded.

    Derived here rather than in SQL because ``percentile_cont`` covers the
    percentiles but there is no matching aggregate for the ratio. Guarded so a
    zero-call window yields ``None`` ("no traffic to average") instead of a
    division error or a zero that would read as "zero tokens per call".
    """
    if not total_calls:
        return None
    return total_tokens / total_calls


def _scope_fields(scope: AnalyticsScope) -> dict[str, Any]:
    """Presentation fields every analytics response echoes back."""
    return {
        "window": scope.label,
        "start": scope.start,
        "end": scope.end,
        "application_id": str(scope.application_id) if scope.application_id else None,
        "application_name": scope.application_name,
        "application_found": scope.found,
    }


async def get_summary_stats(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    """The six dashboard stat cards, each computed from a real aggregate.

    ``retrieval_score`` is the mean recorded ``RetrievalCall.top_score`` and is
    ``None`` when no retrieval call exists in the window -- an unmeasured
    score is not a score of zero.
    """
    window = scope.window_filter()
    calls = await analytics_repo.call_summary(session, window)
    traces = await analytics_repo.trace_summary(session, window)
    efficiency = await analytics_repo.token_efficiency_rollup(session, window)
    faithfulness, answer_relevance = await analytics_repo.answer_quality(
        session, window
    )

    return {
        **_scope_fields(scope),
        "total_requests": traces.get("total_traces"),
        "total_tokens": calls.get("total_tokens"),
        "total_llm_calls": calls.get("total_calls"),
        "avg_latency_ms": calls.get("avg_latency_ms"),
        "p50_latency_ms": calls.get("p50_latency_ms"),
        "p95_latency_ms": calls.get("p95_latency_ms"),
        "p99_latency_ms": calls.get("p99_latency_ms"),
        "avg_duration_ms": traces.get("avg_duration_ms"),
        "error_rate": analytics_repo.error_rate(
            traces.get("error_count"), traces.get("total_traces")
        ),
        "retrieval_score": await analytics_repo.retrieval_score(session, window),
        "avg_faithfulness": faithfulness,
        "avg_answer_relevance": answer_relevance,
        "token_efficiency": efficiency.get("score"),
        "wasted_tokens": efficiency.get("wasted_tokens"),
        "potential_waste_pct": efficiency.get("potential_waste_pct"),
        "total_cost": calls.get("estimated_cost"),
    }


async def get_time_series(
    session: AsyncSession,
    scope: AnalyticsScope,
    *,
    interval: str | None = None,
) -> dict[str, Any]:
    """Bucketed request / token / latency series across the window.

    Empty buckets arrive from :func:`analytics_repo.bucketize` as ``None`` and
    are passed through unchanged.
    """
    window = scope.window_filter()
    unit = interval or scope.bucket_unit
    return {
        **_scope_fields(scope),
        "interval": unit,
        "series": await analytics_repo.bucketize(session, window, unit=unit),
    }


async def get_token_analytics(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    """Token volume, percentiles, and the *measured* waste profile.

    ``potential_waste_pct`` is the ratio of tokens that the recorded
    ingest-time analysis identified as wasted. It is a measurement of what is
    in the data, not a projection of money saved by some hypothetical fix --
    no such number is produced anywhere in this module.
    """
    window = scope.window_filter()
    calls = await analytics_repo.call_summary(session, window)
    efficiency = await analytics_repo.token_efficiency_rollup(session, window)

    return {
        **_scope_fields(scope),
        "total_llm_calls": calls.get("total_calls"),
        "total_input_tokens": calls.get("input_tokens"),
        "total_output_tokens": calls.get("output_tokens"),
        "total_tokens": calls.get("total_tokens"),
        "avg_tokens_per_call": _avg_tokens_per_call(
            calls.get("total_tokens"), calls.get("total_calls")
        ),
        "p50_tokens": calls.get("p50_tokens"),
        "p95_tokens": calls.get("p95_tokens"),
        "p99_tokens": calls.get("p99_tokens"),
        "max_tokens": calls.get("max_tokens"),
        "efficiency_score": efficiency.get("score"),
        "duplicate_document_count": efficiency.get("duplicate_document_count"),
        "context_share": efficiency.get("avg_context_share"),
        "output_yield": efficiency.get("avg_output_yield"),
        "wasted_tokens": efficiency.get("wasted_tokens"),
        "potential_waste_pct": efficiency.get("potential_waste_pct"),
        "scored_traces": efficiency.get("scored_traces"),
        "basis": "measured_from_ingest_metadata",
    }


async def get_latency_distribution(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    """Latency percentiles, computed in PostgreSQL by ``percentile_cont``."""
    window = scope.window_filter()
    # `latency_summary` is the call-level view (over llm_calls); the trace-level
    # percentiles come from trace_summary. Both are returned so a client can
    # show request duration and per-call generation latency separately.
    return {
        **_scope_fields(scope),
        "llm_calls": await analytics_repo.latency_summary(session, window),
        "traces": await analytics_repo.trace_summary(session, window),
    }


async def get_model_breakdown(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    """Per-model token and simulated-cost rollup."""
    window = scope.window_filter()
    return {
        **_scope_fields(scope),
        "models": await analytics_repo.grouped_by_model(session, window),
    }


async def get_user_breakdown(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    window = scope.window_filter()
    return {
        **_scope_fields(scope),
        "users": await analytics_repo.grouped_by_user(session, window),
    }


async def get_application_breakdown(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    window = scope.window_filter()
    return {
        **_scope_fields(scope),
        "applications": await analytics_repo.grouped_by_application(session, window),
    }


async def get_span_kind_breakdown(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    window = scope.window_filter()
    return {
        **_scope_fields(scope),
        "span_kinds": await analytics_repo.grouped_by_span_kind(session, window),
    }


async def get_cost_quality(
    session: AsyncSession, scope: AnalyticsScope
) -> dict[str, Any]:
    """Cost-vs-quality scatter data.

    ``cost`` is a *simulated* figure derived from the reference pricing table,
    and every row carries ``pricing_label`` so the UI can mark it as such.
    Local models legitimately cost ``0.0`` -- that is a real measurement of a
    free local model, not a missing value.
    """
    window = scope.window_filter()
    return {
        **_scope_fields(scope),
        "pricing_note": (
            "Cost is a simulation from the reference pricing table in "
            "app.utils.pricing, not a provider invoice."
        ),
        "models": await analytics_repo.model_comparison(session, window),
    }


__all__ = [
    "get_application_breakdown",
    "get_cost_quality",
    "get_latency_distribution",
    "get_model_breakdown",
    "get_span_kind_breakdown",
    "get_summary_stats",
    "get_time_series",
    "get_token_analytics",
    "get_user_breakdown",
]
