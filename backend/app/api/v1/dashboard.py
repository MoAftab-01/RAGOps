"""Dashboard overview: the landing payload the whole UI hangs off.

Everything here is a projection of repository aggregates. The rule this router
exists to enforce is that *absence stays absence*: a window with no retrieval
recorded reports ``retrieval_score: null``, not ``0.0``; a window with no
answer evaluation reports ``faithfulness: null``, not ``0.0``. The difference
is not cosmetic — ``0.0`` says "retrieval is failing", and a dashboard that
says that before anything is instrumented is lying about the user's system.

TODO(analytics_service): this router calls ``analytics_repo`` directly and
assembles the response itself, even though ``app/services/analytics_service.py``
now exists. It was written before the service, and the two have since diverged:
the service returns ``total_requests``/``total_llm_calls``/``total_cost`` where
this route's schema says ``total_traces``/``total_calls``/``estimated_cost``, and
it carries no cost label. Reconciling the service's key names with
``DashboardOverview`` is the service's job, not this route's — so the aggregation
stays here for now rather than being rewritten against a schema that would change
underneath it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.deps import DbSession, KnownScope, ScopeQuery
from app.api.v1.token_analytics import window_cost_label
from app.core.cache import cached_call
from app.core.logging import get_logger
from app.models import Anomaly, OptimizationRecommendation
from app.repositories import analytics_repo as repo
from app.repositories.common import apply_tenant_filter
from app.schemas.analytics import DashboardOverview
from app.schemas.common import TimeSeriesPoint
from app.services.window import AnalyticsScope, cache_parts
from app.utils.pricing import pricing_label

logger = get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get(
    "/overview",
    response_model=DashboardOverview,
    summary="Dashboard overview for a window",
)
async def overview(session: DbSession, scope: KnownScope) -> DashboardOverview:
    """Headline numbers, one time series, and the two usage breakdowns.

    The query is memoised on the resolved scope: the dashboard polls this on a
    timer and the underlying aggregates are identical between refreshes.
    """
    payload = await cached_call(
        "analytics",
        lambda: _build_overview(session, scope),
        **cache_parts(scope, "dashboard-overview"),
    )
    return DashboardOverview.model_validate(payload)


async def _build_overview(session: DbSession, scope: ScopeQuery) -> dict:
    window = scope.window_filter()

    calls = await repo.call_summary(session, window)
    traces = await repo.trace_summary(session, window)
    # Null when nothing was retrieved / evaluated in the window, and left null
    # all the way to the response. See the module docstring.
    retrieval = await repo.retrieval_score(session, window)
    faithfulness, answer_relevance = await repo.answer_quality(session, window)
    efficiency = await repo.token_efficiency_rollup(session, window)
    series = await repo.bucketize(session, window, unit=scope.bucket_unit)
    by_model = await repo.grouped_by_model(session, window)
    by_application = await repo.grouped_by_application(session, window)
    cost_label = window_cost_label(by_model)
    open_recommendations, anomaly_total = await _open_insight_counts(session, scope)

    total_calls = int(calls["total_calls"] or 0)
    total_tokens = int(calls["total_tokens"] or 0)

    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "application_id": scope.application_id,
        "application_name": scope.application_name,
        "total_calls": total_calls,
        "total_traces": int(traces["total_traces"] or 0),
        "error_count": int(traces["error_count"] or 0),
        "error_rate": repo.error_rate(traces["error_count"], traces["total_traces"]),
        "total_input_tokens": int(calls["input_tokens"] or 0),
        "total_output_tokens": int(calls["output_tokens"] or 0),
        "total_tokens": total_tokens,
        "estimated_cost": float(calls["estimated_cost"] or 0.0),
        "cost_label": cost_label,
        # Latency aggregates are NULL upstream for an empty window and stay
        # null here, for the same reason retrieval_score does: a 0.0ms latency
        # on a window with no calls in it is a statement about nobody.
        "avg_latency_ms": _optional_float(calls["avg_latency_ms"]),
        "p50_latency_ms": _optional_float(calls["p50_latency_ms"]),
        "p95_latency_ms": _optional_float(calls["p95_latency_ms"]),
        "p99_latency_ms": _optional_float(calls["p99_latency_ms"]),
        "avg_tokens_per_request": (total_tokens / total_calls) if total_calls else 0.0,
        "retrieval_score": retrieval,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "token_efficiency": efficiency.get("score"),
        "unique_users": int(traces["unique_users"] or 0),
        "anomaly_count": anomaly_total,
        "open_recommendation_count": open_recommendations,
        "time_series": [TimeSeriesPoint.model_validate(row).model_dump() for row in series],
        "model_usage": [_model_usage_row(row) for row in by_model],
        "application_usage": [
            _application_usage_row(row, cost_label) for row in by_application
        ],
    }


def _optional_float(value: Any) -> float | None:
    """A SQL aggregate as a float, or ``None`` when the database returned NULL.

    ``percentile_cont`` over no rows is NULL, and so is ``avg`` over a window
    where every value was NULL. Both mean "nothing was measured here", which the
    response must be able to say.
    """
    return None if value is None else float(value)


def _model_usage_row(row: dict) -> dict:
    """Project one per-model breakdown row onto the overview's usage shape."""
    model_name = str(row["key"])
    return {
        "model_name": model_name,
        "provider": row["extra"].get("provider"),
        "call_count": int(row["count"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "estimated_cost": float(row["estimated_cost"] or 0.0),
        "cost_label": pricing_label(model_name),
    }


def _application_usage_row(row: dict, window_label: str) -> dict:
    """Project one per-application row, labelling its cost for the whole window.

    A per-application breakdown groups by application alone and carries no
    model attribution, so the label is the one resolved for the window these
    rows are a slice of. That is the scope the dollar figure is actually
    covered by, and it is why every row in a window where only local models
    ran is labelled local-free rather than "no model calls" — a figure that is
    sitting right there is not an absence.
    """
    return {
        "application": str(row["key"]),
        "trace_count": int(row["count"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "estimated_cost": float(row["estimated_cost"] or 0.0),
        "cost_label": window_label,
    }


async def _open_insight_counts(session: DbSession, scope: AnalyticsScope) -> tuple[int, int]:
    """``(open recommendations, total anomalies)`` for the scope.

    Scoped through the whole scope so the cards agree with the rest of the
    payload. It takes an :class:`AnalyticsScope` rather than a bare
    ``application_id`` because the latter is a spelling of "filter by
    application" with no place to put an organization -- and a dashboard whose
    insight cards counted every company's open recommendations while the rest of
    the panel showed only the tenant's would contradict itself in the most
    visible way available.

    TODO(anomaly_repo): these two counts belong in the repository layer with
    the anomaly and recommendation queries; they are here only because those
    modules do not exist yet.
    """
    recommendations = select(func.count(OptimizationRecommendation.id)).where(
        OptimizationRecommendation.status == "open"
    )
    anomalies = select(func.count(Anomaly.id))
    window = scope.window_filter()
    recommendations = apply_tenant_filter(
        recommendations, window, OptimizationRecommendation.application_id
    )
    anomalies = apply_tenant_filter(anomalies, window, Anomaly.application_id)
    return (
        int((await session.execute(recommendations)).scalar_one()),
        int((await session.execute(anomalies)).scalar_one()),
    )
