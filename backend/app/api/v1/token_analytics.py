"""Token analytics: where the tokens went, and what they cost.

Every figure is a sum or a SQL percentile over ``llm_calls`` and ``traces`` in
the resolved window. Percentiles are computed by ``percentile_cont`` in
PostgreSQL — pulling rows into Python to sort them would put a full table scan
behind a dashboard page load.

TODO(token_service): this router calls ``analytics_repo`` directly because
``app/services/token_service.py`` does not exist. The per-denominator ratios
below (``cost_per_request`` and friends) are the kind of arithmetic a service
should own; they are here only so the endpoint is real rather than a stub.
``analytics_service.get_token_analytics`` overlaps this but cannot be
substituted for it: it binds ``avg_tokens_per_call`` to ``avg_latency_ms``
(a milliseconds figure answering a token question), omits the per-call ratios
entirely, and names the totals differently from the response schema.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DbSession, KnownScope, ScopeQuery
from app.core.cache import cached_call
from app.core.logging import get_logger
from app.repositories import analytics_repo as repo
from app.schemas.analytics import TokenAnalytics
from app.schemas.common import TimeSeriesPoint
from app.services.window import cache_parts
from app.utils.pricing import pricing_label

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics/tokens", tags=["analytics"])


def _per_call_ratio(numerator: float, denominator: int | None) -> float:
    """``numerator / denominator``, or ``0.0`` when nothing was recorded.

    The zero is a real count-of-zero divided by nothing, not a missing value
    being dressed up: the accompanying ``total_*`` says the window was empty.
    """
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def window_cost_label(by_model: list[dict]) -> str:
    """Provenance for a window's dollar figure, from the models actually seen.

    Resolved from the same rows the breakdown returns, so the label and the
    numbers can never describe different scopes. A window that only ever called
    local models is genuinely free; one that called a metered model is showing
    a simulated list price. Reporting the first as the second would make a free
    local stack look like it is accruing a bill, which is the specific lie these
    labels exist to prevent.

    With no calls at all there is no model to attribute cost to, so the label
    says that rather than defaulting to the local-free reading — which would
    claim a measurement nobody made.
    """
    labels = {pricing_label(str(row["key"])) for row in by_model if row["key"]}
    if not labels:
        return "No model calls recorded in this window"
    if len(labels) == 1:
        return labels.pop()
    return "Mixed local and metered models"


@router.get("", response_model=TokenAnalytics, summary="Token usage analytics")
async def token_analytics(session: DbSession, scope: KnownScope) -> TokenAnalytics:
    """Token totals, percentiles, and the per-user/app/model breakdowns."""
    payload = await cached_call(
        "analytics",
        lambda: _build(session, scope),
        **cache_parts(scope, "token-analytics"),
    )
    return TokenAnalytics.model_validate(payload)


async def _build(session: DbSession, scope: ScopeQuery) -> dict:
    window = scope.window_filter()
    calls = await repo.call_summary(session, window)
    per_user = await repo.grouped_by_user(session, window)
    per_application = await repo.grouped_by_application(session, window)
    per_model = await repo.grouped_by_model(session, window)
    series = await repo.bucketize(session, window, unit=scope.bucket_unit)

    total_calls = int(calls["total_calls"] or 0)
    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "total_tokens": int(calls["total_tokens"] or 0),
        "input_tokens": int(calls["input_tokens"] or 0),
        "output_tokens": int(calls["output_tokens"] or 0),
        "avg_tokens_per_request": _per_call_ratio(calls["total_tokens"] or 0, total_calls),
        "p50_tokens": calls["p50_tokens"],
        "p95_tokens": calls["p95_tokens"],
        "p99_tokens": calls["p99_tokens"],
        "max_tokens": int(calls["max_tokens"] or 0),
        "cost_per_request": _per_call_ratio(calls["estimated_cost"] or 0.0, total_calls),
        "total_cost": float(calls["estimated_cost"] or 0.0),
        "cost_label": window_cost_label(per_model),
        "tokens_per_user": per_user,
        "tokens_per_application": per_application,
        "tokens_per_model": per_model,
        "time_series": [TimeSeriesPoint.model_validate(row).model_dump() for row in series],
    }


@router.get("/by-user", summary="Token usage per user")
async def tokens_per_user(session: DbSession, scope: KnownScope) -> list[dict]:
    """Per-user token totals, heaviest first.

    Traces with no user are grouped under ``(anonymous)`` by the repository
    rather than dropped, so these rows sum to the same total as the headline
    figure above them.
    """
    return await repo.grouped_by_user(session, scope.window_filter())


@router.get("/by-model", summary="Token usage per model")
async def tokens_per_model(session: DbSession, scope: KnownScope) -> list[dict]:
    """Per-model token totals, calls, cost and p95 latency.

    Each row carries its own cost label, resolved from that row's model, so a
    table mixing local and metered models labels each line correctly instead of
    inheriting one label for the whole table.
    """
    rows = await repo.grouped_by_model(session, scope.window_filter())
    for row in rows:
        row["cost_label"] = pricing_label(str(row["key"]))
    return rows
