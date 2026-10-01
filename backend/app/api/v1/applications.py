"""Application inventory, and the per-application statistics page.

Two different questions, kept in one router because they answer the same
question at different scopes. ``GET /applications`` is a catalogue: what exists,
and is it still being used. ``GET /applications/{id}/stats`` is a measurement
over one window, and every figure in it is scoped to that application only — the
window is the shared ``ScopeQuery``, so this page and the dashboard can never
disagree about which traces they are looking at.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.deps import DbSession, KnownScope, Pagination, TenantPrincipal, not_found
from app.core.cache import cached_call
from app.core.logging import get_logger
from app.models import Application, Trace
from app.repositories import analytics_repo as repo
from app.repositories.common import apply_organization_filter, apply_pagination, count_rows
from app.schemas.common import Page, TimeSeriesPoint
from app.schemas.insights import ApplicationOut
from app.services.window import AnalyticsScope, cache_parts

logger = get_logger(__name__)

router = APIRouter(prefix="/applications", tags=["applications"])


class ApplicationStats(BaseModel):
    """Everything the per-application statistics page shows, in one object.

    Not a paginated shape: this is a single window's summary, and the totals
    (``trace_count``, ``total_tokens``) describe the whole window rather than a
    page of it. ``model_usage`` and ``time_series`` are complete for the window
    too — paging them would make the totals and the breakdown describe different
    populations.
    """

    application: str
    application_id: uuid.UUID
    window: str
    start: datetime
    end: datetime
    trace_count: int
    total_tokens: int
    estimated_cost: float
    cost_label: str
    avg_latency_ms: float | None
    error_rate: float
    model_usage: list[dict[str, Any]] = Field(default_factory=list)
    time_series: list[TimeSeriesPoint] = Field(default_factory=list)


@router.get("", response_model=Page[ApplicationOut], summary="List applications")
async def list_applications(
    session: DbSession,
    pagination: Pagination,
    principal: TenantPrincipal,
    include_inactive: bool = False,
) -> Page[ApplicationOut]:
    """Registered applications, with their recorded activity.

    ``trace_count``, ``last_seen_at`` and ``total_tokens`` are lifetime figures
    over the whole table, not a window: this is the catalogue, and a
    "last seen 3 days ago" that silently meant "in the selected window" would be
    a different number answering the same question. An application with no
    traces has all three null rather than zero — nothing was recorded, which is
    not the same as a recorded zero.

    A tenant sees only its own organization's applications. This route has no
    ``?application=`` filter, so the shared ``window_scope`` funnel never runs
    and there is no ``WindowFilter`` to narrow it -- which makes this the one
    place a company could enumerate its competitors' application names off the
    API.
    """
    activity = (
        select(
            Trace.application_id.label("application_id"),
            func.count(Trace.id).label("trace_count"),
            func.max(Trace.start_time).label("last_seen_at"),
            func.coalesce(func.sum(Trace.total_tokens), 0).label("total_tokens"),
        )
        .group_by(Trace.application_id)
        .subquery()
    )

    stmt = (
        select(Application, activity.c.trace_count, activity.c.last_seen_at, activity.c.total_tokens)
        .outerjoin(activity, Application.id == activity.c.application_id)
        .order_by(Application.name.asc())
    )
    if not include_inactive:
        stmt = stmt.where(Application.is_active.is_(True))
    stmt = apply_organization_filter(
        stmt, principal.organization_id, Application.organization_id
    )

    total = int((await session.execute(count_rows(stmt))).scalar_one())
    result = await session.execute(apply_pagination(stmt, pagination.page, pagination.page_size))
    rows = result.all()

    items = [
        ApplicationOut(
            id=application.id,
            name=application.name,
            description=application.description,
            environment=application.environment,
            is_active=application.is_active,
            created_at=application.created_at,
            trace_count=int(trace_count) if trace_count is not None else None,
            last_seen_at=last_seen_at,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
        )
        for application, trace_count, last_seen_at, total_tokens in rows
    ]
    return Page[ApplicationOut].build(
        items=items,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get(
    "/{application_id}/stats",
    response_model=ApplicationStats,
    summary="Statistics for one application",
)
async def application_stats(
    session: DbSession,
    application_id: uuid.UUID,
    scope: KnownScope,
    principal: TenantPrincipal,
) -> ApplicationStats:
    """One application's activity in the requested window.

    404s on an unknown id: this URL names a specific application, so silently
    falling back to platform-wide totals would answer a different question than
    the one asked.

    ``avg_latency_ms`` is null on a window with no trace, and ``estimated_cost``
    is 0.0 only when the application was called and the calls really did cost
    nothing — both come from the aggregate rather than from a fallback, and the
    distinction is preserved in the response.
    """
    application = (
        await session.execute(
            apply_organization_filter(
                select(Application).where(Application.id == application_id),
                principal.organization_id,
                Application.organization_id,
            )
        )
    ).scalar_one_or_none()
    if application is None:
        # 404, not 403. This route names a specific application, so it has no
        # window to narrow -- but a 403 would confirm the id exists, which is
        # itself the enumeration channel, and the platform answer ("no such
        # application") is both safe and the codebase's existing idiom.
        raise await not_found(f"No application with id {application_id}.")

    # The URL names the application; the shared scope carries whatever the
    # caller also passed as ``?application=``. The URL wins, and a disagreeing
    # query parameter is a client bug worth reporting rather than a silent
    # re-scope — otherwise a caller asking for application A's stats would
    # receive B's, with A's name on the page.
    if scope.application_id is not None and scope.application_id != application_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"?application resolves to a different application than the "
                f"path id {application_id}. Omit the query parameter, or use "
                f"the same application in both."
            ),
        )
    scoped = replace(scope, application_id=application_id, application_name=application.name)

    payload = await cached_call(
        "analytics",
        lambda: _build_stats(session, scoped),
        **cache_parts(scoped, "application-stats"),
    )
    return ApplicationStats.model_validate(
        {
            "application": application.name,
            "application_id": application.id,
            **payload,
        }
    )


async def _build_stats(session: DbSession, scope: AnalyticsScope) -> dict[str, Any]:
    """Window aggregates for the scope's application.

    The scope's own ``application_id`` is used rather than the URL's, after
    :func:`_scoped_scope` has confirmed the two agree — otherwise the cache key
    and the query could describe different applications.
    """
    window = scope.window_filter()
    traces = await repo.trace_summary(session, window)
    per_model = await repo.grouped_by_model(session, window)
    series = await repo.bucketize(session, window, unit=scope.bucket_unit)
    avg_duration = traces.get("avg_duration_ms")
    cost = float(traces["estimated_cost"] or 0.0)

    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "trace_count": int(traces["total_traces"] or 0),
        "total_tokens": int(traces["total_tokens"] or 0),
        "estimated_cost": cost,
        # Every dollar figure is labelled, because the price behind it is the
        # registry's simulated list price unless the model is local.
        "cost_label": _cost_label(per_model),
        # Null, not 0.0: no trace in the window recorded a duration, which is
        # not the same as every trace being instantaneous.
        "avg_latency_ms": None if avg_duration is None else float(avg_duration),
        "error_rate": repo.error_rate(traces.get("error_count"), traces.get("total_traces")),
        "model_usage": list(per_model),
        "time_series": [TimeSeriesPoint.model_validate(row).model_dump() for row in series],
    }


def _cost_label(per_model: list[dict[str, Any]]) -> str:
    """Provenance of the cost figure, from the models that were actually used.

    An application that made no calls in this window is labelled as such rather
    than "Simulated list price": there is no price behind a figure that no call
    contributed to.
    """
    if not per_model:
        return "No model calls in this window"
    local = {row.get("is_local") for row in per_model if row.get("is_local") is not None}
    if not local:
        return "Unregistered models — cost unavailable"
    if local == {True}:
        return "Local inference — no API cost"
    if local == {False}:
        return "Simulated list price"
    return "Mixed local and metered models"
