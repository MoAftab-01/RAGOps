"""Stored optimisation recommendations, and the endpoint that produces them.

Recommendations are evidence-backed by construction: the model refuses to store
one whose ``evidence`` does not name the measurement that triggered it, and
:class:`~app.services.recommendation_service.RecommendationEngine` refuses to
build one without a measured value to cite. That makes them the one place in
RAGOps where a *claim* is stored deliberately, which is why the rules that
produce them are deterministic functions of recorded data rather than generated
text.

The listing side is complete and independent of the generator: recommendations
written by the engine, by a script, or by a person at the psql prompt are all
readable here with their evidence intact.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import (
    DbSession,
    KnownScope,
    Pagination,
    TenantPrincipal,
    WriteGuard,
    require_known_application,
    resolve_application_or_404,
)
from app.core.logging import get_logger
from app.models import Application, OptimizationRecommendation, utcnow
from app.repositories.common import apply_pagination, apply_tenant_filter, count_rows
from app.schemas.common import Page
from app.schemas.insights import RecommendationGenerateRequest, RecommendationOut
from app.services.recommendation_service import RecommendationEngine
from app.services.window import resolve_application_and_window

logger = get_logger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationGenerateResponse(BaseModel):
    """Result of a generation request.

    ``notes`` is not decoration. A response of ``generated: 0`` with no
    explanation is indistinguishable from "we looked and found nothing to
    improve", and those two are very different things to show a user. The note
    is where the route says which one this is.

    ``rules_evaluated`` closes the remaining gap: it reports every rule that ran
    and the measurement it looked at, including the ones that stayed below
    their threshold. A caller can then see that "no recommendation" means "six
    statistics were computed and none justified one", rather than having to
    trust that something happened at all.
    """

    generated: int
    recommendations: list[RecommendationOut] = Field(default_factory=list)
    notes: str | None = None
    rules_evaluated: list[dict[str, Any]] = Field(default_factory=list)


@router.get("", response_model=Page[RecommendationOut], summary="List recommendations")
async def list_recommendations(
    session: DbSession,
    scope: KnownScope,
    pagination: Pagination,
    category: Annotated[str | None, Query(description="Filter by category")] = None,
    priority: Annotated[str | None, Query(description="Filter by priority")] = None,
    status: Annotated[str | None, Query(description="Filter by status")] = None,
) -> Page[RecommendationOut]:
    """Stored recommendations in the window, newest first.

    Sorted by ``severity_score`` rather than recency: this list exists to be
    worked through, and the most severe recorded opportunity should not be
    buried under a page of recent trivia. ``created_at`` breaks ties so the
    order is stable across pages.
    """
    stmt = (
        select(OptimizationRecommendation, Application.name)
        .join(Application, OptimizationRecommendation.application_id == Application.id, isouter=True)
        .where(
            OptimizationRecommendation.created_at >= scope.start,
            OptimizationRecommendation.created_at <= scope.end,
        )
        .order_by(
            OptimizationRecommendation.severity_score.desc(),
            OptimizationRecommendation.created_at.desc(),
        )
    )
    stmt = apply_tenant_filter(
        stmt, scope.window_filter(), OptimizationRecommendation.application_id
    )
    if category:
        stmt = stmt.where(OptimizationRecommendation.category == category)
    if priority:
        stmt = stmt.where(OptimizationRecommendation.priority == priority)
    if status:
        stmt = stmt.where(OptimizationRecommendation.status == status)

    total = int((await session.execute(count_rows(stmt))).scalar_one())
    rows = (
        await session.execute(apply_pagination(stmt, pagination.page, pagination.page_size))
    ).all()

    return Page[RecommendationOut].build(
        items=[_to_out(recommendation, application_name) for recommendation, application_name in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.post(
    "/generate",
    response_model=RecommendationGenerateResponse,
    summary="Generate recommendations from recorded data",
    dependencies=[WriteGuard],
)
async def generate_recommendations(
    session: DbSession,
    payload: RecommendationGenerateRequest,
    principal: TenantPrincipal,
) -> RecommendationGenerateResponse:
    """Run the evidence-gated rule set and return what it found.

    Resolves and validates the request exactly as before — an unknown
    application is a 404 — then hands the window to
    :class:`~app.services.recommendation_service.RecommendationEngine`.

    A pass that produces nothing still returns ``rules_evaluated``: the metrics
    each rule looked at, and why it stayed quiet. "No rule fired" and "no rule
    ran" are different answers, and only the first one is a finding about your
    deployment.
    """
    await resolve_application_or_404(session, payload.application, principal)
    logger.info(
        "recommendations.generate.requested",
        application=payload.application,
        window=payload.window,
        persist=payload.persist,
        min_severity=payload.min_severity,
    )

    end = utcnow()
    scope = await resolve_application_and_window(
        session,
        application_name=payload.application,
        window=payload.window,
        start=None,
        end=end,
    )
    scope = await require_known_application(scope)

    engine = RecommendationEngine()
    result = await engine.generate(
        session,
        scope,
        min_severity=payload.min_severity,
        persist=payload.persist,
    )

    return RecommendationGenerateResponse(
        generated=result.generated,
        recommendations=[
            _to_out(recommendation, scope.application_name)
            for recommendation in result.recommendations
        ],
        rules_evaluated=result.rules_evaluated,
        notes=(
            f"Ran {len(result.rules_evaluated)} deterministic rules over "
            f"{scope.window} for {scope.application_name or 'all applications'}; "
            f"{result.generated} produced an evidence-backed recommendation."
            if result.generated
            else (
                f"All {len(result.rules_evaluated)} rules ran over {scope.window} "
                f"for {scope.application_name or 'all applications'} and none "
                f"crossed its threshold. That is a measurement, not an absence of "
                f"analysis — see rules_evaluated for what each rule looked at."
            )
        ),
    )




def _to_out(
    recommendation: OptimizationRecommendation, application_name: str | None
) -> RecommendationOut:
    """Project a stored recommendation, naming the application it was joined to.

    ``application_name`` is null when the row's application has since been
    deleted, or when the recommendation is platform-wide and has no application
    at all. Neither is reported as an empty string.

    ``id`` and ``created_at`` are the two fields a row only has once it has been
    written, and both are nullable here so that a dry run (``persist: false``)
    can still show what *would* be stored. A dry run reporting a made-up id
    would be worse than reporting none: the value would look like a link to a
    recommendation that does not exist.
    """
    return RecommendationOut(
        id=recommendation.id,
        application_id=recommendation.application_id,
        application_name=application_name,
        category=recommendation.category,
        title=recommendation.title,
        rationale=recommendation.rationale,
        recommendation=recommendation.recommendation,
        priority=recommendation.priority,
        severity_score=float(recommendation.severity_score),
        evidence=recommendation.evidence,
        metrics=recommendation.metrics,
        status=recommendation.status,
        confidence=float(recommendation.confidence),
        created_at=recommendation.created_at,
    )
