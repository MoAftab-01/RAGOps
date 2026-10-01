"""Cost and latency analytics, and the cost/quality trade-off table.

The cost/quality page exists to answer "is the expensive model worth it". It can
only answer that honestly if both axes are labelled for what they are: the cost
axis is a *simulated* list price for models RAGOps does not call, and the
quality axis is a mean of persisted answer evaluations or nothing at all. A
model with no evaluation is reported with null quality rather than ranked
against a mean that was never computed — ranking unmeasured models as if they
were the best in the set is the specific failure this router exists to prevent.

TODO(cost_service): this router calls ``analytics_repo`` directly because
``app/services/cost_service.py`` and ``latency_service.py`` do not exist. The
per-denominator ratios below are service work that moved here only so the
endpoints return real numbers instead of stubs.
``analytics_service.get_cost_quality`` and ``get_latency_distribution`` cover the
same ground but do not return this router's response shapes: the former carries a
``pricing_note`` string instead of a per-row ``cost_label``, and the latter
nests the percentiles under ``llm_calls``/``traces`` rather than flattening them.
The per-row label is the load-bearing part, so the assembly stays here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import Float, and_, func, select

from app.api.deps import DbSession, KnownScope
from app.api.v1.token_analytics import _per_call_ratio, window_cost_label
from app.core.cache import cached_call
from app.core.logging import get_logger
from app.models import AnswerEvaluation, Application, LLMCall, Trace
from app.repositories import analytics_repo as repo
from app.repositories.common import WindowFilter, apply_tenant_filter
from app.schemas.analytics import (
    CostAnalytics,
    CostQualityReport,
    LatencyAnalytics,
    ModelComparison,
    TokenEfficiencyReport,
)
from app.schemas.common import TimeSeriesPoint
from app.services.window import AnalyticsScope, cache_parts
from app.utils.pricing import pricing_label

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

#: Shown next to the quality column so a reader knows what the number is a mean
#: of, and that its absence means "no evaluation was run", not "evaluated badly".
QUALITY_DEFINITION = (
    "Mean of faithfulness and answer relevance from answer_evaluations recorded "
    "in this window; null means no evaluation has been run for that model."
)


@router.get("/cost", response_model=CostAnalytics, summary="Cost analytics")
async def cost_analytics(session: DbSession, scope: KnownScope) -> CostAnalytics:
    """Total cost and the per-unit divisions, with the provenance label."""
    payload = await cached_call(
        "analytics",
        lambda: _build_cost(session, scope),
        **cache_parts(scope, "cost-analytics"),
    )
    return CostAnalytics.model_validate(payload)


async def _build_cost(session: DbSession, scope: AnalyticsScope) -> dict[str, Any]:
    window = scope.window_filter()
    calls = await repo.call_summary(session, window)
    traces = await repo.trace_summary(session, window)
    per_model = await repo.grouped_by_model(session, window)
    series = await repo.bucketize(session, window, unit=scope.bucket_unit)

    total_cost = float(calls["estimated_cost"] or 0.0)
    total_calls = int(calls["total_calls"] or 0)
    unique_users = int(traces["unique_users"] or 0)
    # "Per application" means per distinct application, not per row in the
    # breakdown: the breakdown is limited to 50 rows, so its length is a page
    # size rather than a count of applications.
    num_applications = await _count_applications(session, scope)

    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "total_cost": total_cost,
        "cost_label": window_cost_label(per_model),
        "cost_per_request": _per_call_ratio(total_cost, total_calls),
        "cost_per_user": _per_call_ratio(total_cost, unique_users),
        "cost_per_application": _per_call_ratio(total_cost, num_applications),
        # Each row carries its own label so a table mixing local and metered
        # models labels every line, not just the table as a whole.
        "breakdown": [{**row, "cost_label": _model_cost_label(row)} for row in per_model],
        "time_series": [TimeSeriesPoint.model_validate(row).model_dump() for row in series],
    }


async def _count_applications(session: DbSession, scope: AnalyticsScope) -> int:
    """Distinct applications with traffic in the window.

    TODO(application_repo): a platform-wide count has no service to call; this
    is one grouped aggregate and belongs with the application queries.
    """
    stmt = select(func.count(func.distinct(Trace.application_id))).where(
        Trace.start_time >= scope.start, Trace.start_time <= scope.end
    )
    stmt = apply_tenant_filter(stmt, scope.window_filter(), Trace.application_id)
    return int((await session.execute(stmt)).scalar_one())


def _model_cost_label(row: dict[str, Any]) -> str:
    """Cost provenance for one breakdown row, resolved from that row's model."""
    return pricing_label(str(row["key"]))


@router.get("/latency", response_model=LatencyAnalytics, summary="Latency analytics")
async def latency_analytics(session: DbSession, scope: KnownScope) -> LatencyAnalytics:
    """Generation latency percentiles, broken down by stage and by model.

    ``breakdown_by_stage`` comes from ``spans``, which is where the per-stage
    timings of the RAG pipeline (embedding, retrieval, rerank, generation) are
    actually recorded; there is no other source for them.
    """
    payload = await cached_call(
        "analytics",
        lambda: _build_latency(session, scope),
        **cache_parts(scope, "latency-breakdown"),
    )
    return LatencyAnalytics.model_validate(payload)


async def _build_latency(session: DbSession, scope: AnalyticsScope) -> dict[str, Any]:
    window = scope.window_filter()
    calls = await repo.call_summary(session, window)
    by_stage = await repo.grouped_by_span_kind(session, window)
    by_model = await repo.grouped_by_model(session, window)
    series = await repo.bucketize(session, window, unit=scope.bucket_unit)

    # ``call_summary`` reports the largest *token* count, not the largest
    # latency, so the maximum comes from its own aggregate below.
    max_latency = await _max_latency_ms(session, window)

    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "avg_latency_ms": float(calls["avg_latency_ms"] or 0.0),
        "p50_latency_ms": float(calls["p50_latency_ms"] or 0.0),
        "p95_latency_ms": float(calls["p95_latency_ms"] or 0.0),
        "p99_latency_ms": float(calls["p99_latency_ms"] or 0.0),
        "max_latency_ms": max_latency,
        "breakdown_by_stage": by_stage,
        "breakdown_by_model": by_model,
        "time_series": [TimeSeriesPoint.model_validate(row).model_dump() for row in series],
    }


async def _max_latency_ms(session: DbSession, window: WindowFilter) -> float:
    """Longest single generation latency in the window, or ``0.0`` if none.

    TODO(latency_service): this aggregate has no repository function to call.
    It is a single ``MAX`` over the same ``llm_calls`` window every other figure
    on this page uses, added here rather than left as a hole in the response.
    """
    stmt = select(func.max(LLMCall.latency_ms)).where(
        LLMCall.created_at >= window.start, LLMCall.created_at <= window.end
    )
    stmt = apply_tenant_filter(stmt, window, LLMCall.application_id)
    return float((await session.execute(stmt)).scalar_one() or 0.0)


@router.get(
    "/token-efficiency",
    response_model=TokenEfficiencyReport,
    summary="Token-waste analysis",
)
async def token_efficiency(session: DbSession, scope: KnownScope) -> TokenEfficiencyReport:
    """Where the prompt budget went, from the figures recorded at ingest.

    Every number here is read back out of ``traces.metadata['token_efficiency']``
    rather than recomputed, so this page and the stored per-trace analysis can
    never disagree. A window in which nothing was analysed reports zeros: the
    schema types these fields as non-nullable, and the accompanying
    ``scored_traces``-derived findings line says plainly that no trace was
    measured rather than implying the system is efficient.
    """
    payload = await cached_call(
        "analytics",
        lambda: _build_token_efficiency(session, scope),
        **cache_parts(scope, "token-efficiency"),
    )
    return TokenEfficiencyReport.model_validate(payload)


async def _build_token_efficiency(
    session: DbSession, scope: AnalyticsScope
) -> dict[str, Any]:
    window = scope.window_filter()
    rollup = await repo.token_efficiency_rollup(session, window)
    offenders = await _top_offenders(session, window)
    waste_by_application = await _waste_by_application(session, window)
    scored = int(rollup.get("scored_traces") or 0)
    total_input = int((await repo.call_summary(session, window))["input_tokens"] or 0)

    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "score": float(rollup["score"] or 0.0),
        "potential_waste_pct": float(rollup["potential_waste_pct"] or 0.0),
        "wasted_tokens": int(rollup["wasted_tokens"] or 0),
        "total_input_tokens": total_input,
        "duplicate_document_count": int(rollup["duplicate_document_count"] or 0),
        "avg_context_share": float(rollup["avg_context_share"] or 0.0),
        "avg_output_yield": float(rollup["avg_output_yield"] or 0.0),
        "waste_by_application": waste_by_application,
        "top_offenders": offenders,
        "findings": _efficiency_findings(rollup, scored),
    }


async def _top_offenders(session: DbSession, window: WindowFilter, *, limit: int = 10) -> list:
    """The traces whose recorded waste was worst in this window.

    TODO(token_service): reads the same JSONB block as the rollup; it belongs
    with it. Scoped to traces that actually carry the analysis, so a trace
    ingested before efficiency analysis existed is not ranked as if it had been
    measured and found clean.
    """
    score = func.cast(func.jsonb_extract_path_text(Trace.extra_metadata, "token_efficiency", "score"), Float)
    waste = func.cast(
        func.jsonb_extract_path_text(Trace.extra_metadata, "token_efficiency", "potential_waste_pct"),
        Float,
    )
    stmt = (
        select(
            Trace.trace_id.label("trace_id"),
            waste.label("waste_pct"),
            score.label("score"),
            func.cast(
                func.jsonb_extract_path_text(
                    Trace.extra_metadata, "token_efficiency", "duplicate_document_count"
                ),
                Float,
            ).label("duplicate_document_count"),
            Trace.total_tokens.label("input_tokens"),
        )
        .where(
            Trace.start_time >= window.start,
            Trace.start_time <= window.end,
            score.is_not(None),
        )
        .order_by(waste.desc())
        .limit(limit)
    )
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    return [dict(row) for row in (await session.execute(stmt)).mappings().all()]


async def _waste_by_application(session: DbSession, window: WindowFilter) -> list[dict[str, Any]]:
    """Recorded waste grouped by application.

    TODO(token_service): as above. Grouped over traces that carry the analysis
    only, so the rows sum to the same population the headline figures average.
    """
    waste = func.cast(
        func.jsonb_extract_path_text(Trace.extra_metadata, "token_efficiency", "potential_waste_pct"),
        Float,
    )
    stmt = (
        select(
            Application.name.label("application"),
            func.avg(waste).label("avg_waste_pct"),
            func.count(Trace.id).label("trace_count"),
            func.coalesce(func.sum(Trace.total_tokens), 0).label("total_tokens"),
        )
        .select_from(Trace)
        .join(Application, Trace.application_id == Application.id)
        .where(
            Trace.start_time >= window.start,
            Trace.start_time <= window.end,
            waste.is_not(None),
        )
        .group_by(Application.name)
        .order_by(func.avg(waste).desc())
    )
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    return [dict(row) for row in (await session.execute(stmt)).mappings().all()]


def _efficiency_findings(rollup: dict[str, Any], scored_traces: int) -> list[str]:
    """Findings that state what was measured, and nothing that was not.

    The empty case is stated explicitly. An empty ``findings`` list on a
    dashboard whose numbers are all zero reads as "analysed, found nothing
    wrong", which is a claim the window cannot support if it held no analysed
    traces at all.
    """
    if not scored_traces:
        return [
            "Not analysed: no trace in this window recorded a token-efficiency "
            "analysis. Every figure below is an empty window, not a clean result."
        ]
    waste = float(rollup["potential_waste_pct"] or 0.0)
    duplicates = float(rollup["duplicate_document_count"] or 0.0)
    findings = [
        f"Measured across {scored_traces} analysed "
        f"trace{'s' if scored_traces != 1 else ''}: "
        f"{waste:.1f}% of input tokens were duplicated context or unread context."
    ]
    if duplicates:
        findings.append(f"Measured: {duplicates:.1f} retrieved documents per request were duplicates.")
    return findings


@router.get("/cost-quality", response_model=CostQualityReport, summary="Cost versus quality")
async def cost_quality(session: DbSession, scope: KnownScope) -> CostQualityReport:
    """One row per model: what it cost, and how well it answered.

    ``best_value`` and ``best_quality`` are computed only over models that
    actually have the relevant figure. Picking a winner from a set where most
    rows are unmeasured would report the least-bad of a measurement nobody took.
    """
    payload = await cached_call(
        "analytics",
        lambda: _build_cost_quality(session, scope),
        **cache_parts(scope, "cost-quality"),
    )
    return CostQualityReport.model_validate(payload)


async def _build_cost_quality(session: DbSession, scope: AnalyticsScope) -> dict[str, Any]:
    window = scope.window_filter()
    rows = await repo.model_comparison(session, window)
    quality = await _per_model_quality(session, window)

    models = [
        _comparison_row(row, quality.get(str(row["model_name"])))
        for row in rows
    ]
    return {
        "window": scope.window,
        "start": scope.start,
        "end": scope.end,
        "models": [model.model_dump() for model in models],
        "cost_label": _window_cost_provenance(rows),
        "quality_metric_definition": QUALITY_DEFINITION,
        "best_value": _best_value(models),
        "best_quality": _best_quality(models),
    }


def _comparison_row(row: dict[str, Any], scores: tuple | None) -> ModelComparison:
    """Project one repository row onto the scatter plot's row schema.

    ``scores`` is the ``(faithfulness, answer_relevance)`` pair for this model,
    or ``None`` when no evaluation of it was recorded in the window. Absent
    scores stay null all the way to the response: a model nobody evaluated has
    no quality, and reporting it as ``0.0`` would put a failure at the origin
    of a chart whose whole purpose is comparing quality.
    """
    faithfulness, answer_relevance = scores if scores else (None, None)
    return ModelComparison(
        model_name=str(row["model_name"]),
        provider=str(row["provider"]),
        call_count=int(row["call_count"] or 0),
        total_tokens=int(row["total_tokens"] or 0),
        avg_input_tokens=float(row["avg_input_tokens"] or 0.0),
        avg_output_tokens=float(row["avg_output_tokens"] or 0.0),
        avg_latency_ms=float(row["avg_latency_ms"] or 0.0),
        estimated_cost=float(row["estimated_cost"] or 0.0),
        # Null when the model recorded no tokens: the repository declines to
        # divide by a zero it did not receive.
        cost_per_1k_tokens=row["cost_per_1k_tokens"],
        cost_label=pricing_label(str(row["model_name"])),
        # Null, not False: an unregistered model's local/metered status was
        # never established, and claiming it was free is a claim nobody made.
        is_local=row["is_local"],
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        quality_score=_mean_or_none(faithfulness, answer_relevance),
    )


async def _per_model_quality(session: DbSession, window: WindowFilter) -> dict[str, tuple]:
    """``{model_name: (mean faithfulness, mean answer relevance)}`` for the window.

    The evaluations are joined through their trace to the model that answered
    them, which is the only recorded link between the two — an answer
    evaluation has no model of its own. The join takes the trace's *last* call
    rather than every one of them: a multi-call trace would otherwise credit its
    single evaluation to each model it happened to call, and the mean built
    from those credits would describe a set of calls that never happened. Rows
    whose trace recorded no model contribute to no model, rather than to an
    ``(unknown)`` bucket that would then be attributed to nobody.

    TODO(cost_service): this belongs with the cost/quality assembly rather than
    in a route. Every value in it is a SQL mean; nothing is computed in Python.
    """
    ranked_calls = (
        select(
            LLMCall.trace_id.label("trace_id"),
            LLMCall.model_name.label("model_name"),
            func.row_number()
            .over(partition_by=LLMCall.trace_id, order_by=LLMCall.created_at.desc())
            .label("call_rank"),
        ).subquery()
    )
    final_call = ranked_calls.c
    stmt = (
        select(
            final_call.model_name.label("model_name"),
            func.avg(AnswerEvaluation.faithfulness).label("faithfulness"),
            func.avg(AnswerEvaluation.answer_relevance).label("answer_relevance"),
        )
        .select_from(AnswerEvaluation)
        .join(Trace, AnswerEvaluation.trace_id == Trace.trace_id)
        .join(
            ranked_calls,
            and_(ranked_calls.c.trace_id == Trace.id, ranked_calls.c.call_rank == 1),
        )
        .where(Trace.start_time >= window.start, Trace.start_time <= window.end)
        .group_by(final_call.model_name)
    )
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    return {
        str(row["model_name"]): (row["faithfulness"], row["answer_relevance"])
        for row in (await session.execute(stmt)).mappings().all()
    }


def _mean_or_none(*values: float | None) -> float | None:
    """Mean of the values that exist, or ``None`` if none do.

    A window with faithfulness but no answer relevance still has a real mean of
    one measurement, reported as such — with a single input it equals that
    input, and the component it came from is visible in its own column.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _window_cost_provenance(rows: list[dict[str, Any]]) -> str:
    """Whether these figures are simulated list prices or genuinely free.

    Derived from the registry flag the repository already resolved rather than
    from the dollar amounts: a metered model that happened to cost nothing this
    window is still metered, and the label should say where the number came
    from rather than how big it is.
    """
    flags = {row["is_local"] for row in rows if row["is_local"] is not None}
    if not flags:
        return "No registered models in this window"
    if flags == {True}:
        return "Local inference — no API cost"
    if flags == {False}:
        return "Simulated list price"
    return "Mixed local and metered models"


def _best_value(models: list[ModelComparison]) -> str | None:
    """Cheapest model per token, among models that recorded any tokens.

    ``None`` when none did: ranking a cost-per-token from a zero-token row
    would be ranking a division by nothing.
    """
    priced = [model for model in models if model.cost_per_1k_tokens is not None]
    if not priced:
        return None
    return min(priced, key=lambda model: model.cost_per_1k_tokens).model_name


def _best_quality(models: list[ModelComparison]) -> str | None:
    """Highest-scoring model, among models that have a quality score.

    The mean is over recorded evaluation rows only, so an unmeasured model is
    not eligible — a chart that crowned the best of the measured as the best of
    the set would be reporting a subset as if it were the whole.
    """
    scored = [model for model in models if model.quality_score is not None]
    if not scored:
        return None
    return max(scored, key=lambda model: model.quality_score).model_name