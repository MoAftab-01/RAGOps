"""Answer evaluation, evaluation-run history, and run-to-run comparison.

Two rules shape everything here.

*Judge numbers stay quarantined.* ``use_llm_judge`` is off by default, and a
judge that is disabled, unreachable, or returns nonsense yields ``None`` in the
``judge_*`` columns — never ``0.0``, and never a fallback into the
deterministic columns. The UI shows "not run"; it must never show "scored zero".

*Comparison reports recorded differences, not causes.* ``/evaluations/compare``
deltas the metrics two runs actually stored and lists the configuration keys
that actually differ. It says "``rerank`` changed from ``true`` to ``false``",
never "reranking was disabled, which is why recall dropped" — the second
sentence is a hypothesis, and a regression report that asserts hypotheses is a
report that will confidently be wrong. The arithmetic lives in
``app/evaluation/regression.py``; this route only loads the two runs.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import (
    DbSession,
    KnownScope,
    Pagination,
    TenantPrincipal,
    WriteGuard,
    not_found,
    paginate,
    resolve_application_or_404,
)
from app.core.logging import get_logger
from app.evaluation.answer_evaluator import (
    DETERMINISTIC_METHOD,
    LLMJudge,
    evaluate_answer_detailed,
)
# Aliased because the route handler below is also called `compare_runs`. Left
# unaliased it would shadow this import inside its own body, and the call that
# computes the comparison would recurse into the handler with two dicts --
# which fails on `_load_run`, so the endpoint 500s and the shadowing hides
# behind an unrelated-looking error. `make lint` catches it as F811; the
# endpoint test for a real comparison is what keeps it fixed.
from app.evaluation.regression import compare_runs as compare_recorded_runs
from app.ml.embeddings import get_embedder
from app.models import AnswerEvaluation, EvaluationResult, EvaluationRun, utcnow
from app.repositories.common import (
    apply_pagination,
    apply_tenant_filter,
    count_rows,
    tenant_predicates,
)
from app.schemas.common import Page
from app.schemas.evaluation import (
    AnswerEvaluationRequest,
    AnswerEvaluationResult,
    AnswerEvaluationSummary,
    EvaluationRunDetail,
    EvaluationRunSummary,
    PerQueryResult,
    RegressionReport,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


class CompareRequest(BaseModel):
    """Two recorded runs, and the size of a drop worth calling a regression.

    ``max_regression_pct`` is a threshold on the *recorded* relative change, not
    a tolerance the report adjusts to produce a verdict: setting it to 0 flags
    any movement at all, which is what you want when the baseline is a
    published number other people are relying on.
    """

    baseline_run_id: uuid.UUID
    candidate_run_id: uuid.UUID
    max_regression_pct: float = Field(default=5.0, ge=0.0)




# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------


@router.post(
    "/answer",
    response_model=AnswerEvaluationSummary,
    summary="Evaluate answers for faithfulness and relevance",
    dependencies=[WriteGuard],
)
async def evaluate_answers(
    session: DbSession,
    payload: AnswerEvaluationRequest,
    principal: TenantPrincipal,
) -> AnswerEvaluationSummary:
    """Score a batch of question/answer/context triples.

    Deterministic metrics are computed locally against the real embedder; the
    judge is consulted only when ``use_llm_judge`` is set, and its verdict
    lands in the ``judge_*`` fields alone.
    """
    application_id = await resolve_application_or_404(session, payload.application, principal)
    judge = LLMJudge(enabled=payload.use_llm_judge, model=payload.judge_model)
    embedder = get_embedder()

    results: list[AnswerEvaluationResult] = []
    run_id: uuid.UUID | None = None
    if payload.persist:
        # `.id`, not the row: `_close_run` takes an id and compares it to a
        # column, so handing it the ORM object makes SQLAlchemy try to inline a
        # whole model as a literal and the whole route 500s at the end of an
        # otherwise successful batch.
        run_id = (
            await _open_run(
                session,
                name=payload.name or "answer-evaluation",
                dataset_name="ad_hoc",
                evaluation_type="answer",
                application_id=application_id,
                num_items=len(payload.items),
            )
        ).id

    for item in payload.items:
        scores = await judge.judge(
            item.question,
            item.answer,
            item.context,
            enabled=payload.use_llm_judge,
        )
        report = evaluate_answer_detailed(
            item.question,
            item.answer,
            item.context,
            embedder=embedder,
            judge=scores,
        )
        result = report.to_schema()
        results.append(result)
        if run_id is not None:
            session.add(_answer_row(run_id, item, result))

    await _close_run(session, run_id)
    logger.info(
        "evaluations.answer.completed",
        run_id=str(run_id) if run_id else None,
        items=len(results),
        judge_requested=payload.use_llm_judge,
        judge_enabled=judge.enabled,
    )
    return _aggregate(results, judge)


def _answer_row(
    run_id: uuid.UUID,
    item: Any,
    result: AnswerEvaluationResult,
) -> AnswerEvaluation:
    """Persist one evaluated item, keeping judge columns in their own fields."""
    return AnswerEvaluation(
        run_id=run_id,
        trace_id=item.trace_id,
        question=item.question,
        answer=item.answer,
        context=list(item.context) or None,
        reference_answer=item.reference_answer,
        faithfulness=result.faithfulness,
        context_relevance=result.context_relevance,
        answer_relevance=result.answer_relevance,
        citation_coverage=result.citation_coverage,
        unsupported_claim_ratio=result.unsupported_claim_ratio,
        # Written only when the judge actually produced a number. A judge that
        # was off, unreachable, or unparseable leaves these null, which is what
        # "not run" looks like in the database as well as on the wire.
        judge_faithfulness=result.judge_faithfulness,
        judge_answer_relevance=result.judge_answer_relevance,
        judge_model=result.judge_model,
        claims=[claim.model_dump() for claim in result.claims] or None,
        duration_ms=result.duration_ms,
    )


def _aggregate(results: list[AnswerEvaluationResult], judge: LLMJudge) -> AnswerEvaluationSummary:
    """Mean the batch, keeping unmeasured judge columns null.

    A judge field is averaged only over the items that actually produced one.
    Averaging over the whole batch would let one unreachable item drag an
    otherwise-complete judge score toward zero — which is the one thing the
    ``judge_*`` quarantine exists to prevent.
    """
    judged = [result for result in results if result.judge_faithfulness is not None]
    judged_relevance = [result for result in results if result.judge_answer_relevance is not None]
    return AnswerEvaluationSummary(
        num_items=len(results),
        method=DETERMINISTIC_METHOD,
        faithfulness=_mean([r.faithfulness for r in results]),
        context_relevance=_mean([r.context_relevance for r in results]),
        answer_relevance=_mean([r.answer_relevance for r in results]),
        citation_coverage=_mean([r.citation_coverage for r in results]),
        unsupported_claim_ratio=_mean([r.unsupported_claim_ratio for r in results]),
        # ``_mean_or_none``, not ``_mean(...) or None``: a judge that scored
        # every item 0.0 produced a real measurement of zero, and `or` would
        # report that as "not run".
        judge_faithfulness=_mean_or_none([r.judge_faithfulness for r in judged]),
        judge_answer_relevance=_mean_or_none([r.judge_answer_relevance for r in judged_relevance]),
        judge_model=judge.model if judged else None,
        per_item=results,
    )


def _mean(values: list[float | None]) -> float:
    """Mean of the values that exist, or ``0.0`` when none do.

    Every list passed here comes from an item that was evaluated, so an empty
    list means the batch itself was empty and the response's ``num_items: 0``
    says so. Use :func:`_mean_or_none` for anything the schema types nullable.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return 0.0
    return sum(present) / len(present)


def _mean_or_none(values: list[float | None]) -> float | None:
    """Mean of the values that exist, or ``None`` when none were measured.

    Distinct from ``_mean`` returning ``0.0``: zero is a score, and a column
    that exists to say "not run" must never carry one.
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=Page[EvaluationRunSummary], summary="List evaluation runs")
async def list_runs(
    session: DbSession,
    scope: KnownScope,
    pagination: Pagination,
    evaluation_type: Annotated[
        str | None, Query(description="retrieval | answer")
    ] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    dataset_name: Annotated[str | None, Query()] = None,
) -> Page[EvaluationRunSummary]:
    """Evaluation runs, newest first, filtered by the same window as the rest.

    Ordered by ``created_at`` rather than by the run's own start time: a
    persisted run's ``created_at`` is when it was written, and a run whose
    ``started_at`` is null would otherwise sort unpredictably.
    """
    stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
    stmt = apply_tenant_filter(stmt, scope.window_filter(), EvaluationRun.application_id)
    if evaluation_type:
        stmt = stmt.where(EvaluationRun.evaluation_type == evaluation_type)
    if status_filter:
        stmt = stmt.where(EvaluationRun.status == status_filter)
    if dataset_name:
        stmt = stmt.where(EvaluationRun.dataset_name == dataset_name)
    return await paginate(session, stmt, pagination, schema=EvaluationRunSummary)


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationRunDetail,
    summary="One evaluation run with its per-query results",
)
async def get_run(
    session: DbSession, run_id: uuid.UUID, principal: TenantPrincipal
) -> EvaluationRunDetail:
    """A run and its recorded per-query rows.

    Answer-evaluation runs have no per-query rows in this table — their detail
    lives in ``answer_evaluations`` — so ``results`` comes back empty for them
    rather than being populated with a re-derivation.
    """
    run = await _load_run(session, run_id, principal)

    rows = (
        await session.execute(
            select(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
            .order_by(EvaluationResult.created_at)
        )
    ).scalars().all()
    summary = EvaluationRunSummary.model_validate(run, from_attributes=True)
    return EvaluationRunDetail(
        **summary.model_dump(), results=[_per_query(row) for row in rows]
    )


def _per_query(row: EvaluationResult) -> PerQueryResult:
    """Project a stored per-query row onto the response shape.

    Reads the metric blob rather than recomputing it: the stored values are what
    the run's own aggregate was computed from, so recomputing here could put the
    detail page and the headline number on the same run slightly out of step.
    """
    metrics = row.metrics or {}
    return PerQueryResult(
        query=row.query,
        k=row.k,
        precision=float(metrics.get("precision", 0.0)),
        recall=float(metrics.get("recall", 0.0)),
        f1=float(metrics.get("f1", 0.0)),
        reciprocal_rank=float(metrics.get("reciprocal_rank", 0.0)),
        ndcg=float(metrics.get("ndcg", 0.0)),
        hit=bool(metrics.get("hit", False)),
        retrieved_document_ids=list(row.retrieved_document_ids or []),
        relevant_document_ids=list(row.relevant_document_ids or []),
        missed_document_ids=list(metrics.get("missed_document_ids") or []),
        latency_ms=row.duration_ms,
    )


@router.get(
    "/runs/{run_id}/results",
    response_model=Page[PerQueryResult],
    summary="Per-query results of one run",
)
async def run_results(
    session: DbSession,
    run_id: uuid.UUID,
    pagination: Pagination,
    principal: TenantPrincipal,
) -> Page[PerQueryResult]:
    """Paginated per-query rows, for inspecting a large dataset query by query.

    Answer-evaluation runs store their detail in ``answer_evaluations`` rather
    than here, so this page comes back empty for them.
    """
    # The existence check goes through the scoped loader rather than its own
    # `select(...).where(id == ...)`: an unscoped check would confirm the run
    # exists to a tenant that is not allowed to read it, and then serve its
    # per-query rows.
    await _load_run(session, run_id, principal)

    stmt = (
        select(EvaluationResult)
        .where(EvaluationResult.run_id == run_id)
        .order_by(EvaluationResult.created_at)
    )
    total = int((await session.execute(count_rows(stmt))).scalar_one())
    rows = (
        await session.execute(apply_pagination(stmt, pagination.page, pagination.page_size))
    ).scalars().all()
    return Page[PerQueryResult].build(
        items=[_per_query(row) for row in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=RegressionReport,
    summary="Compare two runs",
    dependencies=[WriteGuard],
)
async def compare_runs(
    session: DbSession, payload: CompareRequest, principal: TenantPrincipal
) -> RegressionReport:
    """Delta two recorded runs and list the configuration keys that differ.

    Both runs must share a ``dataset_name``: metrics computed over different
    questions are not comparable, and reporting the difference between them as
    a regression would be arithmetic on unrelated quantities. That is a 422
    rather than a warning, because the caller almost certainly did not mean it.
    """
    if payload.baseline_run_id == payload.candidate_run_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="baseline_run_id and candidate_run_id are the same run; "
            "there is nothing to compare.",
        )
    baseline = await _load_run(session, payload.baseline_run_id, principal)
    candidate = await _load_run(session, payload.candidate_run_id, principal)
    if baseline.dataset_name != candidate.dataset_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Runs used different datasets ({baseline.dataset_name!r} vs "
                f"{candidate.dataset_name!r}), so their metrics are not comparable."
            ),
        )

    comparison = compare_recorded_runs(
        {
            "metrics": baseline.metrics or {},
            "config": baseline.config or {},
        },
        {
            "metrics": candidate.metrics or {},
            "config": candidate.config or {},
        },
        payload.max_regression_pct,
    )
    return RegressionReport(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        **comparison,
    )


async def _load_run(
    session: DbSession, run_id: uuid.UUID, principal: TenantPrincipal
) -> EvaluationRun:
    """Load one evaluation run, scoped to the caller's organization.

    Every run-by-id read goes through here — the detail endpoint, the per-query
    page, and the comparison endpoint's two baselines — because a run id is
    guessable enough that leaving any one of the three unscoped would hand out
    one of the others.

    ``evaluation_runs.application_id`` is nullable, so the predicate is a
    semi-join on the owning application's organization. For the platform
    principal it adds nothing at all and the compiled SQL is unchanged; for a
    tenant a run with no application is simply not found, which is the right
    answer because it belongs to no company.

    ``tenant_predicates`` rather than ``apply_tenant_filter``: this lookup has no
    time window at all, and a ``WindowFilter`` invented only to carry the
    organization id would be a fabricated range that reads as a real one.
    """
    stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
    stmt = stmt.where(
        *tenant_predicates(None, principal.organization_id, EvaluationRun.application_id)
    )
    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise await not_found(f"No evaluation run with id {run_id}.")
    return run


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


async def _open_run(
    session: DbSession,
    *,
    name: str,
    dataset_name: str,
    evaluation_type: str,
    application_id: uuid.UUID | None,
    num_items: int,
) -> EvaluationRun:
    """Insert a run row in ``running`` state and return it.

    Written before the work rather than after, so a process that dies mid-batch
    leaves a run that says it never finished instead of no run at all.
    """
    run = EvaluationRun(
        name=name,
        application_id=application_id,
        dataset_name=dataset_name,
        evaluation_type=evaluation_type,
        status="running",
        k=0,
        num_queries=num_items,
        started_at=utcnow(),
    )
    session.add(run)
    await session.flush()
    return run


async def _close_run(session: DbSession, run_id: uuid.UUID | None) -> None:
    """Mark a run completed, aggregating what was measured.

    The judge columns of the aggregate are left out entirely: a run whose items
    were never judged has no judge score, and writing ``None`` under a key that
    exists would read as a recorded zero.
    """
    if run_id is None:
        return
    # The flush is load-bearing. `SessionLocal` sets `autoflush=False`, so the
    # `_answer_row` objects the loop added are still only in the identity map --
    # a SELECT does not push them to the database, it queries around them. The
    # aggregate then computes over zero rows and stores `num_items: 0` with every
    # metric `0.0`, for a batch that was evaluated perfectly well. The individual
    # rows commit correctly, so the damage is invisible unless you compare the
    # run's headline numbers against its own detail page.
    #
    # `_open_run` above flushes for the same reason and is why the run row itself
    # exists at all when this runs; this is that call's counterpart at the other
    # end of the loop.
    await session.flush()
    rows = (
        await session.execute(select(AnswerEvaluation).where(AnswerEvaluation.run_id == run_id))
    ).scalars().all()
    run = (
        await session.execute(select(EvaluationRun).where(EvaluationRun.id == run_id))
    ).scalar_one()
    judged = [row for row in rows if row.judge_faithfulness is not None]
    run.status = "completed"
    run.completed_at = utcnow()
    run.duration_ms = (run.completed_at - run.started_at).total_seconds() * 1000.0
    run.metrics = {
        "faithfulness": _mean([row.faithfulness for row in rows]),
        "context_relevance": _mean([row.context_relevance for row in rows]),
        "answer_relevance": _mean([row.answer_relevance for row in rows]),
        "citation_coverage": _mean([row.citation_coverage for row in rows]),
        "unsupported_claim_ratio": _mean([row.unsupported_claim_ratio for row in rows]),
        "num_items": len(rows),
        "method": DETERMINISTIC_METHOD,
    }
    if judged:
        run.metrics["judge_faithfulness"] = _mean(
            [row.judge_faithfulness for row in judged]
        )
        run.metrics["judge_answer_relevance"] = _mean(
            [row.judge_answer_relevance for row in judged]
        )