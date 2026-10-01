"""Run retrieval evaluations over a labelled dataset.

A thin wrapper around :class:`~app.evaluation.retrieval_evaluator.RetrievalEvaluationRunner`.
The runner owns the index, the scoring, and the persistence; this route only
resolves which application and dataset the request names, and turns the runner's
domain exceptions into HTTP status codes.

The route exists in ``app/api/v1/`` rather than inside the runner because the
runner is also driven from scripts and tests — the HTTP concern stops here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession, TenantPrincipal, WriteGuard, resolve_application_or_404
from app.core.cache import invalidate
from app.core.logging import get_logger
from app.evaluation.retrieval_evaluator import (
    DatasetNotLabelledError,
    EvaluationDataset,
    LabelledQuery,
    RetrievalEvaluationRunner,
    RetrievalEvaluationSummary,
)
from app.models import utcnow
from app.schemas.evaluation import EvaluationRunDetail, RetrievalEvaluationRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])




@router.post(
    "/retrieval",
    response_model=EvaluationRunDetail,
    summary="Run a retrieval evaluation",
    dependencies=[WriteGuard],
)
async def run_retrieval_evaluation(
    session: DbSession,
    payload: RetrievalEvaluationRequest,
    principal: TenantPrincipal,
) -> EvaluationRunDetail:
    """Evaluate a labelled dataset at every requested k, and persist the run.

    Runs are additive: the metrics of this run are stored so a later
    ``/evaluations/compare`` can delta them against a baseline. Nothing is
    compared here — a single run has nothing to be better or worse than.
    """
    application_id = await resolve_application_or_404(session, payload.application, principal)
    dataset = _dataset_from_request(payload)
    runner = RetrievalEvaluationRunner(persist=payload.persist)

    try:
        summary = await runner.run(
            session=session,
            dataset=dataset,
            ks=payload.all_k,
            name=payload.name,
            application_id=application_id,
            config_override=payload.config_override or None,
            persist=payload.persist,
        )
    except (DatasetNotLabelledError, FileNotFoundError) as exc:
        # Both mean the request named a dataset that cannot produce metrics.
        # 422 rather than 500: nothing is broken, the input just cannot be run.
        logger.warning("evaluations.retrieval.rejected", reason=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    await invalidate("analytics")
    logger.info(
        "evaluations.retrieval.completed",
        run_id=summary.run_id,
        queries=summary.num_queries,
        k=summary.k,
        persisted=summary.persisted,
    )
    return EvaluationRunDetail(
        id=summary.run_id,
        name=summary.name,
        application_id=application_id,
        dataset_name=summary.dataset_name,
        evaluation_type="retrieval",
        status="completed" if summary.persisted else "not_persisted",
        k=summary.k,
        num_queries=summary.num_queries,
        metrics=summary.as_metric_rows(),
        config=summary.configuration,
        started_at=None,
        completed_at=None,
        duration_ms=summary.duration_ms,
        notes=_run_notes(summary),
        error=None,
        created_at=utcnow(),
        results=summary.per_query,
    )


def _dataset_from_request(payload: RetrievalEvaluationRequest):
    """The dataset to run, from the request body or from disk.

    Inline ``examples`` win over ``load_from_path``: a caller who sent rows has
    said what to evaluate, and silently substituting the server's file would run
    a different experiment than the one they asked for and label it with their
    dataset name.
    """
    if payload.examples:
        return _inline_dataset(payload)
    if payload.load_from_path:
        # None asks the runner for settings.evaluation_dataset_path, and it
        # raises FileNotFoundError with an actionable message when absent.
        return None
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            "Nothing to evaluate: supply `examples`, or set `load_from_path` to "
            "read the configured evaluation dataset."
        ),
    )


def _inline_dataset(payload: RetrievalEvaluationRequest) -> EvaluationDataset:
    """Wrap request-supplied examples in the runner's dataset type.

    TODO(evaluation_service): this conversion belongs beside the runner, which
    ships a file loader and an ``EvaluationDataset`` type but no adapter from
    the API's ``EvalExample`` rows — the shapes are close but not identical.

    ``reference_answer`` and ``metadata`` are accepted by ``EvalExample`` and
    are not carried here: ``LabelledQuery`` has nowhere to put them. They are
    not silently dropped in the response either — the request body is the
    caller's own record of what it sent.
    """
    queries = tuple(
        LabelledQuery(
            query=example.query,
            relevant=frozenset(example.relevant_documents),
            grades=dict(example.relevance_grades or {}),
            tags=tuple(example.tags),
        )
        for example in payload.examples or []
    )
    if not any(query.relevant for query in queries):
        # The runner raises this itself; raising here keeps the error identical
        # whether the rows came from the body or from disk.
        raise DatasetNotLabelledError(
            f"No labelled queries among the {len(queries)} supplied examples; "
            "every row needs at least one relevant document."
        )
    return EvaluationDataset(
        name=payload.dataset_name,
        path=None,
        queries=queries,
        skipped_rows=0,
        format_errors=0,
    )


def _run_notes(summary: RetrievalEvaluationSummary) -> str:
    """Provenance of the run, in words.

    Records what the run covered and what it could not score, so the numbers
    are read with their own denominator in view — a run where a third of the
    rows were ``no_answer`` is not directly comparable to one without them.
    """
    parts = [
        f"{summary.num_labelled} labelled queries",
        f"{summary.num_no_answer} no-answer queries excluded from scoring",
    ]
    if summary.num_unresolved:
        parts.append(f"{summary.num_unresolved} queries retrieved nothing")
    if not summary.persisted:
        parts.append("not persisted")
    return "; ".join(parts)