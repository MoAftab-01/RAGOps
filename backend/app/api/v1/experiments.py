"""Multi-arm experiments over evaluation runs.

An experiment is a claim under test plus one row per arm. Each arm records the
configuration it ran and, once evaluated, the ``run_id`` whose ``metrics`` blob
holds the result. That blob is the only measurement there is, so the deltas
reported here are differences between *recorded* runs and nothing else: no arm
is scored against an expected value, and no config difference is turned into a
claim about which setting caused a movement.

An arm with no ``run_id`` has not been evaluated. Its ``metrics`` and
``deltas_vs_baseline`` are empty rather than zero-filled, and a metric absent
from one run is skipped rather than treated as a value of zero — a variant that
was never measured must not be ranked alongside one that was.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import (
    DbSession,
    KnownScope,
    Pagination,
    TenantPrincipal,
    WriteGuard,
    not_found,
    resolve_application_or_404,
)
from app.core.logging import get_logger
from app.evaluation.metrics import metric_direction
from app.models import EvaluationRun, Experiment, ExperimentVariant
from app.repositories.common import (
    apply_pagination,
    apply_tenant_filter,
    count_rows,
    tenant_predicates,
)
from app.schemas.common import Page
from app.schemas.insights import (
    ExperimentCreate,
    ExperimentOut,
    ExperimentVariantOut,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])

#: Stored until the runs behind an experiment are compared. Set to completed
#: only by the arm whose runs have all been evaluated and compared.
DEFAULT_STATUS = "pending"


@router.get("", response_model=Page[ExperimentOut], summary="List experiments")
async def list_experiments(
    session: DbSession,
    scope: KnownScope,
    pagination: Pagination,
) -> Page[ExperimentOut]:
    """Experiments created in the window, newest first."""
    stmt = (
        select(Experiment)
        .options(selectinload(Experiment.variants))
        .where(Experiment.created_at >= scope.start, Experiment.created_at <= scope.end)
        .order_by(Experiment.created_at.desc())
    )
    stmt = apply_tenant_filter(stmt, scope.window_filter(), Experiment.application_id)

    total = int((await session.execute(count_rows(stmt))).scalar_one())
    rows = (
        await session.execute(apply_pagination(stmt, pagination.page, pagination.page_size))
    ).scalars().all()

    return Page[ExperimentOut].build(
        items=[await _to_out(session, experiment) for experiment in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/{experiment_id}", response_model=ExperimentOut, summary="Get one experiment")
async def get_experiment(
    session: DbSession, experiment_id: uuid.UUID, principal: TenantPrincipal
) -> ExperimentOut:
    """One experiment with every arm's metrics and deltas resolved."""
    experiment = await _load(session, experiment_id, principal)
    return await _to_out(session, experiment)


@router.post(
    "",
    response_model=ExperimentOut,
    summary="Create an experiment",
    status_code=status.HTTP_201_CREATED,
    dependencies=[WriteGuard],
)
async def create_experiment(
    session: DbSession, payload: ExperimentCreate, principal: TenantPrincipal
) -> ExperimentOut:
    """Register an experiment and its arms, pending evaluation.

    Creates records; it does not run anything. Evaluating an arm is
    ``POST /api/evaluations/retrieval``, and binding the resulting run back to an
    arm is done by setting that arm's ``run_id``. A creation endpoint that also
    ran evaluations would make a multi-arm experiment cost N retrievals before
    the caller had seen a single result, which is not what the body describes.

    ``baseline_variant`` names an arm that must exist. It is validated here
    rather than at comparison time, because a baseline that does not exist makes
    every arm's ``deltas_vs_baseline`` permanently empty and nothing would say
    why.
    """
    application_id = await resolve_application_or_404(session, payload.application, principal)

    names = [variant.name for variant in payload.variants]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Variant names must be unique within an experiment; repeated: "
                f"{', '.join(sorted(duplicates))}."
            ),
        )

    baseline = payload.baseline_variant
    if baseline is None:
        # The contract's example names a baseline explicitly; defaulting to the
        # first arm is a guess, so it is stated as one in the response instead.
        baseline = names[0]
    elif baseline not in names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"baseline_variant {baseline!r} is not one of the supplied "
                f"variants ({', '.join(names)})."
            ),
        )

    experiment = Experiment(
        name=payload.name,
        application_id=application_id,
        description=payload.description,
        hypothesis=payload.hypothesis,
        dataset_name=payload.dataset_name,
        status=DEFAULT_STATUS,
        baseline_run_id=None,
        variants=[
            ExperimentVariant(
                name=variant.name,
                config=variant.config,
                run_id=variant.run_id,
            )
            for variant in payload.variants
        ],
    )
    session.add(experiment)
    await session.flush()
    await session.refresh(experiment, ["variants"])

    logger.info(
        "experiments.created",
        experiment_id=str(experiment.id),
        application=payload.application,
        variants=names,
        baseline=baseline,
    )
    return await _to_out(session, experiment, baseline_name=baseline)


async def _load(
    session: DbSession, experiment_id: uuid.UUID, principal: TenantPrincipal
) -> Experiment:
    """Fetch one experiment with its arms, or 404.

    Scoped to the caller's organization, and 404 rather than 403: this endpoint
    hands over an experiment's hypothesis and every arm's recorded metrics, so
    confirming that a foreign experiment id exists is itself a disclosure.

    ``tenant_predicates`` rather than ``apply_tenant_filter``: this lookup has no
    time window, and a ``WindowFilter`` manufactured only to carry the
    organization id would be a fabricated range that reads as a real one.
    """
    stmt = (
        select(Experiment)
        .options(selectinload(Experiment.variants))
        .where(Experiment.id == experiment_id)
    )
    stmt = stmt.where(
        *tenant_predicates(None, principal.organization_id, Experiment.application_id)
    )
    experiment = (await session.execute(stmt)).scalar_one_or_none()
    if experiment is None:
        raise await not_found(f"No experiment with id {experiment_id}.")
    return experiment




async def _to_out(
    session: DbSession,
    experiment: Experiment,
    baseline_name: str | None = None,
) -> ExperimentOut:
    """Project an experiment, resolving each arm's recorded metrics.

    TODO(experiment_service): the run-loading and delta assembly belong beside
    the comparison logic in ``app/api/v1/answer_eval.py`` rather than in a
    router. The arithmetic here is the same recorded-metric subtraction, not a
    second opinion about what a metric means — :func:`metric_direction` stays the
    only source of "is bigger better", and a metric it does not recognise is
    reported with direction ``neutral`` instead of being ranked.
    """
    runs = await _variant_runs(session, experiment.variants)

    baseline_run = None
    if experiment.baseline_run_id is not None:
        baseline_run = runs.get(experiment.baseline_run_id)
    if baseline_run is None and baseline_name is not None:
        for variant in experiment.variants:
            if variant.name == baseline_name and variant.run_id is not None:
                baseline_run = runs.get(variant.run_id)
                break

    baseline_metrics = (baseline_run.metrics if baseline_run is not None else None) or {}

    variants = [
        _variant_out(variant, runs.get(variant.run_id), baseline_metrics)
        for variant in experiment.variants
    ]

    return ExperimentOut(
        id=experiment.id,
        name=experiment.name,
        application_id=experiment.application_id,
        description=experiment.description,
        hypothesis=experiment.hypothesis,
        status=experiment.status,
        dataset_name=experiment.dataset_name,
        baseline_run_id=experiment.baseline_run_id,
        winner_variant=experiment.winner_variant,
        completed_at=experiment.completed_at,
        created_at=experiment.created_at,
        variants=variants,
    )


async def _variant_runs(
    session: DbSession, variants: list[ExperimentVariant]
) -> dict[uuid.UUID, EvaluationRun]:
    """The evaluation runs the arms are bound to, in one round trip.

    A ``run_id`` pointing at a deleted run is simply missing from the result:
    the arm reports no metrics rather than the experiment 500ing, because
    ``on delete set null`` is the FK's own promise and an arm that lost its run
    is a state the schema explicitly permits.
    """
    run_ids = {variant.run_id for variant in variants if variant.run_id is not None}
    if not run_ids:
        return {}
    rows = (
        await session.execute(select(EvaluationRun).where(EvaluationRun.id.in_(run_ids)))
    ).scalars().all()
    return {run.id: run for run in rows}


def _variant_out(
    variant: ExperimentVariant, run: Any, baseline_metrics: dict[str, Any]
) -> ExperimentVariantOut:
    """One arm: its config, its recorded metrics, and its deltas.

    An unevaluated arm (``run_id`` null, or its run no longer present) reports
    empty metrics and no deltas. The arm is genuinely in the experiment; it just
    has no result yet, and that is a different thing from a result of zero.
    """
    metrics = dict(run.metrics) if run is not None and run.metrics else {}
    deltas = (
        _deltas(baseline_metrics, metrics)
        if run is not None and baseline_metrics
        else []
    )
    return ExperimentVariantOut(
        id=variant.id,
        name=variant.name,
        config=variant.config,
        run_id=variant.run_id,
        metrics=metrics or None,
        deltas_vs_baseline=deltas,
    )


def _deltas(baseline: dict[str, Any], variant: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-metric difference between a baseline run and one arm's run.

    Only metrics present in *both* runs are compared. A metric the baseline
    never recorded has no value to difference against, and treating its absence
    as zero would manufacture a movement of the full size of the arm's own
    score.
    """
    deltas: list[dict[str, Any]] = []
    for metric in sorted(set(baseline) & set(variant)):
        base_value = _numeric(baseline[metric])
        variant_value = _numeric(variant[metric])
        if base_value is None or variant_value is None:
            continue
        change = variant_value - base_value
        deltas.append(
            {
                "metric": metric,
                "baseline": base_value,
                "variant": variant_value,
                "change": change,
                "direction": metric_direction(metric),
            }
        )
    return deltas


def _numeric(value: Any) -> float | None:
    """A metric value as a float, or ``None`` if it is not a single number.

    Run ``metrics`` blobs are JSON, so a value may be a nested object, a list,
    a string, or a genuine null. None of those is a number to subtract, and
    coercing any of them to 0.0 would put a fake movement in the table.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
