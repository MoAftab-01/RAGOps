"""Window-scoped analytics as SQL, never as Python loops.

Every number the dashboard shows is produced here, and every one of them is an
aggregate evaluated by PostgreSQL. That is not a style preference: a 30-day
window over a busy instrumented application is millions of rows, and computing
a mean or a p95 by streaming those rows into Python would be both slow and —
more importantly — the wrong contract. ``percentile_cont`` in SQL is computed
over the *population*; ``numpy.percentile`` over a Python list is computed over
whatever the transport happened to carry, and a silent truncation there is a
fabricated number.

Three invariants hold throughout this module:

1. **The filter column is the indexed one.** Requests count on ``llm_calls``
   filter ``llm_calls.created_at`` (``ix_llm_calls_app_time``); trace counts on
   ``traces.start_time`` (``ix_traces_app_start``); retrieval on
   ``retrieval_calls.created_at`` (``ix_retrieval_app_time``). Filtering a
   token aggregate on ``traces.start_time`` instead would either miss calls made
   after the trace was stamped or force a join the planner cannot index.
2. **Empty is ``NULL``, never ``0``.** A chart must break its line over a
   window with no traffic; a zero would read as "this hour served nothing
   successfully" and quietly mislead. :func:`bucketize` therefore LEFT JOINs a
   ``generate_series`` skeleton so empty buckets exist as rows whose metrics are
   ``None`` — that shape is also what ``TimeSeriesPoint`` declares.
3. **No fabricated join.** A quality figure is only ever the mean of rows that
   a real evaluation wrote. When no evaluation has been run in the window the
   aggregate returns ``None`` and the UI says so.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Final, Sequence

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    Interval,
    Select,
    cast,
    func,
    literal,
    literal_column,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.logging import get_logger
from app.models import (
    AnswerEvaluation,
    Application,
    EvaluationRun,
    LLMCall,
    Model,
    RetrievalCall,
    Span,
    Trace,
    User,
)
from app.repositories.common import (
    WindowFilter,
    apply_tenant_filter,
    tenant_predicates,
    window_predicates,
)
from app.utils.pricing import resolve_window

logger = get_logger(__name__)

__all__ = [
    "BUCKET_UNITS",
    "answer_quality",
    "bucket_series",
    "bucketize",
    "call_summary",
    "error_rate",
    "grouped_by_application",
    "grouped_by_model",
    "grouped_by_span_kind",
    "grouped_by_user",
    "latency_summary",
    "model_comparison",
    "percentile_cont",
    "resolve_window_filter",
    "retrieval_score",
    "token_efficiency_rollup",
    "trace_summary",
]

# ---------------------------------------------------------------------------
# Small SQL building blocks
# ---------------------------------------------------------------------------

#: The two bucket widths the API contract promises. Anything else is a bug in
#: the caller, and is rejected rather than interpolated into SQL.
BUCKET_UNITS: Final[frozenset[str]] = frozenset({"hour", "day"})

_INTERVAL_PER_UNIT: Final[dict[str, str]] = {"hour": "1 hour", "day": "1 day"}

# Sources that can be merged onto a bucket skeleton. Names match the
# ``include`` argument of :func:`bucketize`.
BUCKET_SOURCES: Final[frozenset[str]] = frozenset({"calls", "traces", "retrieval", "answers"})


def _as_utc(value: datetime) -> datetime:
    """Normalise a bound to timezone-aware UTC.

    ``resolve_window`` hands back whatever the request supplied, and a naive
    ISO-8601 string is legal input. Comparing a naive value against a
    ``timestamptz`` column makes PostgreSQL fall back to the server's local
    zone, which would quietly shift a "7d" window by the server's offset.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def resolve_window_filter(
    window: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    application_id: uuid.UUID | None = None,
) -> WindowFilter:
    """Turn a request's window parameters into a :class:`WindowFilter`.

    The single entry point repositories use for scoping, so every query in this
    module agrees on what "7d" means. Thin, but deliberately the *only* place
    the string window is interpreted below this layer.
    """
    resolved_start, resolved_end, _label = resolve_window(window, start, end)
    return WindowFilter(
        start=_as_utc(resolved_start),
        end=_as_utc(resolved_end),
        application_id=application_id,
    )


def percentile_cont(column: ColumnElement[Any], fraction: float) -> ColumnElement[Any]:
    """``percentile_cont(fraction) WITHIN GROUP (ORDER BY column)``.

    The column is cast to double precision because PostgreSQL only offers
    ``percentile_cont`` for floats and doubles — asking for the ordered-set
    form over an ``integer`` column fails to resolve. Casting is exact for the
    token and latency magnitudes involved, so it changes no reported digit.
    """
    return func.percentile_cont(fraction).within_group(cast(column, Float))


def _date_trunc(unit: str, column: ColumnElement[Any]) -> ColumnElement[Any]:
    """``date_trunc('hour'|'day', column)`` with the unit as a SQL literal.

    ``func.date_trunc`` with a bound parameter leaves PostgreSQL unable to infer
    the type of the first argument in some resolution paths. The unit is
    validated against :data:`BUCKET_UNITS` first, so inlining it cannot carry
    user input into the statement.
    """
    if unit not in BUCKET_UNITS:
        raise ValueError(
            f"Unsupported bucket unit {unit!r}; expected one of {sorted(BUCKET_UNITS)}"
        )
    return func.date_trunc(literal_column(f"'{unit}'"), column)


def _trace_metadata_float(*path: str) -> ColumnElement[Any]:
    """Read a float out of ``traces.metadata`` JSONB, ``NULL`` when absent.

    Token-efficiency figures are computed once at ingest and stored in the
    trace's metadata (see ``app.utils.tokens.token_efficiency``). Reading them
    from JSONB rather than recomputing keeps the "recorded, not projected"
    guarantee; the ``#>>`` path cast is what makes a missing key ``NULL`` rather
    than a text-to-float error.
    """
    return func.cast(func.jsonb_extract_path_text(Trace.extra_metadata, *path), Float)


# ---------------------------------------------------------------------------
# Bucket skeleton
# ---------------------------------------------------------------------------


def bucket_series(start: datetime, end: datetime, unit: str = "day") -> Any:
    """A ``generate_series`` subquery with one row per bucket, no gaps.

    This is the LEFT JOIN anchor. Building the skeleton in SQL rather than in
    Python is what makes the empty-bucket guarantee hold for a 90-day daily
    window as cheaply as for a 1-hour hourly one — the alternative (fill gaps
    after fetching) needs the same rows in memory anyway.

    Both bounds are truncated to the bucket unit and then stepped, so the
    series always *starts* on a bucket boundary; a window that begins at
    10:37 therefore yields an 11:00 bucket and never a half-hour 10:00 bucket
    that would read as "no traffic".
    """
    if unit not in BUCKET_UNITS:
        raise ValueError(
            f"Unsupported bucket unit {unit!r}; expected one of {sorted(BUCKET_UNITS)}"
        )
    start_bound = func.cast(
        literal(start, type_=DateTime(timezone=True)), DateTime(timezone=True)
    )
    end_bound = func.cast(
        literal(end, type_=DateTime(timezone=True)), DateTime(timezone=True)
    )
    step = cast(literal(_INTERVAL_PER_UNIT[unit]), Interval)
    bucket = func.generate_series(
        _date_trunc(unit, start_bound), _date_trunc(unit, end_bound), step
    ).label("bucket")
    return select(bucket).subquery()


def _calls_aggregate(window: WindowFilter, unit: str) -> Any:
    """Per-bucket request/token/cost/latency aggregate over ``llm_calls``."""
    bucket = _date_trunc(unit, LLMCall.created_at)
    stmt = (
        select(
            bucket.label("bucket"),
            func.count(LLMCall.id).label("request_count"),
            func.count(LLMCall.id)
            .filter(LLMCall.status == "error")
            .label("error_count"),
            func.coalesce(func.sum(LLMCall.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(LLMCall.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(LLMCall.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(LLMCall.estimated_cost), 0.0).label("estimated_cost"),
            func.avg(LLMCall.latency_ms).label("avg_latency_ms"),
            percentile_cont(LLMCall.latency_ms, 0.5).label("p50_latency_ms"),
            percentile_cont(LLMCall.latency_ms, 0.95).label("p95_latency_ms"),
            percentile_cont(LLMCall.latency_ms, 0.99).label("p99_latency_ms"),
        )
        .where(LLMCall.created_at >= window.start, LLMCall.created_at <= window.end)
        .group_by(bucket)
    )
    stmt = apply_tenant_filter(stmt, window, LLMCall.application_id)
    return stmt.subquery()


def _traces_aggregate(window: WindowFilter, unit: str) -> Any:
    """Per-bucket trace count and error count over ``traces``.

    Errors are counted from the recorded ``error`` text rather than the
    ``status`` column so a trace that failed without an explicit status flip is
    still counted. Both are recorded facts — no inference.
    """
    bucket = _date_trunc(unit, Trace.start_time)
    stmt = (
        select(
            bucket.label("bucket"),
            func.count(Trace.id).label("trace_count"),
            func.count(Trace.id)
            .filter(Trace.error.is_not(None))
            .label("trace_error_count"),
        )
        .where(Trace.start_time >= window.start, Trace.start_time <= window.end)
        .group_by(bucket)
    )
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    return stmt.subquery()


def _retrieval_aggregate(window: WindowFilter, unit: str) -> Any:
    """Per-bucket mean retrieval score. ``NULL`` when nothing was retrieved."""
    bucket = _date_trunc(unit, RetrievalCall.created_at)
    stmt = (
        select(
            bucket.label("bucket"),
            func.avg(RetrievalCall.top_score).label("avg_retrieval_score"),
            func.count(RetrievalCall.id).label("retrieval_count"),
        )
        .where(
            RetrievalCall.created_at >= window.start,
            RetrievalCall.created_at <= window.end,
        )
        .group_by(bucket)
    )
    stmt = apply_tenant_filter(stmt, window, RetrievalCall.application_id)
    return stmt.subquery()


def _answers_aggregate(window: WindowFilter, unit: str) -> Any:
    """Per-bucket mean faithfulness / answer relevance from persisted rows.

    Scoped to the application through ``evaluation_runs`` because
    ``answer_evaluations`` has no direct ``application_id``. Rows whose run
    cannot be found are excluded when an application filter is active — a
    quality figure attributed to the wrong application is worse than none.
    """
    bucket = _date_trunc(unit, AnswerEvaluation.created_at)
    stmt = (
        select(
            bucket.label("bucket"),
            func.avg(AnswerEvaluation.faithfulness).label("avg_faithfulness"),
            func.avg(AnswerEvaluation.answer_relevance).label("avg_answer_relevance"),
            func.count(AnswerEvaluation.id).label("evaluation_count"),
        )
        .where(
            AnswerEvaluation.created_at >= window.start,
            AnswerEvaluation.created_at <= window.end,
        )
        .group_by(bucket)
    )
    # The tenant predicate has to go *inside* the subquery. `AnswerEvaluation`
    # has no `application_id` column of its own, so the organization is reached
    # through the run -- and the run's organization predicate is a narrowing of
    # the same subquery, not an extra WHERE on the outer statement. Putting it
    # outside would need a column the outer query does not have, and filtering
    # the outer statement on the bucket alone would let every tenant's
    # evaluations aggregate into one series.
    run_ids = select(EvaluationRun.id)
    stmt = stmt.where(
        AnswerEvaluation.run_id.in_(
            run_ids.where(
                *window_predicates(window, EvaluationRun.application_id)
            )
        )
    )
    return stmt.subquery()


def _optional_source(
    source: str, window: WindowFilter, unit: str
) -> tuple[Any, tuple[ColumnElement[Any], ...]] | None:
    """Build one aggregate subquery and its projected columns, or ``None``."""
    if source not in BUCKET_SOURCES:
        raise ValueError(
            f"Unknown bucket source {source!r}; expected one of {sorted(BUCKET_SOURCES)}"
        )
    if source == "calls":
        sub = _calls_aggregate(window, unit)
        columns = (
            sub.c.request_count,
            sub.c.error_count,
            sub.c.input_tokens,
            sub.c.output_tokens,
            sub.c.total_tokens,
            sub.c.estimated_cost,
            sub.c.avg_latency_ms,
            sub.c.p50_latency_ms,
            sub.c.p95_latency_ms,
            sub.c.p99_latency_ms,
        )
    elif source == "traces":
        sub = _traces_aggregate(window, unit)
        columns = (sub.c.trace_count, sub.c.trace_error_count)
    elif source == "retrieval":
        sub = _retrieval_aggregate(window, unit)
        columns = (sub.c.avg_retrieval_score, sub.c.retrieval_count)
    else:
        sub = _answers_aggregate(window, unit)
        columns = (
            sub.c.avg_faithfulness,
            sub.c.avg_answer_relevance,
            sub.c.evaluation_count,
        )
    return sub, columns


async def bucketize(
    session: AsyncSession,
    window: WindowFilter,
    *,
    unit: str = "day",
    include: Sequence[str] = ("calls", "traces", "retrieval", "answers"),
) -> list[dict[str, Any]]:
    """One dict per bucket across the window, gaps included with ``None``.

    ``unit`` is ``"hour"`` or ``"day"``; the API contract promises hourly
    buckets for ``1h``/``24h`` and daily beyond, which is the caller's job to
    decide. ``include`` selects which aggregates get merged onto the skeleton —
    the ``calls`` set is the one that populates ``request_count`` and the token
    columns of ``TimeSeriesPoint``, so leaving it out yields a series of
    ``bucket``-only rows, which is occasionally useful for axis rendering.

    Rows are returned as ``dict`` rather than ORM objects because these are
    aggregate rows with no identity: SQLAlchemy has nothing to map them onto,
    and the service layer writes them straight into ``TimeSeriesPoint``.
    """
    series = bucket_series(window.start, window.end, unit)
    columns: list[ColumnElement[Any]] = [series.c.bucket]

    # Every aggregate subquery is LEFT JOINed onto the *same* bucket skeleton,
    # so the FROM clause has to accumulate those joins rather than be rebuilt
    # per source. Re-selecting `series.join(...)` inside the loop re-introduced
    # the skeleton on every pass and PostgreSQL rejected the fourth one with
    # `table name "anon_1" specified more than once`, which broke every
    # time-series endpoint.
    from_clause: Any = series
    for source in include:
        built = _optional_source(source, window, unit)
        if built is None:  # pragma: no cover - defensive
            continue
        sub, source_columns = built
        columns.extend(source_columns)
        # LEFT JOIN is the whole point: a bucket with no traffic keeps its row
        # and reports NULL metrics rather than disappearing from the series.
        from_clause = from_clause.join(
            sub, series.c.bucket == sub.c.bucket, isouter=True
        )

    stmt: Select[Any] = (
        select(*columns).select_from(from_clause).order_by(series.c.bucket)
    )

    result = await session.execute(stmt)
    return [dict(row) for row in result.mappings().all()]


# ---------------------------------------------------------------------------
# Window totals
# ---------------------------------------------------------------------------


async def call_summary(session: AsyncSession, window: WindowFilter) -> dict[str, Any]:
    """Request/token/cost totals plus latency percentiles over ``llm_calls``.

    Averages are ``NULL`` (not ``0.0``) when the window holds no calls, so the
    dashboard can say "no traffic" instead of "0 ms average". Sums are
    coalesced to ``0`` because a count of zero is a real count.
    """
    stmt = select(
        func.count(LLMCall.id).label("total_calls"),
        func.count(LLMCall.id).filter(LLMCall.status == "error").label("error_count"),
        func.coalesce(func.sum(LLMCall.input_tokens), 0).label("input_tokens"),
        func.coalesce(func.sum(LLMCall.output_tokens), 0).label("output_tokens"),
        func.coalesce(func.sum(LLMCall.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(LLMCall.estimated_cost), 0.0).label("estimated_cost"),
        func.max(LLMCall.total_tokens).label("max_tokens"),
        func.avg(LLMCall.latency_ms).label("avg_latency_ms"),
        percentile_cont(LLMCall.latency_ms, 0.5).label("p50_latency_ms"),
        percentile_cont(LLMCall.latency_ms, 0.95).label("p95_latency_ms"),
        percentile_cont(LLMCall.latency_ms, 0.99).label("p99_latency_ms"),
        percentile_cont(LLMCall.total_tokens, 0.5).label("p50_tokens"),
        percentile_cont(LLMCall.total_tokens, 0.95).label("p95_tokens"),
        percentile_cont(LLMCall.total_tokens, 0.99).label("p99_tokens"),
    ).where(LLMCall.created_at >= window.start, LLMCall.created_at <= window.end)
    stmt = apply_tenant_filter(stmt, window, LLMCall.application_id)

    row = (await session.execute(stmt)).mappings().one()
    return dict(row)


async def trace_summary(session: AsyncSession, window: WindowFilter) -> dict[str, Any]:
    """Trace count, error rate and distinct users over ``traces.start_time``.

    ``error_count`` counts a trace whose ``error`` column is populated; that
    covers both an explicit ``status='error'`` and a trace that recorded a
    failure without a status flip, so it never under-reports a failure that is
    actually in the data.
    """
    stmt = select(
        func.count(Trace.id).label("total_traces"),
        func.count(Trace.id).filter(Trace.error.is_not(None)).label("error_count"),
        func.count(func.distinct(Trace.user_id)).label("unique_users"),
        func.coalesce(func.sum(Trace.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(Trace.estimated_cost), 0.0).label("estimated_cost"),
        func.coalesce(func.sum(Trace.context_tokens), 0).label("context_tokens"),
        func.avg(Trace.duration_ms).label("avg_duration_ms"),
        percentile_cont(Trace.duration_ms, 0.5).label("p50_duration_ms"),
        percentile_cont(Trace.duration_ms, 0.95).label("p95_duration_ms"),
        percentile_cont(Trace.duration_ms, 0.99).label("p99_duration_ms"),
        func.max(Trace.duration_ms).label("max_duration_ms"),
    ).where(Trace.start_time >= window.start, Trace.start_time <= window.end)
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)

    row = (await session.execute(stmt)).mappings().one()
    return dict(row)


async def latency_summary(session: AsyncSession, window: WindowFilter) -> dict[str, Any]:
    """Latency percentiles for generations, over the window's ``llm_calls``."""
    return await call_summary(session, window)


def error_rate(error_count: int | None, total: int | None) -> float:
    """Error fraction in ``[0.0, 1.0]``. ``0.0`` when the denominator is zero.

    A percentage is deliberately *not* produced here: a caller multiplying by
    100 a second time is the most common way a dashboard starts lying about
    its own error rate.
    """
    if not total:
        return 0.0
    return float(error_count or 0) / float(total)


async def retrieval_score(session: AsyncSession, window: WindowFilter) -> float | None:
    """Mean ``retrieval_calls.top_score`` in the window, or ``None``.

    ``None`` means *not measured*, which is different from ``0.0`` and is the
    difference between "retrieval is failing" and "no retrieval was recorded".
    """
    stmt = select(func.avg(RetrievalCall.top_score)).where(
        RetrievalCall.created_at >= window.start,
        RetrievalCall.created_at <= window.end,
    )
    stmt = apply_tenant_filter(stmt, window, RetrievalCall.application_id)
    return (await session.execute(stmt)).scalar_one()


async def answer_quality(
    session: AsyncSession, window: WindowFilter
) -> tuple[float | None, float | None]:
    """``(mean faithfulness, mean answer relevance)`` from persisted rows.

    Both are ``None`` when no answer evaluation ran in the window. The contract
    is explicit that these must never be computed on the fly for the
    dashboard: the evaluation run is the unit of record, and an on-the-fly
    figure would be a number no evaluation ever produced.
    """
    stmt = select(
        func.avg(AnswerEvaluation.faithfulness),
        func.avg(AnswerEvaluation.answer_relevance),
    ).where(
        AnswerEvaluation.created_at >= window.start,
        AnswerEvaluation.created_at <= window.end,
    )
    # See `_answers_aggregate`: the tenant predicate belongs inside the run
    # subquery, because `answer_evaluations` has no `application_id` column.
    stmt = stmt.where(
        AnswerEvaluation.run_id.in_(
            select(EvaluationRun.id).where(
                *window_predicates(window, EvaluationRun.application_id)
            )
        )
    )
    row = (await session.execute(stmt)).one()
    return row[0], row[1]


async def token_efficiency_rollup(
    session: AsyncSession, window: WindowFilter
) -> dict[str, Any]:
    """Mean of the token-efficiency scores *recorded at ingest*.

    Nothing is recomputed here: the per-trace figures live in
    ``traces.metadata['token_efficiency']`` and this is their mean. A trace
    ingested before efficiency analysis existed contributes ``NULL`` and is
    skipped, which is why the average is over recorded traces only.
    """
    score = _trace_metadata_float("token_efficiency", "score")
    waste = _trace_metadata_float("token_efficiency", "potential_waste_pct")
    wasted = _trace_metadata_float("token_efficiency", "wasted_tokens")
    duplicates = _trace_metadata_float("token_efficiency", "duplicate_document_count")
    context_share = _trace_metadata_float("token_efficiency", "context_share")
    output_yield = _trace_metadata_float("token_efficiency", "output_yield")

    stmt = select(
        func.avg(score).label("score"),
        func.avg(waste).label("potential_waste_pct"),
        func.avg(wasted).label("wasted_tokens"),
        func.avg(duplicates).label("duplicate_document_count"),
        func.avg(context_share).label("avg_context_share"),
        func.avg(output_yield).label("avg_output_yield"),
        func.count(score).label("scored_traces"),
    ).where(Trace.start_time >= window.start, Trace.start_time <= window.end)
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)

    row = dict((await session.execute(stmt)).mappings().one())
    # These two are *averages of per-trace counts*, and the average of a count is
    # a real quantity, not a count. Rounding it to an integer reports "0
    # duplicate documents per request" when the true mean is 0.32 -- a silent
    # "there is no waste here" conclusion drawn from rounding alone. Round to
    # two decimals instead, which keeps the figure readable without collapsing it
    # to zero.
    for key in ("wasted_tokens", "duplicate_document_count"):
        if row.get(key) is not None:
            row[key] = round(float(row[key]), 2)
    return row


# ---------------------------------------------------------------------------
# Grouped rollups
# ---------------------------------------------------------------------------


def _breakdown(
    key: ColumnElement[Any],
    label: ColumnElement[Any],
    count: ColumnElement[Any],
    *,
    input_tokens: ColumnElement[Any] | None = None,
    output_tokens: ColumnElement[Any] | None = None,
    total_tokens: ColumnElement[Any] | None = None,
    cost: ColumnElement[Any] | None = None,
    latency: ColumnElement[Any] | None = None,
    extra: dict[str, ColumnElement[Any]] | None = None,
) -> Select[Any]:
    """Project a ``GROUP BY`` into the ``BreakdownItem`` field names.

    Centralised so every breakdown on the site has byte-identical keys — a
    frontend that special-cases one chart to work around a differently-shaped
    response from another is a bug waiting to happen.
    """
    columns: list[ColumnElement[Any]] = [
        key.label("key"),
        label.label("label"),
        count.label("count"),
    ]
    if input_tokens is not None:
        columns.append(func.coalesce(input_tokens, 0).label("input_tokens"))
    if output_tokens is not None:
        columns.append(func.coalesce(output_tokens, 0).label("output_tokens"))
    if total_tokens is not None:
        columns.append(func.coalesce(total_tokens, 0).label("total_tokens"))
    if cost is not None:
        columns.append(func.coalesce(cost, 0.0).label("estimated_cost"))
    if latency is not None:
        columns.append(latency.label("avg_latency_ms"))
    for extra_key, extra_column in (extra or {}).items():
        columns.append(extra_column.label(extra_key))
    return select(*columns)


async def grouped_by_user(
    session: AsyncSession, window: WindowFilter, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Per-user token and cost totals.

    Grouped on the recorded ``user_external_id``. Traces with no user are
    grouped under the empty key rather than dropped, so a per-user table that
    silently omits anonymous traffic would be a different total than the
    headline number next to it.
    """
    user_key = func.coalesce(Trace.user_external_id, "(anonymous)")
    stmt = _breakdown(
        user_key,
        user_key,
        func.count(Trace.id),
        total_tokens=func.sum(Trace.total_tokens),
        cost=func.sum(Trace.estimated_cost),
        latency=func.avg(Trace.duration_ms),
        extra={"agent_name": func.max(Trace.agent_name)},
    ).where(Trace.start_time >= window.start, Trace.start_time <= window.end)
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    stmt = stmt.group_by(user_key).order_by(func.coalesce(func.sum(Trace.total_tokens), 0).desc())
    stmt = stmt.limit(limit)
    return await _finish_breakdown(session, stmt, extra_keys=("agent_name",))


async def grouped_by_application(
    session: AsyncSession, window: WindowFilter, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Per-application trace counts and tokens, via a join to ``applications``."""
    app_key = Application.name
    stmt = (
        _breakdown(
            app_key,
            app_key,
            func.count(Trace.id),
            total_tokens=func.sum(Trace.total_tokens),
            cost=func.sum(Trace.estimated_cost),
            latency=func.avg(Trace.duration_ms),
            extra={
                "error_count": func.count(Trace.id).filter(Trace.error.is_not(None)),
                "unique_users": func.count(func.distinct(Trace.user_id)),
            },
        )
        .join(Application, Trace.application_id == Application.id)
        .where(Trace.start_time >= window.start, Trace.start_time <= window.end)
    )
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    stmt = stmt.group_by(app_key).order_by(func.count(Trace.id).desc()).limit(limit)
    return await _finish_breakdown(session, stmt, extra_keys=("error_count", "unique_users"))


async def grouped_by_model(
    session: AsyncSession, window: WindowFilter, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Per-model call counts, tokens, cost and latency over ``llm_calls``."""
    model_key = LLMCall.model_name
    stmt = _breakdown(
        model_key,
        model_key,
        func.count(LLMCall.id),
        input_tokens=func.sum(LLMCall.input_tokens),
        output_tokens=func.sum(LLMCall.output_tokens),
        total_tokens=func.sum(LLMCall.total_tokens),
        cost=func.sum(LLMCall.estimated_cost),
        latency=func.avg(LLMCall.latency_ms),
        extra={
            "provider": func.max(LLMCall.provider),
            "error_count": func.count(LLMCall.id).filter(LLMCall.status == "error"),
            "p95_latency_ms": percentile_cont(LLMCall.latency_ms, 0.95),
        },
    ).where(LLMCall.created_at >= window.start, LLMCall.created_at <= window.end)
    stmt = apply_tenant_filter(stmt, window, LLMCall.application_id)
    stmt = stmt.group_by(model_key, LLMCall.provider).order_by(
        func.count(LLMCall.id).desc()
    )
    stmt = stmt.limit(limit)
    return await _finish_breakdown(
        session, stmt, extra_keys=("provider", "error_count", "p95_latency_ms")
    )


async def grouped_by_span_kind(
    session: AsyncSession, window: WindowFilter, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Per-stage latency over ``spans``, scoped through the parent trace.

    Spans carry no application or timestamp of their own for scoping, so the
    window is taken from ``traces.start_time`` — the column ``ix_spans_trace_start``
    and ``ix_traces_app_start`` are built for. ``call_count`` is the span count:
    it is the number of times the stage ran, which is what a stage breakdown
    is asking, and it is not multiplied by any join below.
    """
    kind_key = Span.kind
    stmt = (
        _breakdown(
            kind_key,
            kind_key,
            func.count(Span.id),
            total_tokens=func.coalesce(
                func.sum(cast(Span.attributes["token_count"].astext, Integer)), 0
            ),
            latency=func.avg(Span.duration_ms),
            extra={
                "error_count": func.count(Span.id).filter(Span.error.is_not(None)),
                "total_duration_ms": func.coalesce(func.sum(Span.duration_ms), 0.0),
                "p95_latency_ms": percentile_cont(Span.duration_ms, 0.95),
            },
        )
        .join(Trace, Span.trace_id == Trace.id)
        .where(Trace.start_time >= window.start, Trace.start_time <= window.end)
    )
    stmt = apply_tenant_filter(stmt, window, Trace.application_id)
    stmt = stmt.group_by(kind_key).order_by(func.avg(Span.duration_ms).desc()).limit(limit)
    return await _finish_breakdown(
        session, stmt, extra_keys=("error_count", "total_duration_ms", "p95_latency_ms")
    )


async def grouped_by_user_catalog(
    session: AsyncSession,
    application_id: uuid.UUID | None = None,
    *,
    organization_id: uuid.UUID | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """The user roster for an application, for filter dropdowns.

    Not window-scoped: a filter that hides users merely because they were not
    active in the selected window makes the list look complete when it is not.

    Takes an ``organization_id`` rather than a ``WindowFilter`` because it has
    no window to be scoped by, and it builds no timestamp predicate at all. It
    is nevertheless routed through the shared predicate builder so "unscoped"
    keeps its single meaning -- there is no path here by which omitting an
    organization silently means "every organization's users".
    """
    stmt = (
        select(
            User.id.label("id"),
            User.external_id.label("external_id"),
            User.display_name.label("display_name"),
            func.count(Trace.id).label("trace_count"),
            func.max(Trace.start_time).label("last_seen_at"),
        )
        .outerjoin(Trace, Trace.user_id == User.id)
        .group_by(User.id, User.external_id, User.display_name)
        .order_by(func.count(Trace.id).desc())
        .limit(limit)
    )
    stmt = stmt.where(
        *tenant_predicates(application_id, organization_id, User.application_id)
    )
    return [dict(row) for row in (await session.execute(stmt)).mappings().all()]


async def model_comparison(
    session: AsyncSession, window: WindowFilter
) -> list[dict[str, Any]]:
    """One row per model for the cost/quality scatter plot.

    ``is_local`` and the context window come from the ``models`` registry when
    the call's ``model_id`` resolves; a call whose model was never registered
    keeps its recorded ``provider`` and reports ``is_local`` as ``None`` so the
    chart can render it as unknown rather than assume it was free.
    """
    stmt = (
        select(
            LLMCall.model_name.label("model_name"),
            LLMCall.provider.label("provider"),
            func.count(LLMCall.id).label("call_count"),
            func.coalesce(func.sum(LLMCall.total_tokens), 0).label("total_tokens"),
            func.avg(LLMCall.input_tokens).label("avg_input_tokens"),
            func.avg(LLMCall.output_tokens).label("avg_output_tokens"),
            func.avg(LLMCall.latency_ms).label("avg_latency_ms"),
            func.coalesce(func.sum(LLMCall.estimated_cost), 0.0).label("estimated_cost"),
        )
        .where(LLMCall.created_at >= window.start, LLMCall.created_at <= window.end)
        .group_by(LLMCall.model_name, LLMCall.provider)
        .order_by(func.count(LLMCall.id).desc())
    )
    stmt = apply_tenant_filter(stmt, window, LLMCall.application_id)
    rows = [dict(row) for row in (await session.execute(stmt)).mappings().all()]

    registry = await _model_registry(session)
    for row in rows:
        entry = registry.get((row["provider"], row["model_name"]))
        row["is_local"] = entry["is_local"] if entry else None
        row["input_cost_per_1k"] = entry["input_cost_per_1k"] if entry else None
        row["output_cost_per_1k"] = entry["output_cost_per_1k"] if entry else None
        row["context_window"] = entry["context_window"] if entry else None
        # Per-1k cost is derived from this row's own totals rather than a second
        # aggregate over the same group. A separate SUM would be a different
        # number for no reason, and a second pass over the same rows is how the
        # two would quietly drift apart.
        tokens = int(row["total_tokens"] or 0)
        row["cost_per_1k_tokens"] = (
            round(float(row["estimated_cost"] or 0.0) * 1000.0 / tokens, 8) if tokens else None
        )
        row["retrieval_score"] = None
        row["faithfulness"] = None
        row["answer_relevance"] = None
        row["quality_score"] = None
    return rows


async def _model_registry(session: AsyncSession) -> dict[tuple[str, str], dict[str, Any]]:
    """``(provider, name) -> pricing row`` for every registered model.

    Read wholesale rather than per model: it is a table with one row per model
    in the deployment, and a lookup per scatter point would be a round trip per
    point.
    """
    result = await session.execute(
        select(
            Model.provider,
            Model.name,
            Model.is_local,
            Model.input_cost_per_1k,
            Model.output_cost_per_1k,
            Model.context_window,
        )
    )
    return {
        (row.provider, row.name): {
            "is_local": row.is_local,
            "input_cost_per_1k": row.input_cost_per_1k,
            "output_cost_per_1k": row.output_cost_per_1k,
            "context_window": row.context_window,
        }
        for row in result
    }


async def _finish_breakdown(
    session: AsyncSession, stmt: Select[Any], *, extra_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Run a breakdown and fold the ``extra`` labels into an ``extra`` dict.

    ``extra`` is a dict on ``BreakdownItem`` precisely so a dimension can carry
    its own fields without changing the shared shape; unpacking it here keeps
    every breakdown caller from repeating the same seven lines.
    """
    rows = (await session.execute(stmt)).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        out.append(
            {
                "key": item.pop("key"),
                "label": item.pop("label", None),
                "count": item.pop("count", 0),
                "total_tokens": item.pop("total_tokens", 0),
                "input_tokens": item.pop("input_tokens", 0),
                "output_tokens": item.pop("output_tokens", 0),
                "estimated_cost": item.pop("estimated_cost", 0.0),
                "avg_latency_ms": item.pop("avg_latency_ms", None),
                "extra": {key: item.pop(key, None) for key in extra_keys},
            }
        )
    return out

