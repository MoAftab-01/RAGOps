"""Trace list, detail, and the incremental telemetry write endpoints.

Reads are open so the dashboard works before anything is instrumented; every
write requires ``X-API-Key`` via :data:`~app.api.deps.AuthDependency`.

TODO(trace_repo): ``app/repositories/trace_repo.py`` does not exist, so the
list and detail statements are built here. They are plain filtered selects and
contain no metric arithmetic — every number on this page is a column that was
recorded at ingest. When the repository lands these move into it verbatim.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import Select, or_, select
from sqlalchemy.orm import selectinload

from app.api.deps import (
    DbSession,
    KnownScope,
    Pagination,
    TenantPrincipal,
    WriteGuard,
    not_found,
    paginate,
)
from app.core.cache import invalidate
from app.core.logging import get_logger
from app.models import Application, LLMCall, RetrievalCall, Trace
from app.repositories.common import (
    ApplicationOwnedByAnotherOrganization,
    apply_organization_filter,
    apply_tenant_filter,
    apply_window,
)
from app.schemas.common import MessageResponse, Page
from app.schemas.ingest import BatchIngest, GenerationIn, IngestResponse, RetrievalIn, TraceCreate
from app.schemas.trace import (
    LLMCallOut,
    RetrievedDocumentOut,
    RetrievalOut,
    SpanOut,
    TraceDetail,
    TraceSummary,
)
from app.telemetry.ingest_service import IngestError, get_ingest_service
from app.utils.pricing import calculate_cost

logger = get_logger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])

#: Contract: list rows carry a preview, not the full prompt. Shipping whole
#: inputs in a list response is how a dashboard turns into a data-exfiltration
#: endpoint; the detail endpoint is where a full prompt is available.
PREVIEW_CHARS = 200


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _trace_id_filter(stmt: Select[Any], trace_id: str) -> Select[Any]:
    """Match a path segment against either the internal UUID or the caller id.

    The contract accepts both spellings, and a client holding the wrong one
    should get the trace rather than a 404 that reads as "no such trace".
    """
    try:
        parsed = uuid.UUID(trace_id)
    except ValueError:
        return stmt.where(Trace.trace_id == trace_id)
    return stmt.where(or_(Trace.id == parsed, Trace.trace_id == trace_id))


async def _find_trace(
    session: DbSession, trace_id: str, principal: TenantPrincipal
) -> Trace:
    """Load one trace by either id spelling, with its children.

    Scoped through ``Application`` for the same reason :func:`_application_for`
    is: the trace's own ``application_id`` is nullable, and a tenant asking for
    an unattached trace should be refused rather than served it. The join is
    inner and un-annotated, so a trace with no application is simply not found
    -- which is the same answer the tenant would get for an id that does not
    exist, and therefore reveals nothing.
    """
    stmt = (
        _trace_id_filter(
            select(Trace)
            .join(Application, Trace.application_id == Application.id)
            .options(
                selectinload(Trace.spans),
                selectinload(Trace.llm_calls),
                selectinload(Trace.retrieval_calls).selectinload(RetrievalCall.documents),
                selectinload(Trace.application),
            )
            .order_by(Trace.start_time.desc()),
            trace_id,
        )
        .limit(1)
    )
    stmt = apply_organization_filter(
        stmt, principal.organization_id, Application.organization_id
    )
    trace = (await session.execute(stmt)).scalars().first()
    if trace is None:
        raise await not_found(f"No trace matching {trace_id!r}.")
    return trace


def _preview(text: str | None) -> str | None:
    if not text:
        return None
    clipped = text[:PREVIEW_CHARS]
    # A truncation marker only when something was actually cut, so a short
    # prompt is not labelled as if it had been clipped.
    return f"{clipped}..." if len(text) > PREVIEW_CHARS else clipped


def _to_summary(trace: Trace) -> TraceSummary:
    return TraceSummary(
        id=trace.id,
        trace_id=trace.trace_id,
        application_id=trace.application_id,
        application_name=trace.application.name if trace.application else None,
        user_external_id=trace.user_external_id,
        session_id=trace.session_id,
        kind=trace.kind,
        status=trace.status,
        name=trace.name,
        start_time=trace.start_time,
        end_time=trace.end_time,
        duration_ms=trace.duration_ms,
        total_tokens=trace.total_tokens,
        estimated_cost=trace.estimated_cost,
        context_tokens=trace.context_tokens,
        agent_name=trace.agent_name,
        agent_iterations=trace.agent_iterations,
        input_preview=_preview(trace.input_text),
        has_error=bool(trace.error) or trace.status == "error",
    )


@router.get("", response_model=Page[TraceSummary], summary="List traces")
async def list_traces(
    session: DbSession,
    scope: KnownScope,
    pagination: Pagination,
    status_filter: Annotated[
        str | None, Query(alias="status", description="success | error | running | cancelled")
    ] = None,
    kind: Annotated[str | None, Query(description="chat | rag | agent | evaluation")] = None,
    model: Annotated[str | None, Query(description="Filter by a model the trace called")] = None,
    user_id: Annotated[str | None, Query(description="External user id")] = None,
    has_error: Annotated[bool | None, Query(description="Only traces that failed")] = None,
    min_duration_ms: Annotated[float | None, Query(ge=0)] = None,
    max_duration_ms: Annotated[float | None, Query(ge=0)] = None,
    min_tokens: Annotated[int | None, Query(ge=0)] = None,
    search: Annotated[
        str | None, Query(description="Substring of the input, output, or trace id")
    ] = None,
) -> Page[TraceSummary]:
    """Filtered, paginated trace list.

    Filters compose with AND, and an unrecognised value yields an empty page
    rather than being ignored — a filter the UI thinks it applied but the
    server dropped is worse than one that visibly matched nothing.
    """
    stmt = (
        select(Trace)
        .join(Application, Trace.application_id == Application.id)
        .options(selectinload(Trace.application))
        .order_by(Trace.start_time.desc())
    )
    stmt = apply_window(stmt, Trace.start_time, scope.window_filter())
    stmt = apply_tenant_filter(stmt, scope.window_filter(), Trace.application_id)
    if status_filter:
        stmt = stmt.where(Trace.status == status_filter)
    if kind:
        stmt = stmt.where(Trace.kind == kind)
    if user_id:
        stmt = stmt.where(Trace.user_external_id == user_id)
    if has_error is not None:
        stmt = (
            stmt.where(Trace.error.is_not(None))
            if has_error
            else stmt.where(Trace.error.is_(None))
        )
    if min_duration_ms is not None:
        stmt = stmt.where(Trace.duration_ms >= min_duration_ms)
    if max_duration_ms is not None:
        stmt = stmt.where(Trace.duration_ms <= max_duration_ms)
    if min_tokens is not None:
        stmt = stmt.where(Trace.total_tokens >= min_tokens)
    if search:
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Trace.input_text.ilike(pattern),
                Trace.output_text.ilike(pattern),
                Trace.trace_id.ilike(pattern),
            )
        )
    if model:
        # A trace may call several models; "called this model" is an EXISTS,
        # not a join, or the trace repeats once per matching call.
        stmt = stmt.where(
            select(LLMCall.id)
            .where(LLMCall.trace_id == Trace.id, LLMCall.model_name == model)
            .exists()
        )

    return await paginate(session, stmt, pagination, schema=TraceSummary)


@router.get("/{trace_id}", response_model=TraceDetail, summary="One trace in full")
async def get_trace(
    session: DbSession, trace_id: str, principal: TenantPrincipal
) -> TraceDetail:
    """Full trace: input, output, spans, retrievals with documents, and calls.

    The retrieval analysis is the block the ingest service computed from the
    stored rows, passed through unchanged. It is not recomputed here — a figure
    recalculated on read would be a second, subtly different number from the
    one every efficiency trend was built on.

    This is the one endpoint that returns a company's actual prompts and model
    output in full, so it is scoped by the principal rather than left open.
    """
    trace = await _find_trace(session, trace_id, principal)
    metadata = trace.extra_metadata or {}

    return TraceDetail(
        **_summary_fields(trace),
        input_text=trace.input_text,
        output_text=trace.output_text,
        error=trace.error,
        input_tokens=sum(call.input_tokens for call in trace.llm_calls),
        output_tokens=sum(call.output_tokens for call in trace.llm_calls),
        tags=list(trace.tags) if trace.tags else None,
        metadata=metadata,
        created_at=trace.created_at,
        spans=[
            SpanOut(
                id=span.id,
                name=span.name,
                kind=span.kind,
                start_time=span.start_time,
                end_time=span.end_time,
                duration_ms=span.duration_ms,
                status=span.status,
                error=span.error,
                attributes=span.attributes,
            )
            for span in sorted(trace.spans, key=lambda item: item.start_time)
        ],
        retrievals=[
            RetrievalOut(
                id=retrieval.id,
                query=retrieval.query,
                retriever=retrieval.retriever,
                top_k=retrieval.top_k,
                num_results=retrieval.num_results,
                latency_ms=retrieval.latency_ms,
                context_tokens=retrieval.context_tokens,
                top_score=retrieval.top_score,
                configuration=retrieval.configuration,
                created_at=retrieval.created_at,
                documents=[
                    RetrievedDocumentOut(
                        id=document.id,
                        document_id=document.document_id,
                        title=document.title,
                        rank=document.rank,
                        final_score=document.final_score,
                        vector_score=document.vector_score,
                        rerank_score=document.rerank_score,
                        token_count=document.token_count,
                        content_preview=document.content_preview,
                    )
                    for document in sorted(retrieval.documents, key=lambda item: item.rank)
                ],
            )
            for retrieval in sorted(trace.retrieval_calls, key=lambda item: item.created_at)
        ],
        llm_calls=[
            LLMCallOut.model_validate(call, from_attributes=True)
            for call in sorted(trace.llm_calls, key=lambda item: item.created_at)
        ],
        retrieval_analysis=metadata.get("retrieval_analysis") or {},
    )


def _summary_fields(trace: Trace) -> dict[str, Any]:
    """The fields ``TraceDetail`` inherits from ``TraceSummary``.

    Pulled from the summary projection so the list and the detail page can
    never disagree about the same trace's headline numbers.
    """
    summary = _to_summary(trace)
    data = summary.model_dump()
    for dropped in ("input_preview", "has_error"):
        data.pop(dropped, None)
    data["user_id"] = trace.user_id
    return data


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


async def _invalidate_analytics() -> None:
    """Drop cached analytics after telemetry lands.

    Best-effort: a Redis outage must not fail an ingest that already committed.
    The stale read is a number from thirty seconds ago, which is a far smaller
    problem than dropping a client's telemetry on the floor.
    """
    await invalidate("analytics")


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create or update a trace",
    dependencies=[WriteGuard],
)
async def create_trace(
    session: DbSession,
    payload: TraceCreate,
    principal: TenantPrincipal,
) -> dict[str, str]:
    """Create a trace in ``running`` state, or update one that already exists.

    Idempotent on ``(application_id, trace_id)``, so a client that retries after
    a timeout re-sends rather than duplicating.

    The organization comes from the resolved principal, never from the body. A
    payload that could name its own organization would be a way to write into
    another company's application without holding any of its keys.
    """
    try:
        trace_id = await get_ingest_service().ingest_trace(
            session, payload, organization_id=principal.organization_id
        )
    except ApplicationOwnedByAnotherOrganization as exc:
        # 409, distinct from the 422 below. A malformed payload is worth
        # retrying after a fix; this one is a naming collision that will fail
        # identically forever, and saying so is the difference between a client
        # that corrects itself and one that retries until it gives up.
        logger.warning("traces.create.rejected", reason="application_name_taken")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except IngestError as exc:
        logger.warning("traces.create.rejected", reason=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    await _invalidate_analytics()
    logger.info("traces.create", trace_id=trace_id, application=payload.application)
    return {"trace_id": trace_id, "status": "created"}


@router.post(
    "/batch",
    response_model=IngestResponse,
    summary="Ingest a batch of traces",
    dependencies=[WriteGuard],
)
async def ingest_batch(
    session: DbSession, payload: BatchIngest, principal: TenantPrincipal
) -> IngestResponse:
    """Ingest up to 500 traces in one transaction.

    A malformed trace is reported in ``errors`` rather than failing the batch:
    an endpoint that 500s on one bad payload makes the client discard the
    healthy ones along with it. A name that belongs to another company is
    reported the same way -- per item, not for the batch.
    """
    result = await get_ingest_service().ingest_batch(
        session, payload.traces, organization_id=principal.organization_id
    )
    await _invalidate_analytics()
    logger.info(
        "traces.batch",
        accepted=result.accepted,
        errors=len(result.errors),
        requested=len(payload.traces),
    )
    return result




async def _application_for(
    session: DbSession, trace_id: str, principal: TenantPrincipal
) -> tuple[str, str]:
    """``(application_name, resolved_trace_id)`` for a path-supplied trace id.

    The ingest service is keyed on ``(application, trace_id)``, so an
    incremental write has to resolve the application from the stored row rather
    than take it from the request body — otherwise a client could append a
    generation to another application's trace.

    The stored ``trace_id`` is returned too, because the path segment may have
    been the internal UUID and the service needs the caller-visible string.

    The organization's applications are joined rather than the trace filtered by
    an ``application_id``, because the trace's own column is nullable: an
    unattached trace has no application to scope by and would leak to whichever
    company asked first. Joining from ``Application`` makes "belongs to this
    tenant" and "has an application at all" the same question, so the unattached
    rows drop out for the same reason in the same clause.
    """
    stmt = (
        _trace_id_filter(
            select(Trace.trace_id, Application.name)
            .join(Application, Trace.application_id == Application.id),
            trace_id,
        )
        .limit(1)
    )
    stmt = apply_organization_filter(
        stmt, principal.organization_id, Application.organization_id
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        # 404 rather than 403: a 403 would confirm the trace id exists, and
        # confirming that is the enumeration this endpoint must not offer. Three
        # write routes reach this, and for the platform principal it stays an
        # ordinary "no such trace".
        raise await not_found(f"No trace matching {trace_id!r}.")
    return str(row.name), str(row.trace_id)


async def _last_child_id(session: DbSession, trace_id: str, model, order_column) -> str | None:
    """Id of the most recent child row of ``model`` for a resolved trace.

    Read after the ingest so the response can name the row that was just
    written. Flushes first: the ingest service flushes its own writes, so the
    row is visible to this query, and reading it back is the only way to report
    an id the caller can use to correlate the stage.

    Returns ``None`` only when the write produced no row, which for a
    validated payload means the trace was found but nothing was appended.
    """
    trace_uuid = (
        await session.execute(
            select(Trace.id).where(Trace.trace_id == trace_id).limit(1)
        )
    ).scalar_one_or_none()
    if trace_uuid is None:
        return None
    return (
        await session.execute(
            select(model.id)
            .where(model.trace_id == trace_uuid)
            .order_by(order_column.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


@router.post(
    "/{trace_id}/retrieval",
    status_code=status.HTTP_201_CREATED,
    summary="Attach a retrieval stage and its documents",
    dependencies=[WriteGuard],
)
async def add_retrieval(
    session: DbSession,
    trace_id: str,
    payload: RetrievalIn,
    principal: TenantPrincipal,
) -> dict[str, Any]:
    """Add one retrieval stage to an existing trace.

    The response names the retrieval row so the client can correlate the stage,
    and reports the document count actually stored.
    """
    application, resolved = await _application_for(session, trace_id, principal)
    await get_ingest_service().ingest_trace(
        session,
        TraceCreate(trace_id=resolved, application=application, retrievals=[payload]),
        organization_id=principal.organization_id,
    )
    await _invalidate_analytics()
    retrieval_id = await _last_child_id(session, resolved, RetrievalCall, RetrievalCall.created_at)
    logger.info("traces.retrieval_added", trace_id=resolved, documents=len(payload.documents))
    return {
        "trace_id": resolved,
        "retrieval_id": str(retrieval_id) if retrieval_id else None,
        "num_documents": len(payload.documents),
    }


@router.post(
    "/{trace_id}/generation",
    status_code=status.HTTP_201_CREATED,
    summary="Attach a generation to a trace",
    dependencies=[WriteGuard],
)
async def add_generation(
    session: DbSession,
    trace_id: str,
    payload: GenerationIn,
    principal: TenantPrincipal,
) -> dict[str, Any]:
    """Add one LLM call to an existing trace.

    Token counts come from the request when supplied and are otherwise derived
    server-side by the deterministic counter — a model is never asked how many
    tokens it used. Cost is reported from the same pricing function the stored
    call used, so the response and the row cannot disagree.
    """
    application, resolved = await _application_for(session, trace_id, principal)
    await get_ingest_service().ingest_trace(
        session,
        TraceCreate(trace_id=resolved, application=application, generations=[payload]),
        organization_id=principal.organization_id,
    )
    await _invalidate_analytics()
    call_id = await _last_child_id(session, resolved, LLMCall, LLMCall.created_at)

    input_tokens = payload.input_tokens or 0
    output_tokens = payload.output_tokens or 0
    logger.info("traces.generation_added", trace_id=resolved, model=payload.model)
    return {
        "trace_id": resolved,
        "llm_call_id": str(call_id) if call_id else None,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": calculate_cost(input_tokens, output_tokens, payload.model),
    }


@router.post(
    "/{trace_id}/complete",
    summary="Close a trace and recompute its totals",
    dependencies=[WriteGuard],
)
async def complete_trace(
    session: DbSession,
    trace_id: str,
    payload: TraceCreate,
    principal: TenantPrincipal,
) -> MessageResponse:
    """Close a trace; denormalised totals are recomputed from its LLM calls.

    ``end_time`` is sent through only when the caller supplied one. Left null,
    ``resolve_timing`` keeps the stored timing rather than stamping the server's
    clock as the moment the request ended, which is a different fact.
    """
    application, resolved = await _application_for(session, trace_id, principal)
    await get_ingest_service().ingest_trace(
        session,
        TraceCreate(
            trace_id=resolved,
            application=application,
            status=payload.status,
            end_time=payload.end_time,
            duration_ms=payload.duration_ms,
            output_text=payload.output_text,
            error=payload.error,
            tags=payload.tags,
            metadata=payload.metadata,
        ),
        organization_id=principal.organization_id,
    )
    await _invalidate_analytics()

    totals = (
        await session.execute(
            _trace_id_filter(select(Trace.total_tokens, Trace.estimated_cost), resolved)
        )
    ).first()
    logger.info("traces.completed", trace_id=resolved, status=payload.status)
    return MessageResponse(
        message=f"Trace {resolved} is {payload.status}.",
        detail={
            "trace_id": resolved,
            "status": payload.status,
            "total_tokens": int(totals[0] or 0) if totals else None,
            "estimated_cost": float(totals[1] or 0.0) if totals else None,
        },
    )
