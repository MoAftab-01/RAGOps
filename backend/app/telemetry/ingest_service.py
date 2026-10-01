"""The telemetry write path — every number the dashboard shows starts here.

A trace arrives as a :class:`~app.schemas.ingest.TraceCreate` payload from an
instrumented application. This module turns that payload into rows, and it is
the only place in the codebase that decides what a stored token count, a stored
cost, or a stored context size means. Three rules shape the design:

**Nothing is invented.** When a caller omits ``input_tokens`` or
``output_tokens`` the counts are *derived* from the recorded text with
:func:`app.utils.tokens.count_tokens` — a deterministic counter that needs no
model and returns the same answer on the SDK and on the server. A payload with
neither text nor counts contributes zero, which is a measurement rather than a
guess. No branch below produces a number the caller did not supply and the
counter did not compute.

**The trace and its children can never disagree.** Token totals and cost are
recomputed from the trace's own ``LLMCall`` rows with a SQL aggregation once
they are persisted, and ``context_tokens`` is recomputed from the
``retrieved_documents`` rows. A caller cannot post a trace total that contradicts
the generations underneath it, and re-sending a trace cannot leave a stale
denormalised total behind.

**Re-sending is not an error.** Instrumentation retries, SDK buffers replay
after a network blip, and both are normal. Ingest is idempotent on
``(application_id, trace_id)``: a second arrival updates the trace in place
rather than raising a duplicate-key error that would fail an entire batch.

Nothing here touches a connection at import time, and the pure helpers
(:func:`find_duplicate_document_ids`, :func:`derive_generation_tokens`,
:func:`resolve_timing`) are plain functions so the arithmetic can be exercised
without a database.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Final

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import Timer, get_logger, log_db_query
from app.models import (
    LLMCall,
    RetrievedDocument,
    RetrievalCall,
    Span,
    Trace,
    utcnow,
)
from app.repositories.common import get_or_create_application, get_or_create_user
from app.schemas.ingest import (
    GenerationIn,
    IngestResponse,
    RetrievedDocumentIn,
    TraceCreate,
)
from app.utils.pricing import calculate_cost
from app.utils.tokens import count_tokens, estimate_tokens, token_efficiency

logger = get_logger(__name__)

__all__ = [
    "IngestError",
    "IngestService",
    "IngestValidationError",
    "derive_generation_tokens",
    "document_token_count",
    "find_duplicate_document_ids",
    "get_ingest_service",
    "normalise_title",
    "reset_ingest_service",
    "resolve_timing",
]

# A trace in one of these states has finished. Only a finished trace gets an
# end_time derived from a reported duration, because a duration on a trace that
# is still running measures nothing.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"success", "error", "cancelled"})

# Metadata keys this service owns. A caller-supplied value under either key is
# dropped in favour of the measured one, so the dashboard can never display a
# number that no code path computed.
COMPUTED_METADATA_KEYS: Final[tuple[str, ...]] = (
    "token_efficiency",
    "retrieval_analysis",
)

# Below this many characters a title or content fingerprint is too short to be
# evidence of a duplicate.
_MIN_FINGERPRINT_CHARS: Final = 3

# How much retrieved text is kept per document. The token count is taken over
# the text as sent, so trimming the stored copy never changes a measured number;
# storing whole passages instead would turn the write path into a document store.
_PREVIEW_CHARS: Final = 2000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IngestError(Exception):
    """Base class for ingest failures a caller can act on."""


class IngestValidationError(IngestError):
    """The payload cannot be written as given (no application name)."""


# ---------------------------------------------------------------------------
# Pure helpers — no I/O, unit-testable without a database
# ---------------------------------------------------------------------------


def _normalise(text: str | None) -> str:
    """Case-fold and collapse whitespace so equivalent text fingerprints match."""
    if not text:
        return ""
    return " ".join(text.split()).casefold()


def normalise_title(title: str | None) -> str:
    """Fingerprint for a document title.

    Markdown heading markers are stripped because titles are derived from the
    first level-one heading of a knowledge-base file, and ``# Getting Started``
    and ``Getting Started`` are the same title arriving by two code paths.
    Counting them as distinct documents would hide a genuine duplicate.
    """
    if not title:
        return ""
    return _normalise(title.lstrip("#").strip())


def _document_text(document: RetrievedDocumentIn) -> str:
    """The most complete text the caller sent for a document."""
    return document.content or document.content_preview or ""


def document_token_count(document: RetrievedDocumentIn) -> int:
    """Token count of a retrieved document.

    ``RetrievedDocumentIn`` carries no count of its own, so the count is always
    measured here from the text the caller sent, preferring the full content
    over the preview. A document sent with neither contributes zero, which is
    what it costs to store.
    """
    return estimate_tokens(_document_text(document))


def find_duplicate_document_ids(
    documents: Sequence[RetrievedDocumentIn],
) -> set[str]:
    """Document ids a retrieval paid for more than once.

    Three ways a retriever returns the same thing twice, all measured from the
    payload rather than inferred:

    1. the same ``document_id`` at more than one rank (or twice at the same
       rank, which is a client bug and still waste);
    2. two different ids sharing a title;
    3. two entries with identical content.

    Cases 2 and 3 matter because chunked corpora routinely re-emit the same
    passage under a different id — an overlapping chunk, a mirrored file, a
    re-chunk with a new offset — and a retriever that returns both has paid to
    send the same text twice.
    """
    return _duplicates_from_columns(
        [document.document_id for document in documents],
        [document.title for document in documents],
        [_document_text(document) for document in documents],
    )


def derive_generation_tokens(generation: GenerationIn) -> tuple[int, int]:
    """``(input_tokens, output_tokens)`` for one generation.

    Caller-reported counts win: Ollama reports exact ``prompt_eval_count`` and
    ``eval_count``, and an exact measurement always beats a derived one. A count
    is measured from the recorded text only when the caller omitted it. A
    payload with no text yields zero — the honest answer for text that was never
    sent.
    """
    input_tokens = generation.input_tokens
    if input_tokens is None:
        input_tokens = count_tokens(generation.prompt, generation.model)

    output_tokens = generation.output_tokens
    if output_tokens is None:
        output_tokens = count_tokens(generation.completion, generation.model)

    return max(0, input_tokens), max(0, output_tokens)


def _elapsed_ms(start_time: datetime, end_time: datetime) -> float:
    """Milliseconds between two instants, floored at zero.

    A negative span is clock skew between the client and the server, not a
    measurement, and a negative latency would corrupt every percentile it fed.
    """
    return round(max(0.0, (end_time - start_time).total_seconds() * 1000.0), 3)


def resolve_timing(
    *,
    start_time: datetime,
    end_time: datetime | None,
    duration_ms: float | None,
    terminal: bool,
) -> tuple[datetime | None, float | None]:
    """Settle ``end_time`` and ``duration_ms`` from whatever the caller sent.

    The two are redundant, so one fills in for the other. An end time always
    implies a duration. A duration implies an end time only for a trace that has
    finished — deriving one for a running trace would stamp a completion time on
    a request that has not completed. When neither is known the trace keeps a
    null duration rather than being given the current time, because the moment
    a server logs a row is not the moment the user's request ended.
    """
    if end_time is not None and duration_ms is None:
        return end_time, _elapsed_ms(start_time, end_time)
    if end_time is None and duration_ms is not None and terminal:
        return start_time + timedelta(milliseconds=duration_ms), duration_ms
    return end_time, duration_ms


# ---------------------------------------------------------------------------
# Ingest service
# ---------------------------------------------------------------------------


class IngestService:
    """Writes trace payloads into the telemetry tables.

    Stateless: every method takes the session it writes through and never
    commits, so the caller owns the transaction boundary. That is what lets the
    collector isolate one failing trace from the rest of a batch with a
    savepoint.
    """

    def __init__(self, *, preview_chars: int = _PREVIEW_CHARS) -> None:
        self._preview_chars = preview_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def ingest_trace(
        self,
        session: AsyncSession,
        payload: TraceCreate,
        *,
        application_name: str | None = None,
        organization_id: uuid.UUID | None = None,
    ) -> str:
        """Persist one trace and return its caller-visible ``trace_id``.

        ``application_name`` overrides ``payload.application`` so a route that
        has already resolved the application cannot be steered to a different
        one by the request body.

        ``organization_id`` is the tenant the write is attributed to, and it is
        resolved from the *principal* rather than the payload for the same
        reason: a body that could name its own organization could claim another
        one's telemetry. It stamps a newly created application, and it makes a
        name that belongs to somebody else an error instead of a silent hit.
        ``None`` is the platform principal, which is the local demo and the
        shared key.
        """
        name = (application_name or payload.application or "").strip()
        if not name:
            raise IngestValidationError(
                "A trace needs an application name, either on the payload or as an argument."
            )

        application = await get_or_create_application(
            session, name, organization_id=organization_id
        )
        user = await self._resolve_user(session, application, payload.user_id)

        trace_id = (payload.trace_id or "").strip() or uuid.uuid4().hex
        trace = await self._find_trace(session, application.id, trace_id)
        created = trace is None
        if trace is None:
            # ``start_time`` is non-nullable, and a payload that omits it still
            # has to be stored — the ingest moment is a fact about the request
            # only in the sense that it is the earliest point the server can
            # vouch for, so it is recorded as the start rather than discarded.
            trace = Trace(
                trace_id=trace_id,
                application_id=application.id,
                start_time=payload.start_time or utcnow(),
            )
            session.add(trace)
        else:
            # The pre-read returned a populated instance whose ``spans``,
            # ``llm_calls`` and ``retrieval_calls`` collections may already be
            # loaded. A partial re-send leaves those collections alone, and a
            # loaded collection would otherwise be written back on flush,
            # resurrecting the rows the replacement delete just removed.
            session.expire(trace, ["spans", "llm_calls", "retrieval_calls"])

        self._apply_trace_fields(trace, payload, user=user, created=created)
        await session.flush()

        await self._persist_spans(session, trace, payload)
        await self._persist_retrievals(session, trace, application.id, payload)
        await self._persist_generations(session, trace, application.id, payload)

        # Last, from the rows just written, so the denormalised trace totals are
        # a function of its children and can never drift from them.
        await self._recompute_totals(session, trace, payload=payload)
        await session.flush()

        logger.info(
            "telemetry.trace_ingested",
            trace_id=trace_id,
            application_id=str(application.id),
            application_name=name,
            created=created,
            spans=len(payload.spans),
            retrievals=len(payload.retrievals),
            generations=len(payload.generations),
            total_tokens=trace.total_tokens,
            context_tokens=trace.context_tokens,
            estimated_cost=trace.estimated_cost,
        )
        return trace_id

    async def ingest_batch(
        self,
        session: AsyncSession,
        payloads: Sequence[TraceCreate],
        *,
        organization_id: uuid.UUID | None = None,
    ) -> IngestResponse:
        """Ingest several payloads into one session, reporting per-item outcome.

        A malformed trace is recorded in ``errors`` rather than aborting the
        batch: an ingest endpoint that 500s on one bad payload would make the
        client throw away the healthy ones along with it.

        ``organization_id`` is the writing tenant and is threaded into every
        item. A name collision with another company therefore lands in ``errors``
        for that one item, and the rest of the batch is still stored -- which is
        the right outcome: the batch's other traces are good data, and the
        colliding one is a naming mistake rather than a server fault.
        """
        accepted = 0
        trace_ids: list[str] = []
        errors: list[str] = []

        for payload in payloads:
            identifier = payload.trace_id or "-"
            try:
                trace_ids.append(
                    await self.ingest_trace(
                        session, payload, organization_id=organization_id
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad item must not fail the batch
                errors.append(f"{identifier}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "telemetry.trace_rejected",
                    trace_id=identifier,
                    application=payload.application,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                accepted += 1

        return IngestResponse(accepted=accepted, trace_ids=trace_ids, errors=errors)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    async def _resolve_user(
        self, session: AsyncSession, application: Any, external_id: str | None
    ) -> Any | None:
        """Resolve the end user, creating the row on first sight.

        Goes through the repository rather than writing a row inline so first
        sighting is race-safe: two collectors meeting ``user-42`` at the same
        moment must not collide on the unique constraint.
        """
        if external_id is None:
            return None
        external_id = external_id.strip()
        if not external_id:
            return None
        return await get_or_create_user(session, application.id, external_id)

    async def _find_trace(
        self, session: AsyncSession, application_id: uuid.UUID, trace_id: str
    ) -> Trace | None:
        """Look up an existing trace on the idempotency key.

        This read is the fast path for a re-send. The unique constraint on
        ``(application_id, trace_id)`` remains the backstop: a genuinely
        concurrent duplicate cannot be caught by any pre-read, and the
        collector's savepoint handles that.
        """
        result = await session.execute(
            select(Trace).where(
                Trace.application_id == application_id, Trace.trace_id == trace_id
            )
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Trace row
    # ------------------------------------------------------------------
    def _apply_trace_fields(
        self,
        trace: Trace,
        payload: TraceCreate,
        *,
        user: Any | None,
        created: bool,
    ) -> None:
        """Copy the payload's scalar fields onto ``trace``.

        On an update a field is only overwritten when the caller actually sent
        it — ``model_fields_set`` separates "sent as null" from "left at the
        schema default". Without that check a partial completion call (status
        and duration only) would drag a finished trace back to the schema's
        default ``running`` status, because ``running`` is what the schema
        fills in when the caller says nothing at all.
        """
        sent = payload.model_fields_set
        terminal = payload.status in TERMINAL_STATUSES

        if created or payload.start_time is not None:
            trace.start_time = payload.start_time or trace.start_time

        if user is not None:
            trace.user_id = user.id
            trace.user_external_id = user.external_id

        # On a re-send that omitted the timing, keep whatever is already stored:
        # the child collections below are left alone in that case too, so the
        # trace keeps a consistent account of itself.
        end_time = payload.end_time
        if end_time is None and not created:
            end_time = trace.end_time
        duration_ms = payload.duration_ms
        if duration_ms is None and not created:
            duration_ms = trace.duration_ms

        end_time, duration_ms = resolve_timing(
            start_time=trace.start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            terminal=terminal,
        )
        trace.end_time = end_time
        trace.duration_ms = duration_ms

        if created or "status" in sent:
            trace.status = payload.status
        if created or "session_id" in sent:
            trace.session_id = payload.session_id
        if created or "name" in sent:
            trace.name = payload.name
        if created or "input_text" in sent:
            trace.input_text = payload.input_text
        if created or "output_text" in sent:
            trace.output_text = payload.output_text
        if created or "error" in sent:
            trace.error = payload.error
        if created or "agent_name" in sent:
            trace.agent_name = payload.agent_name
        if created or "agent_iterations" in sent:
            trace.agent_iterations = payload.agent_iterations
        if created or "tags" in sent:
            # NULL means untagged. An empty list would read as "tagged with
            # nothing", which is not a thing.
            trace.tags = list(payload.tags) if payload.tags else None
        if created or "metadata" in sent:
            self._merge_metadata(trace, payload.metadata)
        if created:
            # The kind is a routing dimension the analytics layer groups on and
            # is never restated by a completion call, so it is set once.
            trace.kind = payload.kind

    def _merge_metadata(self, trace: Trace, incoming: dict[str, Any]) -> None:
        """Merge caller metadata under the keys this service owns.

        ``token_efficiency`` and ``retrieval_analysis`` are computed from
        measured values after the children are written, so an incoming value
        under either key is replaced rather than trusted — a client-reported
        efficiency score would otherwise become a dashboard metric nobody
        computed.
        """
        metadata: dict[str, Any] = dict(trace.extra_metadata or {})
        metadata.update(incoming)
        for owned in COMPUTED_METADATA_KEYS:
            metadata.pop(owned, None)
        if any(owned in incoming for owned in COMPUTED_METADATA_KEYS):
            logger.debug(
                "telemetry.metadata_computed_key_overridden",
                trace_id=trace.trace_id,
            )
        trace.extra_metadata = metadata

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------
    async def _persist_spans(
        self, session: AsyncSession, trace: Trace, payload: TraceCreate
    ) -> None:
        """Write the trace's spans.

        A payload that carries spans replaces the previous set: the payload is
        the caller's full account of the trace, and keeping the old set would
        double every span in the timeline. A payload with no spans leaves
        existing ones alone, because the incremental endpoints add one stage at
        a time.
        """
        if not payload.spans:
            return

        await self._delete_children(session, Span, trace.id)

        for span_in in payload.spans:
            start_time = span_in.start_time or trace.start_time
            end_time, duration_ms = resolve_timing(
                start_time=start_time,
                end_time=span_in.end_time,
                duration_ms=span_in.duration_ms,
                terminal=span_in.status in TERMINAL_STATUSES,
            )
            session.add(
                Span(
                    # Assigned here rather than left to the column default so the
                    # id exists before the flush: a generation may reference a
                    # span by id, and a default-assigned id is not readable
                    # until after the flush.
                    id=uuid.uuid4(),
                    trace_id=trace.id,
                    name=span_in.name,
                    kind=span_in.kind,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    status=span_in.status,
                    error=span_in.error,
                    attributes=span_in.attributes or None,
                )
            )

    async def _persist_retrievals(
        self,
        session: AsyncSession,
        trace: Trace,
        application_id: uuid.UUID,
        payload: TraceCreate,
    ) -> None:
        """Write retrieval stages with their ranked documents."""
        if not payload.retrievals:
            return

        await self._delete_children(session, RetrievalCall, trace.id)

        # `_recompute_totals` aggregates the token counts of the stored
        # documents with a JOIN, and `_summarise_retrievals` counts the rows
        # back out of the table. Both are SELECTs, so the inserts have to be
        # flushed first -- session.add() alone only stages them in the
        # identity map, which makes every trace record zero context tokens and
        # report no retrieved documents.
        await session.flush()

        for retrieval in payload.retrievals:
            call = RetrievalCall(
                id=uuid.uuid4(),
                trace_id=trace.id,
                application_id=application_id,
                query=retrieval.query,
                retriever=retrieval.retriever,
                top_k=retrieval.top_k,
                num_results=len(retrieval.documents),
                latency_ms=retrieval.duration_ms,
                # Summed here and again from the stored rows afterwards; the
                # stored rows are the ones the trace total is derived from.
                context_tokens=sum(
                    document_token_count(document) for document in retrieval.documents
                ),
                # The strongest hit decides whether the retriever found
                # anything. Null, not zero, when nothing was retrieved.
                top_score=max(
                    (document.final_score for document in retrieval.documents),
                    default=None,
                ),
                configuration=retrieval.configuration or None,
                # `created_at` is the event-time column every analytics window
                # filters and buckets on, so a backdated batch that omitted this
                # would land entirely in the ingest minute. Left as None the
                # column's `server_default=now()` applies.
                **({} if retrieval.timestamp is None else {"created_at": retrieval.timestamp}),
            )
            call.documents = [
                self._build_document(document) for document in retrieval.documents
            ]
            session.add(call)

        # The document rows are cascaded from `call`, so one flush here makes
        # both the retrieval and its documents visible to the aggregates.
        await session.flush()

    def _build_document(self, document: RetrievedDocumentIn) -> RetrievedDocument:
        """One ranked hit, with the stored text trimmed to a preview."""
        text = _document_text(document)
        preview = document.content_preview or text[: self._preview_chars] or None
        return RetrievedDocument(
            id=uuid.uuid4(),
            document_id=document.document_id,
            title=document.title,
            rank=document.rank,
            final_score=document.final_score,
            bm25_score=document.bm25_score,
            vector_score=document.vector_score,
            rerank_score=document.rerank_score,
            token_count=document_token_count(document),
            content_preview=preview,
        )

    async def _persist_generations(
        self,
        session: AsyncSession,
        trace: Trace,
        application_id: uuid.UUID,
        payload: TraceCreate,
    ) -> None:
        """Write LLM calls, deriving token counts the caller omitted.

        ``model_id`` is deliberately left null. The model registry is seeded from
        the default price list, but a call against a model that was never
        registered is still a real call that must be counted, and dropping it
        would lose a number the dashboard reports. ``model_name`` and
        ``provider`` are denormalised onto the row so every ``GROUP BY model``
        in the analytics layer works without the foreign key.
        """
        if not payload.generations:
            return

        await self._delete_children(session, LLMCall, trace.id)

        # Every row must be visible to the aggregate query in
        # `_recompute_totals`, which runs an explicit SELECT over llm_calls.
        # session.add() only stages the row in the identity map, so without
        # this flush the trace total is computed against an empty table and
        # every trace reports zero tokens and zero cost.
        await session.flush()

        for generation in payload.generations:
            input_tokens, output_tokens = derive_generation_tokens(generation)
            session.add(
                LLMCall(
                    id=uuid.uuid4(),
                    trace_id=trace.id,
                    span_id=await self._resolve_span_id(session, trace, generation.span_id),
                    application_id=application_id,
                    model_name=generation.model,
                    provider=generation.provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    latency_ms=generation.duration_ms,
                    time_to_first_token_ms=generation.time_to_first_token_ms,
                    estimated_cost=calculate_cost(
                        input_tokens, output_tokens, generation.model
                    ),
                    status=generation.status,
                    prompt=generation.prompt,
                    completion=generation.completion,
                    temperature=generation.temperature,
                    max_tokens=generation.max_tokens,
                    extra_metadata=generation.metadata or None,
                    # As in `_persist_retrievals`: this is the event-time column
                    # the token, cost and latency windows all bucket on.
                    **(
                        {}
                        if generation.timestamp is None
                        else {"created_at": generation.timestamp}
                    ),
                )
            )

        # See the note in _persist_retrievals: the document-token aggregate in
        # `_recompute_totals` reads retrieved_documents through a JOIN, so the
        # rows have to be flushed before it can see them.
        await session.flush()

    async def _resolve_span_id(
        self, session: AsyncSession, trace: Trace, span_id: uuid.UUID | None
    ) -> uuid.UUID | None:
        """Validate a generation's span reference against this trace.

        A client replaying a payload from a previous response can name a span
        that has since been replaced, and the foreign key would then fail the
        whole trace over a link that carries no measurement. The reference is
        checked against this trace's own spans and dropped with a debug line
        when it no longer resolves.
        """
        if span_id is None:
            return None
        resolved = await session.scalar(
            select(Span.id).where(Span.id == span_id, Span.trace_id == trace.id)
        )
        if resolved is None:
            logger.debug(
                "telemetry.generation_span_unresolved",
                trace_id=trace.trace_id,
                span_id=str(span_id),
            )
            return None
        return resolved

    # ------------------------------------------------------------------
    # Denormalised totals
    # ------------------------------------------------------------------
    async def _recompute_totals(
        self, session: AsyncSession, trace: Trace, *, payload: TraceCreate
    ) -> None:
        """Recompute the trace's denormalised numbers from its own rows.

        ``total_tokens`` and ``estimated_cost`` come from the ``LLMCall`` rows
        and ``context_tokens`` from the ``retrieved_documents`` rows, all with
        SQL aggregations rather than by loading the rows into Python. The
        caller's own ``context_tokens`` is used only when the trace retrieved
        nothing at all, in which case it is the only recorded evidence of how
        much context the request carried.
        """
        input_tokens, output_tokens, total_tokens, cost = await self._sum_llm_calls(
            session, trace.id
        )
        measured_context = await self._sum_document_tokens(session, trace.id)

        trace.total_tokens = total_tokens
        trace.estimated_cost = cost
        trace.context_tokens = measured_context if payload.retrievals else max(
            0, payload.context_tokens
        )

        summaries = await self._summarise_retrievals(session, trace.id)
        self._store_analysis(
            trace,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_tokens=trace.context_tokens,
            summaries=summaries,
        )

    async def _sum_llm_calls(
        self, session: AsyncSession, trace_id: uuid.UUID
    ) -> tuple[int, int, int, float]:
        """Aggregate token counts and cost over a trace's generations."""
        with Timer() as timer:
            row = (
                await session.execute(
                    select(
                        func.coalesce(func.sum(LLMCall.input_tokens), 0),
                        func.coalesce(func.sum(LLMCall.output_tokens), 0),
                        func.coalesce(func.sum(LLMCall.total_tokens), 0),
                        func.coalesce(func.sum(LLMCall.estimated_cost), 0.0),
                    ).where(LLMCall.trace_id == trace_id)
                )
            ).one()
        log_db_query("telemetry.sum_llm_calls", timer.elapsed_ms)
        return int(row[0]), int(row[1]), int(row[2]), float(row[3])

    async def _sum_document_tokens(
        self, session: AsyncSession, trace_id: uuid.UUID
    ) -> int:
        """Aggregate the token count of every document the trace retrieved."""
        with Timer() as timer:
            total = await session.scalar(
                select(func.coalesce(func.sum(RetrievedDocument.token_count), 0))
                .select_from(RetrievedDocument)
                .join(RetrievalCall, RetrievalCall.id == RetrievedDocument.retrieval_call_id)
                .where(RetrievalCall.trace_id == trace_id)
            )
        log_db_query("telemetry.sum_document_tokens", timer.elapsed_ms)
        return int(total or 0)

    async def _summarise_retrievals(
        self, session: AsyncSession, trace_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Per-retrieval analysis, re-derived from the rows that were stored.

        Read back after the retrieval rows are flushed so the summary describes
        what is actually in the database rather than what the payload claimed.
        A payload that carried no retrievals still produces summaries when rows
        exist from an earlier ingest, which keeps the metadata honest for a
        re-sent trace.

        Only the document id, title and stored preview are read: the columns the
        duplicate detector needs, and not the passages themselves.
        """
        rows = await session.execute(
            select(
                RetrievalCall.id,
                RetrievalCall.retriever,
                RetrievalCall.context_tokens,
                RetrievalCall.top_score,
                RetrievedDocument.document_id,
                RetrievedDocument.title,
                RetrievedDocument.content_preview,
            )
            .select_from(RetrievalCall)
            .outerjoin(
                RetrievedDocument, RetrievedDocument.retrieval_call_id == RetrievalCall.id
            )
            .where(RetrievalCall.trace_id == trace_id)
            .order_by(RetrievalCall.created_at, RetrievedDocument.rank)
        )

        by_call: dict[uuid.UUID, dict[str, Any]] = {}
        for call_id, retriever, context_tokens, top_score, doc_id, title, preview in rows:
            summary = by_call.setdefault(
                call_id,
                {
                    "retriever": retriever,
                    "num_documents": 0,
                    "context_tokens": context_tokens,
                    "top_score": top_score,
                    "_ids": [],
                    "_titles": [],
                    "_contents": [],
                },
            )
            if doc_id is None:
                continue
            summary["num_documents"] += 1
            summary["_ids"].append(doc_id)
            summary["_titles"].append(title)
            summary["_contents"].append(preview)

        summaries: list[dict[str, Any]] = []
        for summary in by_call.values():
            document_ids = summary.pop("_ids")
            duplicates = _duplicates_from_columns(
                document_ids, summary.pop("_titles"), summary.pop("_contents")
            )
            num_documents = int(summary["num_documents"])
            summary["document_ids"] = document_ids
            summary["duplicate_document_ids"] = sorted(duplicates)
            summary["duplicate_document_count"] = len(duplicates)
            summary["duplicate_ratio"] = (
                round(len(duplicates) / num_documents, 4) if num_documents else 0.0
            )
            summaries.append(summary)
        return summaries

    def _store_analysis(
        self,
        trace: Trace,
        *,
        input_tokens: int,
        output_tokens: int,
        context_tokens: int,
        summaries: list[dict[str, Any]],
    ) -> None:
        """Write ``token_efficiency`` and ``retrieval_analysis`` into metadata.

        The efficiency score is the deterministic one from
        :func:`app.utils.tokens.token_efficiency`, computed from the recorded
        token counts and the measured duplicate ids — never estimated here. The
        duplicate set is the union across the trace's retrievals, because a
        context assembled from two retrievals that each repeat a document has
        still paid for that document twice.
        """
        duplicate_ids: set[str] = set()
        retrieved_ids: list[str] = []
        for summary in summaries:
            duplicate_ids.update(summary["duplicate_document_ids"])
            retrieved_ids.extend(summary.get("document_ids", []))

        num_documents = sum(int(summary["num_documents"]) for summary in summaries)

        metadata: dict[str, Any] = dict(trace.extra_metadata or {})
        metadata["token_efficiency"] = token_efficiency(
            input_tokens=input_tokens,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            retrieved_documents=retrieved_ids,
            duplicate_document_ids=duplicate_ids,
        )
        metadata["retrieval_analysis"] = {
            "num_documents": num_documents,
            "duplicate_document_ids": sorted(duplicate_ids),
            "duplicate_document_count": len(duplicate_ids),
            "duplicate_ratio": (
                round(len(duplicate_ids) / num_documents, 4) if num_documents else 0.0
            ),
            "context_tokens": context_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retrievals": summaries,
        }
        trace.extra_metadata = metadata

        if input_tokens == 0 and output_tokens == 0:
            # Not an error: a retrieval-only trace has no generation to compare
            # a context against. Logged so a low efficiency score in the
            # dashboard can be traced back to a trace that never called a model.
            logger.debug(
                "telemetry.efficiency_without_generation",
                trace_id=trace.trace_id,
                num_documents=num_documents,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    async def _delete_children(
        session: AsyncSession, model: type[Any], trace_id: uuid.UUID
    ) -> int:
        """Remove the rows of one child type for a re-sent trace.

        Deletes rather than updates because the payload is authoritative about
        how many spans, retrievals or generations the trace has; merging would
        leave orphans behind forever. The ``ON DELETE CASCADE`` on the foreign
        keys takes retrieved documents with their retrieval calls.
        """
        victim_ids = list(
            await session.scalars(select(model.id).where(model.trace_id == trace_id))
        )
        if not victim_ids:
            return 0
        await session.execute(delete(model).where(model.id.in_(victim_ids)))
        return len(victim_ids)


def _duplicates_from_columns(
    document_ids: list[str], titles: list[str | None], contents: list[str | None]
) -> set[str]:
    """The duplicate rule, expressed over three parallel columns.

    One implementation serves both callers — the payload path passes pydantic
    documents, the recompute path passes values read back from the database — so
    the rule cannot drift between "what ingest decided was duplicated" and "what
    the trace view will show as duplicated".
    """
    if len(document_ids) < 2:
        return set()

    id_occurrences: dict[str, int] = defaultdict(int)
    for document_id in document_ids:
        id_occurrences[document_id] += 1

    duplicates = {doc_id for doc_id, seen in id_occurrences.items() if seen > 1}

    fingerprint_groups: dict[str, list[str]] = defaultdict(list)
    for document_id, title, content in zip(document_ids, titles, contents, strict=True):
        for fingerprint in (normalise_title(title), _normalise(content)):
            if len(fingerprint) >= _MIN_FINGERPRINT_CHARS:
                fingerprint_groups[fingerprint].append(document_id)
    for ids in fingerprint_groups.values():
        if len(ids) > 1:
            duplicates.update(ids)
    return duplicates


_SERVICE: IngestService | None = None


def get_ingest_service() -> IngestService:
    """Return the process-wide :class:`IngestService`.

    One instance is enough: the service holds no per-request state, so the
    collector and the ingest routes share it.
    """
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = IngestService()
    return _SERVICE


def reset_ingest_service() -> None:
    """Drop the cached service. For tests and settings reloads."""
    global _SERVICE
    _SERVICE = None
