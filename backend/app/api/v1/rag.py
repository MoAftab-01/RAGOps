"""The RAG demo endpoints: run the pipeline, and see the configuration doing it.

These are the only routes in RAGOps that *do* the thing they measure. Every other
endpoint reads back what telemetry recorded; ``/rag/query`` runs the real hybrid
pipeline against a real Ollama model and then records the run through the same
ingest service the collector uses, so the demo and production traffic are
indistinguishable once they are in the database. That is the point of the demo:
it is a client of the API it is instrumenting.

The response reports what each stage measured. A stage that did not run has a
null timing, and a document missing a stage's score has null for that score —
never ``0.0``, which would read as "scored, and scored zero" and is a different
claim. The scores are on different scales from one another (BM25 is unbounded,
cosine is bounded, the cross-encoder outputs logits), so they are reported as the
retriever recorded them and never normalised into a single score here.

Indexing is CPU-bound and takes seconds, so ``/rag/index`` is a write. Querying
is not a write on the telemetry path by itself, but it records a trace, so it
is guarded too: an open endpoint that appends rows to the database on every GET
is a way to fill one up.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.deps import DbSession, TenantPrincipal, WriteGuard
from app.config import settings
from app.core.logging import get_logger
from app.ml.retriever import get_retrieval_service
from app.models import utcnow
from app.schemas.ingest import (
    GenerationIn,
    RetrievedDocumentIn,
    RetrievalIn,
    TraceCreate,
)
from app.telemetry.ingest_service import get_ingest_service

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

#: The application demo queries are recorded against. Created on first use so a
#: fresh checkout can run the demo without a setup step.
DEMO_APPLICATION = "ragops-demo"


class RagQueryRequest(BaseModel):
    """One demo question.

    ``top_k`` and ``application`` are optional overrides; leaving them unset
    uses the configured pipeline defaults, which is what makes this endpoint a
    faithful demo rather than a second, differently-configured pipeline.
    """

    query: str = Field(min_length=1, max_length=4000, description="The question to ask")
    top_k: Annotated[int | None, Field(ge=1, le=50)] = None
    application: str | None = Field(default=None, max_length=128)


class RagQueryResponse(BaseModel):
    """The answer, its sources, and what it cost.

    ``usage`` distinguishes the three token counts the pipeline actually
    distinguishes: what the model was sent, what it produced, and what the
    retrieved context alone occupied. Collapsing them loses the one number an
    operator needs to tell "the model is verbose" from "we are stuffing the
    prompt".
    """

    trace_id: str
    answer: str
    documents: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    timing: dict[str, float | None] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    """What an index build actually produced.

    ``dimension`` is null until the index exists, because there is no embedding
    dimension to report for an index that was not built — reporting the
    configured model's dimension would be a guess about a computation that has
    not happened.
    """

    num_documents: int
    num_chunks: int
    backend: str
    dimension: int | None = None
    duration_ms: float


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="Run the full RAG pipeline and record the trace",
    dependencies=[WriteGuard],
)
async def rag_query(
    session: DbSession, payload: RagQueryRequest, principal: TenantPrincipal
) -> RagQueryResponse:
    """Answer a question with the real pipeline, and record what happened.

    Retrieval and generation are timed separately and reported as measured. If
    the generation fails, the retrieval that already happened is still recorded
    as a trace in the ``error`` state rather than discarded: a failed demo query
    is a real observation about the deployment, and dropping it would make the
    error rate look better than the system is.
    """
    started = time.perf_counter()
    service = get_retrieval_service()
    application_name = payload.application or DEMO_APPLICATION

    try:
        retrieval = service.query(payload.query, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001 - reported as a 502, not a stack trace
        logger.error(
            "rag.query.retrieval_failed",
            error=f"{type(exc).__name__}: {exc}",
            application=application_name,
        )
        raise _bad_gateway(f"Retrieval failed: {exc}") from exc

    context = service.context(payload.query, top_k=payload.top_k)
    timings = dict(retrieval.timings_ms)

    try:
        answer, generation_ms, usage = await _generate(context, payload.query)
    except Exception as exc:  # noqa: BLE001 - reported as a 502, not a stack trace
        await _record_failure(
            session,
            application_name,
            payload,
            retrieval,
            timings,
            exc,
            organization_id=principal.organization_id,
        )
        raise _bad_gateway(f"Generation failed: {exc}") from exc

    total_ms = (time.perf_counter() - started) * 1000.0
    trace_id = await _record_success(
        session,
        application_name,
        payload,
        retrieval,
        answer,
        usage,
        timings,
        total_ms,
        organization_id=principal.organization_id,
    )

    logger.info(
        "rag.query.completed",
        trace_id=trace_id,
        application=application_name,
        documents=len(retrieval.results),
        total_ms=round(total_ms, 2),
    )
    return RagQueryResponse(
        trace_id=trace_id,
        answer=answer,
        documents=[_document_out(hit) for hit in retrieval.results],
        usage=usage,
        timing={
            "total_ms": total_ms,
            "embedding_ms": timings.get("embedding_ms"),
            "bm25_ms": timings.get("bm25_ms"),
            "vector_ms": timings.get("vector_ms"),
            "rerank_ms": timings.get("rerank_ms"),
            "llm_ms": generation_ms,
        },
        configuration=retrieval.configuration,
    )


async def _generate(context: str, question: str) -> tuple[str, float, dict[str, Any]]:
    """Generate an answer from the retrieved context.

    Token counts come from the provider, which reports whether they are exact
    engine counts or the documented heuristic fallback. That flag is carried
    into ``usage`` so a caller can tell a measured count from an estimated one.
    """
    from app.providers.registry import get_provider

    provider = get_provider("ollama")
    prompt = (
        "Answer the question using only the context below. If the context does "
        "not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = await provider.generate([{"role": "user", "content": prompt}])

    usage: dict[str, Any] = {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "context_tokens": None,
        "token_counts_exact": response.token_counts_exact,
    }
    return response.text, response.latency_ms, usage


async def _record_success(
    session: DbSession,
    application_name: str,
    payload: RagQueryRequest,
    retrieval: Any,
    answer: str,
    usage: dict[str, Any],
    timings: dict[str, float],
    total_ms: float,
    *,
    organization_id: uuid.UUID | None = None,
) -> str:
    """Persist the run through the ingest service, returning the trace id.

    The same service the collector uses, so the demo's traces are stored with
    the same span, document, and call rows as any other traffic. Building
    ``TraceCreate`` here rather than writing the models directly is what keeps
    token totals and metadata derivation identical across both paths.
    """
    trace = TraceCreate(
        application=application_name,
        trace_id=_new_trace_id(),
        kind="rag",
        input_text=payload.query,
        output_text=answer,
        status="success",
        start_time=utcnow(),
        duration_ms=total_ms,
        retrievals=[
            RetrievalIn(
                query=payload.query,
                retriever=str(retrieval.configuration.get("retriever", "hybrid")),
                top_k=len(retrieval.results) or settings.default_top_k,
                duration_ms=timings.get("total_ms"),
                configuration=retrieval.configuration,
                documents=[_document_in(hit) for hit in retrieval.results],
            )
        ],
        generations=[
            GenerationIn(
                model=settings.ollama_model,
                provider="ollama",
                input_tokens=int(usage["input_tokens"] or 0),
                output_tokens=int(usage["output_tokens"] or 0),
                duration_ms=float(timings.get("llm_ms") or 0.0),
                metadata={"token_counts_exact": usage.get("token_counts_exact")},
            )
        ],
        metadata={
            "retrieval_timings_ms": timings,
            "retrieval_configuration": retrieval.configuration,
            "token_counts_exact": usage.get("token_counts_exact"),
            "source": "rag_demo_endpoint",
        },
    )
    return await get_ingest_service().ingest_trace(
        session,
        trace,
        application_name=application_name,
        organization_id=organization_id,
    )


def _document_in(hit: Any) -> RetrievedDocumentIn:
    """One retrieved chunk as the ingest schema wants it.

    ``token_count`` is dropped: ``RetrievedDocumentIn`` has no such field, and
    the server derives the token total from the documents themselves. Inventing
    a place to put it would mean a number stored in one place and counted in
    another.
    """
    return RetrievedDocumentIn(
        document_id=hit.document_id,
        title=hit.title,
        content=hit.content,
        rank=hit.rank,
        final_score=hit.final_score,
        bm25_score=hit.bm25_score,
        vector_score=hit.vector_score,
        rerank_score=hit.rerank_score,
    )


async def _record_failure(
    session: DbSession,
    application_name: str,
    payload: RagQueryRequest,
    retrieval: Any,
    timings: dict[str, float],
    exc: Exception,
    *,
    organization_id: uuid.UUID | None = None,
) -> None:
    """Record a failed demo run, so the error is visible in the dashboard.

    Best-effort: a failure while recording a failure must not replace the real
    error with a bookkeeping one, so any error here is logged and swallowed.
    """
    try:
        trace = TraceCreate(
            application=application_name,
            trace_id=_new_trace_id(),
            kind="rag",
            input_text=payload.query,
            status="error",
            start_time=utcnow(),
            error=f"{type(exc).__name__}: {exc}",
            retrievals=[
                RetrievalIn(
                    query=payload.query,
                    top_k=len(retrieval.results) or settings.default_top_k,
                    duration_ms=timings.get("total_ms"),
                    configuration=retrieval.configuration,
                    documents=[_document_in(hit) for hit in retrieval.results],
                )
            ],
            metadata={
                "retrieval_timings_ms": timings,
                "retrieval_configuration": retrieval.configuration,
                "source": "rag_demo_endpoint",
            },
        )
        await get_ingest_service().ingest_trace(
            session,
            trace,
            application_name=application_name,
            organization_id=organization_id,
        )
    except Exception:  # noqa: BLE001 - never mask the original failure
        logger.error("rag.query.failure_record_failed", exc_info=True)


def _document_out(hit: Any) -> dict[str, Any]:
    """One retrieved document, with each stage's score as it was recorded.

    A stage that produced no score stays null. The preview is truncated rather
    than the full content: the response is a UI payload, and the whole passage is
    already stored and reachable from ``/traces/{id}/retrieval``.
    """
    return {
        "document_id": hit.document_id,
        "title": hit.title,
        "rank": hit.rank,
        "final_score": hit.final_score,
        "bm25_score": hit.bm25_score,
        "vector_score": hit.vector_score,
        "rerank_score": hit.rerank_score,
        "token_count": hit.token_count,
        "content_preview": hit.content[:280],
    }


def _new_trace_id() -> str:
    """A fresh trace id in the format the telemetry path expects."""
    import uuid

    return uuid.uuid4().hex


@router.get("/config", summary="Current retrieval configuration")
async def rag_config() -> dict[str, Any]:
    """The pipeline configuration in force, from settings and the live service.

    Read straight from the service rather than echoed from settings, so it
    describes the retriever that will actually run — including whether an index
    is currently loaded, which is not a setting at all.
    """
    service = get_retrieval_service()
    return {
        **service.configuration(),
        "is_indexed": service.is_indexed,
        "indexed_at": service.indexed_at,
        "knowledge_base_path": str(service.knowledge_base_path),
    }


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="(Re)build the retrieval index",
    dependencies=[WriteGuard],
)
async def rag_index(force: Annotated[bool, Query(description="Rebuild even if current")] = False) -> IndexResponse:
    """Chunk the knowledge base and build the retrieval indexes.

    Synchronous and CPU-bound — embedding a corpus takes seconds, not
    milliseconds — and deliberately not moved to a background task, because a
    caller that gets a 200 before the index exists has no way to know when it
    became usable.

    Reports the real counts. A build over an empty or missing directory returns
    zero documents and zero chunks: that is a corpus that is not there, and the
    count is the fact, not a failure to report around.
    """
    service = get_retrieval_service()
    started = time.perf_counter()
    try:
        num_chunks = service.build_index(force=force)
    except Exception as exc:  # noqa: BLE001 - reported as a 502, not a stack trace
        logger.error("rag.index.failed", error=f"{type(exc).__name__}: {exc}")
        raise _bad_gateway(f"Index build failed: {exc}") from exc

    duration_ms = (time.perf_counter() - started) * 1000.0
    stats = service.stats()
    configuration = service.configuration()
    dimension = configuration.get("dimension") or configuration.get("embedding_dimension")

    logger.info(
        "rag.index.built",
        num_chunks=num_chunks,
        num_sources=stats["num_sources"],
        duration_ms=round(duration_ms, 2),
        forced=force,
    )
    return IndexResponse(
        num_documents=int(stats["num_sources"]),
        num_chunks=int(stats["num_chunks"]),
        backend=str(configuration.get("backend", "unknown")),
        dimension=int(dimension) if dimension else None,
        duration_ms=duration_ms,
    )


def _bad_gateway(detail: str):
    """502 for a downstream dependency that is unreachable or failed.

    502 rather than 500: the API is fine, the model host or corpus is not, and
    the distinction is what tells an operator where to look.
    """
    from fastapi import HTTPException, status

    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
