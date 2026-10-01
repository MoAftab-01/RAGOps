"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, then reranked.

The two retrievers fail in opposite ways, which is the whole reason to run both:

* BM25 matches *strings*. It finds ``ms-marco-MiniLM-L-6-v2`` when the query says
  exactly that, and it is blind to paraphrase.
* Dense search matches *meaning*. It handles "how do I stop the collector from
  dropping traces" -> a passage about buffer flushes, and it is unreliable on
  rare literal tokens.

Fusing them is not a matter of adding the scores. A BM25 score is an unbounded
IDF-weighted sum; a cosine is bounded in ``[-1, 1]``; a cross-encoder logit is
neither. Any weighted sum would be expressing an arbitrary belief about their
relative scales. **Reciprocal Rank Fusion** sidesteps that entirely: it consumes
only *ranks*, so no normalisation constant has to be invented, and the ``k = 60``
discount has been shown to be insensitive across two orders of magnitude
(Cormack et al., SIGIR '09).

Every per-stage score is preserved on the result. That is not decoration: the
trace view has to be able to explain *why* a document ranked where it did, and an
evaluation run has to be able to compare configurations without re-running them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Final

from app.config import settings
from app.core.logging import get_logger
from app.evaluation.metrics import HybridScorer
from app.ml.bm25 import BM25Index
from app.ml.chunking import Chunk
from app.ml.embeddings import get_embedder
from app.ml.reranker import get_reranker
from app.ml.vector_store import VectorStore, get_vector_store
from app.utils.tokens import estimate_tokens

logger = get_logger(__name__)

__all__ = ["HybridRetriever", "RetrievedChunk", "RetrieverResult"]

# Cosine similarity below which a dense *query* is treated as having found
# nothing at all.
#
# Why this exists: an inner-product search always returns its top-k, even when
# the query is nonsense. A bi-encoder puts *every* pair of English-ish strings
# in a narrow band of cosine similarity, so a query that matches nothing still
# scores ~0.1-0.2 against every passage. Without a gate, an unmatched query
# returns a full result list with confident-looking scores, indistinguishable in
# telemetry from a genuine match — precisely the fabricated result the project
# forbids.
#
# It gates the *query*, not each candidate. Filtering every hit individually was
# measured and rejected: legitimate low-ranked passages of a real query sit at
# cosine 0.09-0.22, so a per-candidate floor silently truncates good result sets
# in order to remove noise the gate already handles.
#
# The default sits in the measured gap for all-MiniLM-L6-v2: unmatched queries
# top out at 0.19-0.21, genuine matches start at 0.37. It is a constructor
# argument so a different embedding model can be re-tuned without editing this
# module, and every drop is logged so a badly-tuned value is visible rather than
# silently shrinking result sets.
#
# BM25 needs no equivalent gate: it already drops zero-score documents, so a
# document sharing no terms never enters its ranking.
MIN_VECTOR_RELEVANCE: Final = 0.25


@dataclass(slots=True)
class RetrievedChunk:
    """One ranked result, carrying the score contributed by every stage.

    The field names match ``app.models.core.RetrievedDocument`` one-for-one so a
    hit can become a telemetry row without a translation layer that might quietly
    drop or rename a score. ``bm25_score``/``vector_score``/``rerank_score`` are
    ``None`` when that stage did not produce a value — absence is meaningful and
    is distinguishable from ``0.0``, which means "scored, and scored zero".
    """

    document_id: str
    title: str
    content: str
    rank: int
    final_score: float
    bm25_score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    token_count: int = 0
    source: str | None = None
    chunk_index: int | None = None

    @property
    def context_tokens(self) -> int:
        """Tokens this chunk would add to a prompt.

        Read from the deterministic estimator rather than a model, per the
        project rule that an LLM never produces a token count.
        """
        return self.token_count or estimate_tokens(self.content)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection, used by the RAG API and demo app."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "rank": self.rank,
            "final_score": self.final_score,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "rerank_score": self.rerank_score,
            "token_count": self.token_count,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }


@dataclass(slots=True)
class RetrieverResult:
    """A retrieval run: the hits, the per-stage timing, and the config used.

    ``timings_ms`` and ``configuration`` are what make a run reproducible.
    Persisting them alongside the results is the difference between "the
    dashboard shows recall dropped" and "recall dropped after
    ``reranker_enabled`` flipped, and here is the query that regressed".
    """

    query: str
    results: list[RetrievedChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0

    @property
    def num_results(self) -> int:
        return len(self.results)

    @property
    def context_tokens(self) -> int:
        """Total tokens of the returned context.

        Computed here rather than stored so it always matches the hits actually
        returned, including the empty case (0), which is a real answer and not a
        missing value.
        """
        return sum(result.token_count for result in self.results)

    @property
    def top_score(self) -> float | None:
        """Score of the best hit, or ``None`` when nothing was retrieved.

        ``None`` and ``0.0`` are different claims: "no results" versus "results,
        all scoring zero". The dashboard must be able to tell them apart.
        """
        return self.results[0].final_score if self.results else None

    def document_ids(self) -> list[str]:
        """Ranked ids only — the shape :mod:`app.evaluation.metrics` expects."""
        return [result.document_id for result in self.results]


class HybridRetriever:
    """Owns the two indexes and runs the fused retrieval pipeline.

    Construction is cheap and side-effect free: the embedder and the vector store
    are only *resolved*, never loaded. The first :meth:`run` is what pays for the
    ~90 MB embedding checkpoint, which keeps the FastAPI process start instant.

    Not thread-safe for concurrent ``index_documents`` — indexing rebuilds both
    indexes wholesale and is expected to run from the ingestion path, one job at
    a time. Concurrent *reads* are safe: they touch immutable state.
    """

    def __init__(
        self,
        *,
        bm25: BM25Index | None = None,
        vector_store: VectorStore | None = None,
        reranker: Any | None = None,
        rrf_k: int | None = None,
        candidate_multiplier: int = 4,
        min_vector_relevance: float = MIN_VECTOR_RELEVANCE,
    ) -> None:
        self.bm25 = bm25 if bm25 is not None else BM25Index()
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self.reranker = reranker if reranker is not None else get_reranker()
        self.rrf_k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
        # Each retriever is asked for more than top_k. Fusion can only promote a
        # document that both retrievers surfaced, so a candidate set truncated
        # to exactly top_k would leave RRF nothing to reorder.
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.min_vector_relevance = float(min_vector_relevance)
        self._chunks: dict[str, Chunk] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_indexed(self) -> bool:
        """Whether the retriever has an index it can search."""
        return self.bm25.is_built and self.vector_store.size > 0

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    @property
    def chunks(self) -> dict[str, Chunk]:
        """Copy of the chunk metadata, keyed by ``external_id``."""
        return dict(self._chunks)

    def configuration(self) -> dict[str, Any]:
        """The retrieval config in force, for recording on a telemetry row.

        Every value is read from ``settings`` at call time so a reloaded
        configuration is reported accurately rather than captured at import.
        """
        return {
            "retriever": "hybrid",
            "rrf_k": self.rrf_k,
            "embedding_model": settings.embedding_model,
            "vector_store_backend": settings.vector_store_backend,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "reranker_enabled": self.reranker.enabled,
            "reranker_model": settings.reranker_model if self.reranker.enabled else None,
            "reranker_top_n": self.reranker.top_n,
            "candidate_multiplier": self.candidate_multiplier,
            "min_vector_relevance": self.min_vector_relevance,
            "embedding_device": get_embedder().device,
            "num_indexed_chunks": len(self._chunks),
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_documents(self, chunks: list[Chunk]) -> int:
        """Replace both indexes with ``chunks`` and return the count indexed.

        The chunk metadata dict is kept in memory because neither BM25 nor FAISS
        stores the passage text — without it a search returns ids that cannot be
        resolved to anything the user can read.

        Embeddings are computed here rather than in the caller because
        re-embedding is the single most expensive step in ingestion and it
        belongs next to the code that knows the batching settings.
        """
        if not chunks:
            self.bm25.build([])
            self._chunks = {}
            logger.info("ml.hybrid.index_empty")
            return 0

        texts = [chunk.content for chunk in chunks]
        vectors = get_embedder().embed(texts)

        self.bm25.build([(chunk.external_id, chunk.content) for chunk in chunks])
        self.vector_store.build([chunk.external_id for chunk in chunks], vectors)

        with self._lock:
            self._chunks = {chunk.external_id: chunk for chunk in chunks}

        logger.info(
            "ml.hybrid.indexed",
            chunks=len(chunks),
            dimension=int(vectors.shape[1]) if vectors.size else 0,
        )
        return len(chunks)

    def clear(self) -> None:
        """Drop both indexes, so the next run reports "not indexed" honestly."""
        self.bm25.clear()
        with self._lock:
            self._chunks = {}
        if hasattr(self.vector_store, "reset"):
            self.vector_store.reset()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        top_k: int | None = None,
        *,
        with_timings: bool = True,
    ) -> RetrieverResult:
        """Run the full pipeline for ``query`` and return the ranked hits.

        Stages: embed the query -> BM25 + vector search -> RRF fusion -> optional
        cross-encoder rerank -> truncate to ``top_k``.

        Each stage is timed individually and a failing stage is logged and
        skipped rather than aborting the query. A dense index that failed to load
        should degrade to BM25-only retrieval with an honest empty
        ``vector_score``; failing the whole request would turn a partial outage
        into a total one. When *both* retrievers produce nothing, the result is
        empty and the timings say why.
        """
        limit = top_k if top_k is not None else settings.default_top_k
        limit = max(1, limit)
        # Over-fetch from each retriever so the fusion has material to work with.
        candidates = limit * self.candidate_multiplier

        timings: dict[str, float] = {}
        result = RetrieverResult(query=query, configuration=self.configuration())

        if not query or not query.strip():
            logger.info("ml.hybrid.empty_query")
            return result

        if not self.is_indexed:
            logger.warning("ml.hybrid.not_indexed", query_length=len(query))
            return result

        # -- stage 1: embed ------------------------------------------------
        started = time.perf_counter()
        try:
            query_vector = get_embedder().embed_one(query)
        except Exception as exc:  # noqa: BLE001 - degraded, not fatal
            logger.error("ml.hybrid.embed_failed", error=str(exc))
            timings["embed_ms"] = (time.perf_counter() - started) * 1000.0
            return result
        timings["embed_ms"] = (time.perf_counter() - started) * 1000.0

        # -- stage 2: lexical ---------------------------------------------
        started = time.perf_counter()
        bm25_hits = self.bm25.search(query, top_k=candidates)
        timings["bm25_ms"] = (time.perf_counter() - started) * 1000.0

        # -- stage 3: dense -----------------------------------------------
        started = time.perf_counter()
        vector_hits: list[tuple[str, float]] = []
        try:
            raw_hits = self.vector_store.search(query_vector, top_k=candidates)
            # Gate the whole ranking on its best hit. See MIN_VECTOR_RELEVANCE:
            # if even the closest passage is below the floor, the query matched
            # nothing and the dense stage contributes an empty ranking rather
            # than an arbitrary top-k.
            if raw_hits and raw_hits[0][1] >= self.min_vector_relevance:
                vector_hits = raw_hits
            elif raw_hits:
                logger.info(
                    "ml.hybrid.vector_below_threshold",
                    top_score=round(raw_hits[0][1], 4),
                    threshold=self.min_vector_relevance,
                )
        except Exception as exc:  # noqa: BLE001 - degraded, not fatal
            logger.warning("ml.hybrid.vector_search_failed", error=str(exc))
        timings["vector_ms"] = (time.perf_counter() - started) * 1000.0

        bm25_scores = dict(bm25_hits)
        vector_scores = dict(vector_hits)
        result.candidate_count = len(set(bm25_scores) | set(vector_scores))

        # -- stage 4: RRF fusion ------------------------------------------
        started = time.perf_counter()
        rankings = [
            [doc_id for doc_id, _ in bm25_hits],
            [doc_id for doc_id, _ in vector_hits],
        ]
        fused = HybridScorer.rrf(rankings, k=self.rrf_k)
        timings["fusion_ms"] = (time.perf_counter() - started) * 1000.0

        # Deterministic order: RRF gives no ordering guarantee for equal scores,
        # and an unstable order would make two identical runs look like a
        # regression. Ties break on the id the first retriever ranked higher,
        # which is a stable, explainable rule.
        bm25_order = {doc_id: i for i, (doc_id, _s) in enumerate(bm25_hits)}
        vector_order = {doc_id: i for i, (doc_id, _s) in enumerate(vector_hits)}

        def sort_key(item: tuple[str, float]) -> tuple[float, int, int, str]:
            doc_id, score = item
            return (
                -score,
                min(
                    bm25_order.get(doc_id, len(bm25_order)),
                    vector_order.get(doc_id, len(vector_order)),
                ),
                bm25_order.get(doc_id, len(bm25_order)),
                doc_id,
            )

        ordered = sorted(fused.items(), key=sort_key)

        # -- stage 5: rerank -----------------------------------------------
        started = time.perf_counter()
        rerank_scores: dict[str, float] = {}
        if self.reranker.enabled and ordered:
            # Rerank a bounded window: the cross-encoder is a forward pass per
            # candidate, so it scores the fused head, not the whole corpus.
            window = ordered[: max(limit * self.candidate_multiplier, limit)]
            pairs = [
                (doc_id, self._chunks[doc_id].content)
                for doc_id, _s in window
                if doc_id in self._chunks
            ]
            if pairs:
                try:
                    for doc_id, score in self.reranker.rerank(query, pairs):
                        rerank_scores[doc_id] = score
                    # A cross-encoder logit is unbounded and incommensurable with
                    # an RRF score, so it *reorders* rather than overwrites: the
                    # final score stays the fusion score, and the rerank score
                    # rides along for explanation. Adding them would be the
                    # arbitrary weighted sum this design exists to avoid.
                    def rerank_key(item: tuple[str, float]) -> tuple[Any, ...]:
                        # Documents the reranker scored outrank the rest, and
                        # ties fall back to the fusion order, so the result is
                        # still a total, reproducible ordering.
                        return (
                            -rerank_scores.get(item[0], float("-inf")),
                            *sort_key(item),
                        )

                    ordered = sorted(ordered, key=rerank_key)
                except Exception as exc:  # noqa: BLE001 - degraded, not fatal
                    logger.warning("ml.hybrid.rerank_failed", error=str(exc))
        timings["rerank_ms"] = (time.perf_counter() - started) * 1000.0

        # -- stage 6: materialise + truncate -------------------------------
        started = time.perf_counter()
        hits: list[RetrievedChunk] = []
        for position, (doc_id, final_score) in enumerate(ordered, start=1):
            chunk = self._chunks.get(doc_id)
            if chunk is None:
                # An id with no chunk text is an index/ingest mismatch. Skipping
                # is right: returning a hit we cannot render would put an empty
                # passage into the model's context.
                logger.warning("ml.hybrid.missing_chunk_text", document_id=doc_id)
                continue
            hits.append(
                RetrievedChunk(
                    document_id=doc_id,
                    title=chunk.title,
                    content=chunk.content,
                    rank=position,
                    final_score=final_score,
                    bm25_score=bm25_scores.get(doc_id),
                    vector_score=vector_scores.get(doc_id),
                    rerank_score=rerank_scores.get(doc_id),
                    token_count=chunk.token_count,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                )
            )
            if len(hits) >= limit:
                break
        timings["total_ms"] = (time.perf_counter() - started) * 1000.0
        timings["total_ms"] += sum(
            timings.get(key, 0.0)
            for key in ("embed_ms", "bm25_ms", "vector_ms", "fusion_ms", "rerank_ms")
        )

        result.results = hits
        if with_timings:
            result.timings_ms = {key: round(value, 3) for key, value in timings.items()}

        logger.info(
            "ml.hybrid.retrieval_completed",
            candidates=result.candidate_count,
            returned=len(hits),
            rrf_k=self.rrf_k,
            reranked=bool(rerank_scores),
            duration_ms=round(timings["total_ms"], 2),
        )
        return result

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    def explain(self, query: str, document_id: str) -> dict[str, Any]:
        """Per-stage scores for one document, for a "why this result?" panel.

        Returns explicit ``found`` flags rather than a score of 0.0, because
        "the vector store did not return this document" and "the vector store
        ranked it last" are different facts about the system.
        """
        with self._lock:
            chunk = self._chunks.get(document_id)
        if chunk is None:
            return {"document_id": document_id, "found": False}

        bm25_scores = self.bm25.score_all(query)
        try:
            query_vector = get_embedder().embed_one(query)
            vector_hits = dict(self.vector_store.search(query_vector, top_k=self.size or 1))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ml.hybrid.explain_vector_failed", error=str(exc))
            vector_hits = {}

        return {
            "document_id": document_id,
            "found": True,
            "source": chunk.source,
            "title": chunk.title,
            "bm25_score": bm25_scores.get(document_id),
            "bm25_found": document_id in bm25_scores,
            "vector_score": vector_hits.get(document_id),
            "vector_found": document_id in vector_hits,
            "rrf_score": HybridScorer.rrf(
                [[document_id]], k=self.rrf_k
            ).get(document_id),
            "rrf_k": self.rrf_k,
        }
