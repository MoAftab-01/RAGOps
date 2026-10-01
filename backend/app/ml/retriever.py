"""The retrieval service used by the demo app and by ``POST /api/rag/query``.

This is the seam the rest of the system talks to. It owns the end-to-end flow —
load the knowledge base, chunk it, embed it, build the two indexes, answer a
query — and it records what it actually did on every run, because a retrieval
metric without the configuration that produced it cannot be interpreted. A recall
of 0.62 means nothing until you know whether the reranker was on.

The service holds its index in memory. That is a deliberate trade: the corpus is
a knowledge base, not a continuously-changing feed, and an ingestion that
rebuilds in-process means there is no way for the index and the config to
disagree. The trade is paid at startup, which :meth:`ensure_index` defers to the
first query rather than hiding in an import.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.logging import Timer, get_logger
from app.ml.chunking import Chunk, chunk_documents, load_documents
from app.ml.hybrid import HybridRetriever, RetrievedChunk, RetrieverResult
from app.utils.tokens import estimate_tokens

logger = get_logger(__name__)

__all__ = ["RetrieverService", "get_retrieval_service"]


def _resolve_knowledge_base(explicit: str | Path | None) -> Path:
    """The corpus directory, resolved against the repository rather than the CWD.

    ``KNOWLEDGE_BASE_PATH`` is documented as relative to the repository root,
    but a relative path resolves against whatever directory the process
    happens to be in -- and RAGOps is started from all of them: the FastAPI
    server from ``backend/``, the demo from ``evaluation/``, pytest from
    ``backend/tests/``. From two of those three the configured path simply does
    not exist, and a missing corpus directory produces an *empty* index rather
    than an error. Retrieval then returns nothing for every query and reads as
    "the corpus does not cover this" when in fact no corpus was ever loaded.

    A relative path is therefore tried against every directory from the current
    one up to the filesystem root, and then against the same walk from this
    file's own location. Both walks are needed and neither substitutes for the
    other: the first honours a deliberate run from inside the project, the
    second still finds the corpus when the process was started somewhere else
    entirely (a service manager, a test runner, a temp directory).

    Absolute paths are taken as given -- someone who spelled out a full path
    means it, and rewriting it would send a test that points at a fixture to
    the real corpus instead.
    """
    if explicit is not None:
        return Path(explicit)

    configured = Path(settings.knowledge_base_path)
    if configured.is_absolute():
        return configured

    if configured.is_dir():
        # Resolved, not returned as-is: a relative Path that happens to work
        # from this directory stops working the moment anything changes the
        # CWD, and a service that indexed a corpus on startup and then lost it
        # is worse than one that never found it.
        return configured.resolve()

    # Two independent starting points, each walked all the way up. The first
    # entry of each tuple is the directory itself; the rest are its ancestors.
    # Walking up from the CWD is what makes a run from a nested directory work,
    # and walking up from this file is what makes a run started outside the
    # project work at all.
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents, here, *here.parents):
        candidate = base / configured
        if candidate.is_dir():
            return candidate

    logger.warning(
        "ml.retriever.knowledge_base_unresolved",
        path=str(configured),
        cwd=str(cwd),
    )
    return configured


class RetrieverService:

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        knowledge_base_path: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._retriever = retriever if retriever is not None else HybridRetriever()
        self._knowledge_base_path = _resolve_knowledge_base(knowledge_base_path)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunks: list[Chunk] = []
        self._indexed_at: float | None = None
        # One writer at a time. Indexing is a multi-second operation, and two
        # concurrent rebuilds would leave whichever finished second owning an
        # index built from a half-updated file set.
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def retriever(self) -> HybridRetriever:
        """The underlying hybrid retriever."""
        return self._retriever

    @property
    def is_indexed(self) -> bool:
        """Whether a searchable index exists."""
        return self._retriever.is_indexed

    @property
    def num_chunks(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._chunks)

    @property
    def indexed_at(self) -> float | None:
        """Unix timestamp of the last successful index build, if any."""
        return self._indexed_at

    @property
    def knowledge_base_path(self) -> Path:
        """Directory the corpus is read from."""
        return self._knowledge_base_path

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def build_index(self, *, force: bool = False) -> int:
        """Chunk the knowledge base and build both retrieval indexes.

        Returns the number of indexed chunks. Re-entrant calls are cheap: a
        second call while the index is current returns immediately unless
        ``force`` is set, so a request that arrives during ingestion does not
        trigger a second rebuild.
        """
        with self._index_lock:
            if self.is_indexed and not force:
                logger.debug("ml.retriever.index_already_built", chunks=len(self._chunks))
                return len(self._chunks)

            with Timer() as timer:
                documents = load_documents(self._knowledge_base_path)
                chunks = chunk_documents(
                    documents,
                    self._chunk_size if self._chunk_size is not None else settings.chunk_size,
                    (
                        self._chunk_overlap
                        if self._chunk_overlap is not None
                        else settings.chunk_overlap
                    ),
                )
                self._retriever.index_documents(chunks)

            self._chunks = chunks
            self._indexed_at = time.time()

            if not documents:
                # Not an error: the knowledge base may not have been seeded yet.
                # The log makes it findable, and retrieval will return empty
                # results with a reason rather than failing.
                logger.warning(
                    "ml.retriever.empty_knowledge_base",
                    path=str(self._knowledge_base_path),
                )

            logger.info(
                "ml.retriever.index_built",
                path=str(self._knowledge_base_path),
                documents=len(documents),
                chunks=len(chunks),
                duration_ms=round(timer.elapsed_ms, 2),
            )
            return len(chunks)

    def ensure_index(self) -> int:
        """Build the index only if it is not already there.

        Called at the start of every query. The cost of an unindexed request is
        one corpus scan; the cost afterwards is a boolean.
        """
        if self.is_indexed:
            return len(self._chunks)
        return self.build_index()

    def persist_index(self, path: str | Path | None = None) -> Path | None:
        """Write the vector index to disk, returning the file written.

        Returns ``None`` when the configured backend has nothing local to write
        (Qdrant is already durable) or when the store is empty. Callers treat
        ``None`` as "nothing was persisted", never as a failure to report.
        """
        if not self.is_indexed:
            logger.info("ml.retriever.persist_skipped", reason="not_indexed")
            return None
        try:
            from app.ml.vector_store import default_index_path  # noqa: PLC0415

            target = Path(path) if path is not None else default_index_path()
            return self._retriever.vector_store.save(target)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("ml.retriever.persist_failed", error=str(exc))
            return None

    def load_index(self, path: str | Path) -> bool:
        """Load a previously persisted index. Returns whether it loaded."""
        try:
            self._retriever.vector_store.load(path)
        except Exception as exc:  # noqa: BLE001 - a missing index is recoverable
            logger.warning(
                "ml.retriever.load_failed", path=str(path), error=str(exc)
            )
            return False
        logger.info("ml.retriever.index_loaded", path=str(path))
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        query: str,
        top_k: int | None = None,
        *,
        ensure_index: bool = True,
    ) -> RetrieverResult:
        """Answer a retrieval query and report timings plus the config used.

        ``ensure_index`` is on by default so a caller cannot accidentally query
        an unindexed service and get an empty result that looks like a retrieval
        failure. The index build is included in the total timing, so a slow
        first query is explained rather than mysterious.
        """
        limit = top_k if top_k is not None else settings.default_top_k

        index_ms = 0.0
        if ensure_index:
            with Timer() as timer:
                self.ensure_index()
            index_ms = timer.elapsed_ms

        result = self._retriever.run(query, top_k=limit)

        if index_ms:
            result.timings_ms["index_ms"] = round(index_ms, 3)
            result.timings_ms["total_ms"] = round(
                result.timings_ms.get("total_ms", 0.0) + index_ms, 3
            )
        return result

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Convenience wrapper returning just the ranked hits."""
        return self.query(query, top_k=top_k).results

    def context(
        self, query: str, top_k: int | None = None
    ) -> str:
        """The retrieved passages as one prompt-ready context block.

        Numbered and attributed to the source file, because a generation that
        cites "[1]" has to be able to resolve it. A join with no separators
        would produce a context that reads as one continuous document and makes
        attribution meaningless.
        """
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        return "\n\n".join(
            f"[{index}] {hit.title} ({hit.source or 'unknown source'})\n{hit.content}"
            for index, hit in enumerate(hits, start=1)
        )

    def configuration(self) -> dict[str, Any]:
        """The retrieval configuration in force, for a telemetry row.

        Also carries the corpus facts (path, document count, chunk count,
        total tokens) so a stored configuration fully describes the run rather
        than only the knobs.
        """
        config = self._retriever.configuration()
        config.update(
            {
                "knowledge_base_path": str(self._knowledge_base_path),
                "num_documents": len({chunk.source for chunk in self._chunks}),
                "num_chunks": len(self._chunks),
                "corpus_tokens": sum(chunk.token_count for chunk in self._chunks),
                "top_k": settings.default_top_k,
            }
        )
        return config

    def stats(self) -> dict[str, Any]:
        """Corpus statistics, all measured from the indexed chunks."""
        chunks = self._chunks
        return {
            "is_indexed": self.is_indexed,
            "knowledge_base_path": str(self._knowledge_base_path),
            "num_chunks": len(chunks),
            "num_sources": len({chunk.source for chunk in chunks}),
            "total_tokens": sum(chunk.token_count for chunk in chunks),
            "mean_chunk_tokens": (
                round(sum(c.token_count for c in chunks) / len(chunks), 2) if chunks else 0.0
            ),
            "indexed_at": self._indexed_at,
        }

    def estimate_query_tokens(self, query: str, top_k: int | None = None) -> int:
        """Deterministic token estimate for a query plus its retrieved context.

        Used to pre-check a request against the context window *before* calling
        the model. Computed with :func:`app.utils.tokens.estimate_tokens`, never
        by asking a model, so the guard is deterministic and free.
        """
        hits = self.retrieve(query, top_k=top_k)
        return estimate_tokens(query) + sum(hit.token_count for hit in hits)


# The service is constructed lazily but cached, so a request and a background
# job share one index. Creating it does not load a model or read the disk.
_SERVICE: RetrieverService | None = None
_SERVICE_LOCK = threading.Lock()


def get_retrieval_service() -> RetrieverService:
    """Return the process-wide :class:`RetrieverService`."""
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = RetrieverService()
        return _SERVICE


def reset_retrieval_service() -> None:
    """Drop the cached service. For tests and settings reloads."""
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = None
