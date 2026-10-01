"""Vector store interface with a local FAISS backend and a Qdrant backend.

The contract every backend implements is deliberately narrow — build, search,
save, load, size — because that is the whole surface the retriever needs, and a
narrow interface is what makes the backend swappable without touching
:mod:`app.ml.hybrid`.

**Score semantics are fixed across backends:** every store indexes L2-normalised
vectors and returns a similarity in ``[-1, 1]`` where higher is more similar.
FAISS does it with ``IndexFlatIP``; Qdrant does it by asking for cosine. If a
future backend returns a distance instead, it must negate it before returning —
a silently inverted ranking is the worst possible failure here.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from app.config import settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "FaissVectorStore",
    "QdrantUnavailableError",
    "QdrantVectorStore",
    "VectorStoreError",
    "VectorStoreNotBuiltError",
    "get_vector_store",
    "l2_normalise",
    "reset_vector_store",
]

# Where FAISS indices live. Overridable so a test can point it at tmp_path
# without touching the repo.
DEFAULT_INDEX_DIR: Final = "ml/models"

# Sidecar written next to the .faiss file. FAISS itself only stores vectors, so
# the id->row mapping has to survive a restart or the search results come back
# as meaningless integers.
_ID_MAP_SUFFIX: Final = ".ids.json"
_INDEX_SUFFIX: Final = ".faiss"


class VectorStoreError(RuntimeError):
    """Base class for vector-store failures the caller is expected to handle."""


class VectorStoreNotBuiltError(VectorStoreError):
    """Raised when a search runs against an empty or unloaded store."""


class QdrantUnavailableError(VectorStoreError):
    """Raised when Qdrant is configured but cannot be reached.

    Distinct from a generic error so the API layer can tell "your configuration
    is wrong" apart from "the vector search broke", and so the fallback path in
    :func:`get_vector_store` can catch exactly this and degrade to FAISS.
    """


def l2_normalise(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    """Unit-normalise rows so inner product equals cosine similarity.

    Re-exported from :mod:`app.ml.embeddings` so the vector store guarantees
    the invariant itself instead of trusting that every caller remembered to
    normalise. Zero rows are preserved rather than turned into NaN.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.size == 0:
        return array.astype(np.float32, copy=False)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (array / safe).astype(np.float32, copy=False)


class VectorStore(ABC):
    """Interface every vector backend implements.

    ``build`` replaces the contents wholesale rather than appending. Retrieval
    indexes are rebuilt from the full corpus on each ingestion; supporting
    incremental upserts would add a delete path that has to be exactly right
    before stale chunks stop polluting results.
    """

    @abstractmethod
    def build(self, ids: list[str], embeddings: NDArray[np.floating]) -> None:
        """Replace the index contents with ``ids`` aligned to ``embeddings``."""

    @abstractmethod
    def search(
        self, query_vector: NDArray[np.floating], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(id, similarity)`` pairs, best first."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist the index to ``path`` and return the file actually written."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load a previously saved index."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors indexed."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Embedding width, or ``None`` before the index is built."""

    @property
    def is_built(self) -> bool:
        """Whether the store can serve a search."""
        return self.size > 0


class FaissVectorStore(VectorStore):
    """In-process FAISS store using an exact inner-product index.

    ``IndexFlatIP`` is the right choice at RAGOps' scale: it is exact (no
    approximation, so no recall loss to explain in an evaluation report) and a
    few hundred thousand 384-dimension vectors fit comfortably in RAM. HNSW or
    IVFFlat would only earn their approximation error once the corpus outgrows
    memory, and that is a later, measured decision.

    Vectors are L2-normalised on the way in *and* on the way out, so a score is
    a cosine similarity in ``[-1, 1]`` with 1.0 meaning identical direction.
    """

    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._index: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int | None:
        if self._index is not None:
            return int(self._index.d)
        return self._dimension

    @property
    def ids(self) -> list[str]:
        """Copy of the stored ids, in row order."""
        return list(self._ids)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def build(self, ids: list[str], embeddings: NDArray[np.floating]) -> None:
        """Build an exact inner-product index over normalised ``embeddings``.

        Vectors are re-normalised here rather than assumed normalised, because
        a caller that forgets would otherwise get scores in a meaningless
        range and a ranking that looked plausible enough not to be questioned.
        """
        import faiss  # noqa: PLC0415 - ~30 MB native lib, kept off the import path

        if len(ids) != np.asarray(embeddings).shape[0]:
            raise ValueError(
                f"ids/embeddings length mismatch: {len(ids)} ids, "
                f"{np.asarray(embeddings).shape[0]} vectors"
            )
        if not ids:
            with self._lock:
                self._index = None
                self._ids = []
                self._id_to_row = {}
            return

        matrix = l2_normalise(embeddings)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2-D array")

        dim = int(matrix.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(matrix, dtype=np.float32))

        with self._lock:
            self._index = index
            self._ids = list(ids)
            self._id_to_row = {doc_id: row for row, doc_id in enumerate(ids)}
            self._dimension = dim

        logger.info("ml.vector_store.faiss_built", vectors=len(ids), dimension=dim)

    def search(
        self, query_vector: NDArray[np.floating], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Nearest neighbours by cosine similarity, best first.

        The query is normalised before the inner product, so the returned score
        is comparable with the scores from any other backend.
        """
        index = self._index
        if index is None:
            raise VectorStoreNotBuiltError("FAISS index has not been built or loaded")
        if top_k <= 0:
            return []

        query = l2_normalise(query_vector)
        if query.ndim == 2:
            query = query[0]
        if query.shape[0] != index.d:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match index dimension {index.d}"
            )

        limit = min(top_k, len(self._ids))
        # FAISS returns (scores, row_indices) for the k nearest.
        scores, rows = index.search(
            np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), limit
        )

        results: list[tuple[str, float]] = []
        for score, row in zip(scores[0].tolist(), rows[0].tolist(), strict=True):
            if row < 0 or row >= len(self._ids):
                continue
            results.append((self._ids[int(row)], float(score)))
        return results

    def save(self, path: str | Path) -> Path:
        """Write the index plus its id sidecar, creating parent directories.

        Both files are needed on reload; the id map is what turns FAISS's row
        offsets back into document ids.
        """
        import faiss  # noqa: PLC0415

        index = self._index
        if index is None:
            raise VectorStoreNotBuiltError("cannot save an unbuilt FAISS index")

        target = Path(path)
        if target.is_dir() or target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            target = target / f"ragops{_INDEX_SUFFIX}"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(target))

        # Replace the .faiss suffix rather than appending, so callers can pass
        # either a directory or an explicit filename and get a predictable pair.
        id_map_path = target.with_suffix(_ID_MAP_SUFFIX)
        id_map_path.write_text(
            json.dumps({"dimension": int(index.d), "ids": self._ids}),
            encoding="utf-8",
        )
        logger.info(
            "ml.vector_store.faiss_saved",
            path=str(target),
            id_map=str(id_map_path),
            vectors=len(self._ids),
        )
        return target

    def load(self, path: str | Path) -> None:
        """Read an index and its id sidecar written by :meth:`save`."""
        import faiss  # noqa: PLC0415

        target = Path(path)
        if target.is_dir():
            target = target / f"ragops{_INDEX_SUFFIX}"
        if not target.exists():
            raise VectorStoreNotBuiltError(f"no FAISS index at {target}")
        if target.suffix == "":
            target = target.with_suffix(_INDEX_SUFFIX)

        id_map_path = target.with_suffix(_ID_MAP_SUFFIX)
        if not id_map_path.exists():
            raise VectorStoreNotBuiltError(f"missing id sidecar at {id_map_path}")

        payload = json.loads(id_map_path.read_text(encoding="utf-8"))
        ids = [str(value) for value in payload.get("ids", [])]

        index = faiss.read_index(str(target))
        if index.ntotal != len(ids):
            raise VectorStoreNotBuiltError(
                f"index has {index.ntotal} vectors but the id sidecar lists {len(ids)}"
            )

        with self._lock:
            self._index = index
            self._ids = ids
            self._id_to_row = {doc_id: row for row, doc_id in enumerate(ids)}
            self._dimension = int(index.d)

        logger.info(
            "ml.vector_store.faiss_loaded",
            path=str(target),
            vectors=len(ids),
            dimension=self._dimension,
        )

    def reset(self) -> None:
        """Drop the in-memory index (used when the corpus is rebuilt)."""
        with self._lock:
            self._index = None
            self._ids = []
            self._id_to_row = {}
            self._dimension = None


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store using cosine distance.

    The Qdrant client is imported *inside* every method. ``qdrant-client`` is an
    optional extra, and importing this module must never require it — otherwise
    setting ``vector_store_backend="qdrant"`` in an env file would break the
    whole app on a machine that only meant to test the setting.

    Every network failure is funnelled into :class:`QdrantUnavailableError` so
    the caller has one exception type to catch, and so the message names the URL
    that was tried instead of surfacing a raw httpx traceback.
    """

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._url = url or settings.qdrant_url
        self._collection = collection or settings.qdrant_collection
        self._timeout = timeout_seconds
        self._size: int | None = None
        self._dimension: int | None = None
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        if self._size is not None:
            return self._size
        # Count lazily rather than caching a possibly-stale total from a
        # previous build; Qdrant is authoritative about this.
        try:
            client = self._get_client()
            self._size = int(client.count(collection=self._collection, exact=True).count)
        except Exception:
            return 0
        return self._size

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def collection(self) -> str:
        return self._collection

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def build(self, ids: list[str], embeddings: NDArray[np.floating]) -> None:
        """Upsert ``(id, vector)`` pairs into the configured collection.

        The collection is created on demand with the right size and cosine
        distance, so a first run against an empty Qdrant works without manual
        setup.
        """
        if not ids:
            return
        matrix = l2_normalise(embeddings)
        if len(ids) != matrix.shape[0]:
            raise ValueError(
                f"ids/embeddings length mismatch: {len(ids)} ids, {matrix.shape[0]} vectors"
            )

        from qdrant_client import models  # noqa: PLC0415

        client = self._get_client()
        dim = int(matrix.shape[1])
        self._ensure_collection(client, models, dim)

        # Upsert in batches: a single request with 100k points exceeds Qdrant's
        # default payload limits and fails the whole build.
        batch_size = 256
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            points = [
                models.PointStruct(
                    id=self._point_id(doc_id),
                    vector=matrix[row].tolist(),
                    payload={"external_id": doc_id, "row": row},
                )
                for row, doc_id in enumerate(ids[start:stop], start=start)
            ]
            client.upsert(collection_name=self._collection, points=points)

        self._size = len(ids)
        self._dimension = dim
        logger.info(
            "ml.vector_store.qdrant_built",
            collection=self._collection,
            vectors=len(ids),
            dimension=dim,
        )

    def search(
        self, query_vector: NDArray[np.floating], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search, best first.

        Qdrant returns distances; ``Cosine`` distance is ``1 - cosine_similarity``,
        so the score is negated to keep the interface's "higher is better"
        contract identical to FAISS's.
        """
        from qdrant_client import models  # noqa: PLC0415

        if top_k <= 0:
            return []
        client = self._get_client()
        query = l2_normalise(query_vector)
        if query.ndim == 2:
            query = query[0]

        try:
            response = client.query_points(
                collection_name=self._collection,
                query=query.tolist(),
                limit=top_k,
                with_payload=True,
                query_filter=models.Filter(must=[]),
            )
            points = response.points
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise QdrantUnavailableError(
                f"Qdrant query failed against {self._url}: {exc}"
            ) from exc

        results: list[tuple[str, float]] = []
        for point in points:
            external_id = (point.payload or {}).get("external_id")
            if not external_id:
                continue
            results.append((str(external_id), float(point.score)))
        return results

    def save(self, path: str | Path) -> Path:
        """No-op: Qdrant is the durable store, so there is nothing to write.

        A marker file is written anyway so the caller can record "this corpus
        was ingested" in a way that is symmetric with the FAISS backend.
        """
        target = Path(path)
        if target.is_dir() or target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "qdrant_collection.json"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"url": self._url, "collection": self._collection, "size": self._size},
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    def load(self, path: str | Path) -> None:
        """Verify the configured collection is reachable and read its size.

        Unlike FAISS there is no local file to read — the "load" is a health
        check, which is also the earliest point a misconfigured URL can be
        reported usefully.
        """
        client = self._get_client()
        try:
            info = client.get_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            raise QdrantUnavailableError(
                f"Qdrant collection {self._collection!r} unreachable at {self._url}: {exc}"
            ) from exc

        vectors = getattr(info.config.params, "vectors", None)
        if vectors is not None and getattr(vectors, "size", None):
            self._dimension = int(vectors.size)
        self._size = int(getattr(info, "points_count", 0) or 0)
        logger.info(
            "ml.vector_store.qdrant_loaded",
            collection=self._collection,
            points=self._size,
            dimension=self._dimension,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _point_id(external_id: str) -> int:
        """Map an external id to Qdrant's unsigned-integer point id.

        Qdrant requires integer (or UUID) ids, and our ids are 16 hex chars, so
        the first 63 bits of the digest are used. Stable across runs, which is
        what makes re-ingestion idempotent.
        """
        return int(external_id[:15], 16)

    def _ensure_collection(self, client: Any, models: Any, dim: int) -> None:
        """Create the collection if it is missing, or if its size disagrees."""
        try:
            existing = client.get_collection(self._collection)
        except Exception:  # noqa: BLE001 - not-found is the expected path
            existing = None

        if existing is not None:
            vectors = getattr(existing.config.params, "vectors", None)
            existing_dim = getattr(vectors, "size", None) if vectors else None
            if existing_dim == dim:
                return
            # A dimension change means the corpus was re-embedded with a
            # different model; the old points are meaningless, so recreate.
            logger.warning(
                "ml.vector_store.qdrant_dimension_changed",
                collection=self._collection,
                old=existing_dim,
                new=dim,
            )
            client.delete_collection(self._collection)

        client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )

    def _get_client(self) -> Any:
        """Return a cached client, converting every failure into our error type."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415
        except ImportError as exc:
            # Logged loudly because this is a configuration mistake the user can
            # act on, not a transient fault.
            logger.error(
                "ml.vector_store.qdrant_client_missing",
                hint="pip install qdrant-client, or set vector_store_backend=faiss",
            )
            raise QdrantUnavailableError(
                "qdrant-client is not installed; set vector_store_backend=faiss "
                "or install the optional dependency"
            ) from exc

        try:
            self._client = QdrantClient(url=self._url, timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ml.vector_store.qdrant_client_failed",
                url=self._url,
                error=str(exc),
            )
            raise QdrantUnavailableError(
                f"could not create a Qdrant client for {self._url}: {exc}"
            ) from exc
        return self._client


def default_index_path() -> Path:
    """Conventional on-disk location for the FAISS index."""
    return Path(DEFAULT_INDEX_DIR) / "ragops.faiss"


_VECTOR_STORE: VectorStore | None = None
_VECTOR_STORE_LOCK = threading.Lock()


def get_vector_store(backend: str | None = None) -> VectorStore:
    """Return the configured vector store, degrading instead of crashing.

    ``backend`` defaults to ``settings.vector_store_backend``. Selecting Qdrant
    and finding it down must not take the API down with it: a missing client or
    an unreachable server is logged as a warning and the call falls back to the
    local FAISS store, which needs nothing but the corpus. Retrieval quality
    changes; the service stays up and the reason is in the log.
    """
    global _VECTOR_STORE
    choice = (backend or settings.vector_store_backend or "faiss").strip().lower()

    if choice == "faiss":
        return _cached(FaissVectorStore())

    if choice == "qdrant":
        store = QdrantVectorStore()
        try:
            # Cheap reachability probe, so an unreachable server is reported at
            # the point of selection rather than at the first user query.
            store.load(default_index_path())
        except QdrantUnavailableError as exc:
            logger.warning(
                "ml.vector_store.qdrant_unreachable_falling_back_to_faiss",
                url=settings.qdrant_url,
                collection=settings.qdrant_collection,
                error=str(exc),
            )
            return _cached(FaissVectorStore())
        return _cached(store)

    logger.warning(
        "ml.vector_store.unknown_backend",
        backend=choice,
        hint="expected 'faiss' or 'qdrant'; using faiss",
    )
    return _cached(FaissVectorStore())


def _cached(store: VectorStore) -> VectorStore:
    """Memoise the process-wide store so two calls share one index."""
    global _VECTOR_STORE
    with _VECTOR_STORE_LOCK:
        if _VECTOR_STORE is None:
            _VECTOR_STORE = store
        return _VECTOR_STORE


def reset_vector_store() -> None:
    """Drop the cached store. For tests and settings reloads."""
    global _VECTOR_STORE
    with _VECTOR_STORE_LOCK:
        _VECTOR_STORE = None
