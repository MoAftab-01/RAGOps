"""Dense text embeddings.

One :class:`Embedder` per process wraps a sentence-transformers checkpoint and
turns text into L2-normalised float32 vectors. Normalisation is not cosmetic:
the FAISS index is an ``IndexFlatIP`` (see :mod:`app.ml.vector_store`), so
inner product only *equals* cosine similarity when both sides are unit length.

Nothing here runs at import time. The checkpoint is roughly 90 MB for
``all-MiniLM-L6-v2`` and is downloaded by the first :meth:`Embedder.embed` call,
which means the FastAPI process starts instantly, unit tests that never embed
anything never pay for the download, and importing :mod:`app.ml` never touches
the network.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from app.config import settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch off the import path
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = ["Embedder", "get_embedder", "l2_normalise", "resolve_device"]

# ``sentence-transformers`` is a ~2 GB import (it pulls torch). It is therefore
# imported inside the loader, never at module scope: a cold ``import app.ml``
# must not cost half a second and must not require the ML extras to be present.
_MODEL_LOCK: Final = threading.Lock()


def resolve_device(requested: str | None = None) -> str:
    """Turn ``settings.embedding_device`` into a concrete torch device string.

    ``auto`` (the default) is resolved against what torch actually reports, so
    the same settings object works on a developer laptop with no CUDA build and
    on a GPU host. An explicit ``cpu`` is honoured even when CUDA exists: the
    flag exists precisely so someone can force the slow, memory-light path.
    """
    choice = (requested if requested is not None else settings.embedding_device) or "auto"
    choice = choice.strip().lower()

    if choice != "auto":
        return choice

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing or broken install
        logger.debug("ml.embeddings.device_probe_failed", falling_back_to="cpu")
    return "cpu"


def l2_normalise(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return unit-length float32 rows, leaving all-zero rows untouched.

    A zero vector has no direction, so normalising it would divide by zero and
    produce NaNs that then poison a FAISS inner-product search. Keeping the zero
    row intact means it simply scores 0.0 against everything, which is the
    honest answer for "no signal here".
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.size == 0:
        return array.astype(np.float32, copy=False)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # ``where`` keeps the division away from zero rows instead of warning on them.
    safe = np.where(norms == 0.0, 1.0, norms)
    return (array / safe).astype(np.float32, copy=False)


class Embedder:
    """Lazily-loaded sentence-transformers embedder.

    The instance is reusable and thread-safe with respect to model loading: the
    first caller performs the load while the rest block on a lock, so a burst of
    concurrent requests triggers exactly one (expensive, cached) checkpoint load
    rather than one per request.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._device = resolve_device(device)
        self._batch_size = max(1, batch_size or settings.embedding_batch_size)
        self._model: Any | None = None
        self._dimension: int | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        """The checkpoint that will be (or was) loaded."""
        return self._model_name

    @property
    def device(self) -> str:
        """The resolved torch device, e.g. ``cpu`` or ``cuda``."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Number of texts pushed through the model in one forward pass."""
        return self._batch_size

    @property
    def is_loaded(self) -> bool:
        """Whether the checkpoint is resident. Safe to call before first use."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Embedding width, e.g. 384 for ``all-MiniLM-L6-v2``.

        Reading this before the model is loaded triggers the load, because the
        width is a property of the checkpoint and hardcoding it would be a
        guess that silently breaks the moment ``settings.embedding_model`` is
        swapped for a different model.
        """
        if self._dimension is None:
            self._ensure_model()
        assert self._dimension is not None  # noqa: S101 - set by _ensure_model
        return self._dimension

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed ``texts`` into an ``(len(texts), dimension)`` float32 matrix.

        Rows are L2-normalised so downstream inner products are cosine
        similarities. Batching is handled internally using
        ``settings.embedding_batch_size`` because one giant encode() call is
        the usual cause of a CUDA OOM; a caller should not have to think about
        it. An empty input returns an empty ``(0, dimension)`` matrix rather
        than raising, so a retrieval stage can short-circuit without a guard.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._ensure_model()
        # sentence-transformers cannot encode an empty string, and a blank chunk
        # in a knowledge base is a data bug, not a reason to fail the request.
        cleaned = [text if text and text.strip() else " " for text in texts]

        vectors: list[NDArray[np.float32]] = []
        for start in range(0, len(cleaned), self._batch_size):
            batch = cleaned[start : start + self._batch_size]
            encoded = model.encode(
                batch,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors.append(np.asarray(encoded, dtype=np.float32))

        stacked = np.vstack(vectors)
        # Re-normalise defensively: the guarantee that inner product == cosine
        # is load-bearing for retrieval scores, so it is enforced here rather
        # than trusted from the library.
        return l2_normalise(stacked)

    def embed_one(self, text: str) -> NDArray[np.float32]:
        """Embed a single string, returning a 1-D float32 vector.

        Returned 1-D (not ``(1, dim)``) because the caller is almost always a
        similarity computation that wants to dot it against a matrix.
        """
        return self.embed([text])[0]

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity between two strings, computed on normalised output.

        Used by the deterministic answer evaluator, which needs a scalar
        grounding score and no stored vectors.
        """
        pair = self.embed([left, right])
        return float(np.dot(pair[0], pair[1]))

    def warmup(self) -> int:
        """Force the checkpoint to load and return its dimension.

        Exists so the app lifespan (or a warm script) can pay the download cost
        at start-up instead of on the first user request. Failure is logged, not
        raised: a retriever that cannot warm up still works, just slower.
        """
        try:
            return self.dimension
        except Exception as exc:
            logger.warning(
                "ml.embeddings.warmup_failed",
                model=self._model_name,
                error=str(exc),
            )
            return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        """Return the loaded model, loading it once under a lock if needed."""
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:  # another thread won the race
                return self._model

            # ~90 MB download for all-MiniLM-L6-v2, into the shared HuggingFace
            # cache. It happens here, on first use, and never at import.
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info(
                "ml.embeddings.loading_model",
                model=self._model_name,
                device=self._device,
            )
            model = SentenceTransformer(self._model_name, device=self._device)
            dimension = model.get_sentence_embedding_dimension()
            if not dimension:
                raise RuntimeError(
                    f"Embedding model {self._model_name!r} reported no dimension"
                )

            self._model = model
            self._dimension = int(dimension)
            logger.info(
                "ml.embeddings.model_ready",
                model=self._model_name,
                device=self._device,
                dimension=self._dimension,
            )
            return self._model


_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    """Return the process-wide :class:`Embedder`, creating it on first call.

    One instance per process is deliberate: the checkpoint occupies real memory
    and constructing a second ``SentenceTransformer`` would duplicate it. Tests
    that change ``settings.embedding_model`` should call
    :func:`reset_embedder` first.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = Embedder()
        return _EMBEDDER


def reset_embedder() -> None:
    """Drop the cached instance so the next call re-reads settings.

    Only for tests and for the (rare) case of a settings reload; dropping a live
    embedder means the next request pays a fresh model load.
    """
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = None
