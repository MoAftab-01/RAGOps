"""Cross-encoder reranking.

Bi-encoders (the embedder) score a query and a document independently, which is
what makes them fast enough to search a whole corpus but also why they miss
fine-grained relevance. A cross-encoder reads the query and the passage
*together*, so it is markedly better at the final ordering — at the cost of one
forward pass per candidate, which is why it runs on the top-N survivors of
hybrid retrieval and not on the whole corpus.

The checkpoint (``cross-encoder/ms-marco-MiniLM-L-6-v2``) is roughly 90 MB and
is downloaded on the first :meth:`CrossEncoderReranker.rerank` call, never at
import time.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Final

from app.config import settings
from app.core.logging import get_logger
from app.ml.embeddings import resolve_device

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = ["CrossEncoderReranker", "get_reranker", "reset_reranker"]

_LOAD_LOCK: Final = threading.Lock()


class CrossEncoderReranker:
    """Cross-encoder reranker with a hard on/off switch.

    ``enabled`` is honoured before the model is ever touched, so turning
    reranking off in settings makes the *first* request after a restart cheaper
    too — not just every request after the model happened to load. That matters
    in CI and in tests, where an accidental download would be a slow surprise.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        enabled: bool | None = None,
        top_n: int | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.reranker_model
        self._enabled = settings.reranker_enabled if enabled is None else enabled
        self._top_n = max(1, top_n or settings.reranker_top_n)
        self._device = resolve_device(device)
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether reranking is active. Read without loading the model."""
        return self._enabled

    @property
    def model_name(self) -> str:
        """The checkpoint that will be (or was) loaded."""
        return self._model_name

    @property
    def device(self) -> str:
        """Resolved torch device for the cross-encoder."""
        return self._device

    @property
    def top_n(self) -> int:
        """Maximum number of candidates scored per call."""
        return self._top_n

    @property
    def is_loaded(self) -> bool:
        """Whether the checkpoint is resident. Safe before first use."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def rerank(
        self, query: str, docs: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Re-score ``(doc_id, text)`` pairs against ``query``, best first.

        When disabled the input order is returned with a ``0.0`` score instead of
        an empty list. A passthrough that *drops* the candidates would silently
        truncate the result set to nothing; a passthrough that preserves them
        with a neutral score keeps the pipeline shape identical whether or not
        the reranker is switched on, which is what makes an A/B comparison of
        ``reranker_enabled`` valid.

        Only the first ``settings.reranker_top_n`` candidates are scored, and
        anything past that is appended unchanged at the end of the list so no
        candidate is ever lost.
        """
        if not docs:
            return []

        if not self._enabled:
            logger.debug("ml.reranker.disabled_passthrough", candidates=len(docs))
            return [(doc_id, 0.0) for doc_id, _text in docs]

        window = docs[: self._top_n]
        tail = docs[self._top_n :]

        model = self._ensure_model()
        # CrossEncoder.predict takes a list of (query, passage) pairs and runs
        # them as one batch — a per-pair call would be pure overhead.
        pairs: list[tuple[str, str]] = [(query, text or " ") for _doc_id, text in window]
        raw: NDArray[Any] = model.predict(pairs, show_progress_bar=False)

        scores = [float(value) for value in list(raw)]
        reranked = sorted(
            zip(window, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        results: list[tuple[str, float]] = [(doc_id, score) for (doc_id, _t), score in reranked]

        if tail:
            # Beyond the window the reranker has no opinion, so those candidates
            # keep their incoming order at a neutral score rather than being
            # treated as scored-but-worst.
            results.extend((doc_id, 0.0) for doc_id, _text in tail)
        return results

    def warmup(self) -> bool:
        """Load the checkpoint ahead of time. Returns whether it is available.

        Like the embedder's warmup, failure is logged rather than raised: a
        missing reranker degrades retrieval quality, it does not break the
        service.
        """
        if not self._enabled:
            return False
        try:
            self._ensure_model()
            return True
        except Exception as exc:
            logger.warning(
                "ml.reranker.warmup_failed", model=self._model_name, error=str(exc)
            )
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        """Return the loaded CrossEncoder, loading it once under a lock."""
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            # ~90 MB download for ms-marco-MiniLM-L-6-v2, on first use only.
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info(
                "ml.reranker.loading_model",
                model=self._model_name,
                device=self._device,
            )
            model = CrossEncoder(self._model_name, device=self._device)
            self._model = model
            logger.info("ml.reranker.model_ready", model=self._model_name)
            return self._model


_RERANKER: CrossEncoderReranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    """Return the process-wide reranker, created on first call.

    Same reasoning as :func:`app.ml.embeddings.get_embedder`: one loaded
    checkpoint per process. Creation here does not load the model, so this is
    safe to call at startup to inspect ``enabled``.
    """
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    with _RERANKER_LOCK:
        if _RERANKER is None:
            _RERANKER = CrossEncoderReranker()
        return _RERANKER


def reset_reranker() -> None:
    """Drop the cached instance so the next call re-reads settings."""
    global _RERANKER
    with _RERANKER_LOCK:
        _RERANKER = None
