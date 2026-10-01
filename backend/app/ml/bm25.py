"""Lexical retrieval over BM25Okapi.

Dense vectors are good at paraphrase and bad at exact identifiers: a product
code, a function name, an error string. BM25 covers that gap, and RRF fusion
downstream uses the two rankings together.

The tokeniser here is intentionally the simplest thing that works — lowercase
word characters — rather than a stemmer or a language-specific analyser. A
strawman analyser would be easy to outsmart and would make scores depend on a
third-party download; what matters for the hybrid system is that *build* and
*search* tokenise identically, because any divergence silently zeroes out half
the corpus.
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, Final

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = ["BM25Index", "tokenize"]

# Words and numbers, lowercased. ``\w`` with re.UNICODE keeps non-ASCII words
# (accented European text, Devanagari) intact, which matters for a knowledge
# base that is not strictly English.
_TOKEN_RE: Final = re.compile(r"\w+", re.UNICODE)

# BM25's IDF term can go negative for a word present in more than half the
# corpus. A negative score is not "irrelevant", it is nonsense: it would let a
# document that matches everything sort *below* one that matches nothing. The
# floor keeps the ranking monotonic, so index 0 really is the best candidate.
_MIN_SCORE: Final = 0.0


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation, used for both indexing and querying.

    Kept as a module-level function (not a lambda inside the class) so the two
    call sites are provably the same code path.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A rebuildable BM25Okapi index over ``(id, text)`` pairs.

    Not thread-safe for concurrent ``build`` calls by design: an index is built
    once per ingestion and then read many times, and serialising writers behind
    a lock is far cheaper than copy-on-write semantics nobody needs. Readers
    only ever see a fully-built index because ``build`` swaps the reference
    under the lock rather than mutating in place.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._index: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        # The tokenised corpus, kept alongside the index so a negative score can
        # be resolved against what the document actually contains. See search().
        self._corpus: list[set[str]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_built(self) -> bool:
        """Whether the index holds documents. Guard for :meth:`search`."""
        return self._index is not None and bool(self._doc_ids)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> list[str]:
        """Copy of the indexed ids, in index order."""
        return list(self._doc_ids)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def build(self, documents: list[tuple[str, str]]) -> None:
        """(Re)build the index from ``(doc_id, text)`` pairs.

        An empty corpus is stored as *not built* rather than as an empty
        ``BM25Okapi``, which would divide by a zero corpus size on the first
        query. That turns "no documents" into a clean empty result instead of a
        divide-by-zero deep inside the library.

        Duplicate ids are kept: the caller owns id uniqueness (chunk ids are
        content-hashed), and silently dropping a duplicate here would hide a
        real ingestion bug.
        """
        if not documents:
            with self._lock:
                self._index = None
                self._doc_ids = []
                self._corpus = []
            logger.info("ml.bm25.build_empty")
            return

        corpus = [tokenize(text) for _doc_id, text in documents]
        index = BM25Okapi(corpus, k1=self._k1, b=self._b)

        with self._lock:
            self._index = index
            self._doc_ids = [doc_id for doc_id, _text in documents]
            self._corpus = [set(tokens) for tokens in corpus]

        logger.info("ml.bm25.built", documents=len(documents))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, score)`` pairs, best first.

        Scores are raw BM25 (not normalised) because RRF consumes *ranks*, not
        magnitudes — normalising would imply a comparability between BM25 and
        cosine scores that does not exist.

        Documents scoring at or below zero are dropped. Returning them would
        fill ``top_k`` with arbitrary tail content whenever the query shares no
        terms with the corpus, which reads as a confident answer when it is
        really "nothing matched".

        A score of zero is not always "no match". ``rank_bm25`` can zero out a
        document that plainly *contains* the query term, and it does so in two
        different ways that both only happen on small corpora:

        * **One document.** Every term has ``df = 1 = N``, so the IDF term
          ``log((N - df + 0.5) / (df + 0.5))`` is negative and the sole
          document scores below zero. Measured: ``[-0.2747]`` for "penguins".
        * **Two or three documents.** ``average_idf`` frequently sums to exactly
          ``0.0``, and ``rank_bm25`` then applies its negative-IDF correction as
          ``epsilon * average_idf``, which is ``0``. A document containing the
          term scores exactly ``0.0``. Measured on the two-document corpus in
          ``test_rag_pipeline.py``: raw scores ``[0. 0.]`` for "penguins".

        Both read to a caller as "your query matched nothing", which is a
        different and wrong answer. So a document at or below the floor is
        resolved by asking the tokenised corpus whether the document really
        contains any of the query terms: if it does it is kept, floored at
        zero, because it *is* a match and the only defect is in the score. If
        it does not, it is dropped -- a document sharing no terms with the
        query is noise, not a weak hit.
        """
        index = self._index
        if index is None or not self._doc_ids or top_k <= 0:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores: NDArray[np.float64] = index.get_scores(tokens)
        if scores.size == 0:
            return []

        # Partial selection rather than a full argsort: only top_k is wanted and
        # a corpus can be large.
        limit = min(top_k, scores.shape[0])
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)

        results: list[tuple[str, float]] = []
        for position in ranked:
            raw = float(scores[position])
            if raw <= _MIN_SCORE:
                # Above the floor, the score is trustworthy. At or below it, only
                # the tokenised corpus knows whether this is a real match.
                if not self._matches(tokens, int(position)):
                    continue
                raw = 0.0
            results.append((self._doc_ids[int(position)], raw))
        return results

    def score_all(self, query: str) -> dict[str, float]:
        """Every document's score for ``query``, keyed by id.

        RRF needs a *complete* ranking, including the documents that scored
        zero, because a document that both retrievers agree on at rank N is
        exactly the signal fusion is looking for. Trimming zeros before fusion
        would destroy that agreement.
        """
        index = self._index
        if index is None or not self._doc_ids:
            return {}

        tokens = tokenize(query)
        if not tokens:
            return {}

        scores: NDArray[np.float64] = index.get_scores(tokens)
        return {
            doc_id: max(0.0, float(score))
            for doc_id, score in zip(self._doc_ids, scores, strict=True)
        }

    def _matches(self, tokens: list[str], position: int) -> bool:
        """Whether the document at ``position`` contains *any* query term.

        Resolves the negative-IDF case above without trusting the score. "Any"
        rather than "all" because a document matching one of three query terms
        is still a document the query found, and dropping it would make a
        partial match indistinguishable from no match at all.
        """
        if position >= len(self._corpus):
            return False
        vocabulary = self._corpus[position]
        return any(token in vocabulary for token in tokens)

    def clear(self) -> None:
        """Drop the index so the object can be rebuilt from scratch."""
        with self._lock:
            self._index = None
            self._doc_ids = []
            self._corpus = []
