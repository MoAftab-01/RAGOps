"""Deterministic IR metrics — the single source of truth for RAGOps quality numbers.

Every quality figure RAGOps displays is produced here. This is deliberately the
most boring file in the repository: pure functions, no I/O, no database, no
network, no model, no randomness, no clock, and no import from ``app.*``. Two
consequences are load-bearing:

* the numbers are reproducible — the same retrieval result always yields the
  same value, so a regression detected here is a real change in retrieval
  rather than measurement noise; and
* they can be checked against arithmetic done on paper, which is why
  ``backend/tests/test_metrics.py`` states each expected value in a comment.

Conventions
-----------
Every function takes ``retrieved``: an ordered list of document ids, most
relevant first, so rank 1 is ``retrieved[0]``. Only the first ``k`` entries
count. Results are clamped to ``[0.0, 1.0]``. An empty ``relevant`` set or an
empty ``retrieved`` list yields ``0.0`` — these functions are *total*: they
never raise and never return NaN, because a missing measurement must surface as
"0.0 / no data", not as an exception in the middle of a dashboard query.

Definitions
-----------
``precision_at_k``
    ``|Rel ∩ top-k| / k``. Järvelin, K. & Kekäläinen, T. (2002), "Cumulated
    gain-based evaluation of IR techniques", ACM TOIS 20(4):422-446, which
    defines Precision@k and Recall@k as originally standardised. The
    denominator is ``k`` even when fewer than ``k`` documents came back: a
    system that returns three documents for a request admitting ten really is
    less precise at 10, and normalising by the number returned would make
    precision incomparable between systems.
``recall_at_k``
    ``|Rel ∩ top-k| / |Rel|`` — same source. Separates "retrieved the wrong
    things" from "missed the right things", which precision alone conflates.
``f1_at_k``
    ``2PR / (P + R)``, the harmonic mean of the two, and ``0.0`` when
    ``P + R == 0``. van Rijsbergen, C. J. (1979), *Information Retrieval*
    (2nd ed.), Butterworths.
``reciprocal_rank``
    ``1 / rank`` of the first relevant document in the full ranking, ``0.0``
    when there is none. Järvelin & Kekäläinen (2002); it is the per-query term
    that :func:`mrr` averages.
``mrr``
    Mean Reciprocal Rank: the mean of :func:`reciprocal_rank` across queries.
    Voorhees, E. M. (2001), "The evolution of the TREC question answering
    task", SIGIR Forum 25(1):10-15. Given a single ranking it is identical to
    :func:`reciprocal_rank`; given several it is their mean.
``ndcg_at_k``
    ``DCG@k / IDCG@k`` where ``DCG@k = Σ gain_i / log2(i + 1)`` for ``i = 1..k``
    over the top-k window, and ``IDCG@k`` is that same sum over the ``k`` most
    valuable relevant documents available. Järvelin & Kekäläinen (2002) §5
    introduces both the log2 discount and the ideal-ranking normaliser (NDCG),
    and reports the linear-gain form used here alongside their exponential
    ``2^rel - 1`` form; the two rank systems identically when gains are binary.
    A document contributes gain only when it is in ``relevant``; with ``grades``
    the gain is ``grades.get(id, 1)`` clamped at zero, otherwise it is 1.
    Returns ``0.0`` when ``IDCG == 0`` — nothing relevant was available, or
    every relevant document was graded 0 — never a division by zero.
``hit_rate_at_k``
    ``True`` when the top ``k`` contains at least one relevant document. The
    most robust metric here: it needs no judgement about *how* relevant
    anything was, which makes it the right choice when the grader can only say
    yes or no. Used as "hit ratio" throughout recommender evaluation, e.g.
    Castells, S. et al. (2011), "Novelty and diversity metrics for recommender
    systems", ECIR.

A metric that returns 0.0 for a degenerate input and one that returns 0.0 for a
genuinely bad retrieval are indistinguishable. That is intentional: the number
of queries evaluated and the number that retrieved nothing at all are reported
separately (``num_queries``, ``zero_result_rate``), so a real 0.0 is never
confused with "there was nothing to measure".

Duplicate document ids
----------------------
A retriever that returns the same document twice must not be able to inflate a
score, so every function here counts a relevant document at its *first*
occurrence in the ranking and ignores later repeats. Precision and NDCG are
then bounded by 1.0 by construction rather than by the clamp — without this,
``["a", "a", "b"]`` with ``{"a", "b"}`` relevant would score NDCG > 1.

NaN and infinity
----------------
:func:`aggregate_metrics` and the scalar helpers treat NaN and ±inf as missing
data, never as values. A NaN that reaches the dashboard renders as a broken
chart; a 0.0 with a warning in the log is a fact.

Why nothing here logs on the happy path
---------------------------------------
These functions are pure and total, so there is no error channel to report, and
a metrics module that logs on every call is one nobody can call in a loop. The
one place a value is genuinely discarded — a non-finite or non-numeric entry in
a per-query dict, or a nonsensical fusion parameter — is reported through
:func:`_log_defensive`, which imports the project's structured logger *lazily
and defensively*. That keeps ``import app.evaluation.metrics`` free of any
``app.*`` dependency, so this file also stays usable standalone (vendored into a
benchmark script, for instance). If that import ever fails, the metric is still
computed; only the warning is lost.

Aggregation
-----------
:func:`aggregate_metrics` means each numeric key across the per-query dicts.
Only the bounded ratio metrics listed in ``BOUNDED_METRICS`` are clamped to
``[0.0, 1.0]``: latency and cost means are *supposed* to exceed 1.0, and
clamping them would silently destroy the numbers this project exists to report.
``k`` is part of the published signature but is not echoed in the result — the
means are already the values at the caller's ``k``, and callers that need it on
the record carry it themselves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

__all__ = [
    "BOUNDED_METRICS",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "METRIC_DIRECTIONS",
    "NEUTRAL",
    "HybridScorer",
    "aggregate_metrics",
    "f1_at_k",
    "hit_rate_at_k",
    "metric_direction",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "recall_at_k",
]

# ---------------------------------------------------------------------------
# Direction vocabulary, shared with app.evaluation.regression
# ---------------------------------------------------------------------------

HIGHER_IS_BETTER: Final[str] = "higher_is_better"
LOWER_IS_BETTER: Final[str] = "lower_is_better"
# Returned for a metric we have no opinion about. Regression detection must not
# invent a direction: guessing wrong flags a healthy metric as a regression.
NEUTRAL: Final[str] = "neutral"


# ---------------------------------------------------------------------------
# Internal helpers — coercion, clamping, slicing
# ---------------------------------------------------------------------------


def _log_defensive(event: str, **fields: Any) -> None:
    """Report a discarded or coerced input without ever breaking a metric.

    The structured logger is imported here rather than at module scope so that
    importing this file pulls in nothing from ``app.*``. A failure of that
    import costs a log line, never a number.
    """
    try:
        from app.core.logging import get_logger

        payload: dict[str, Any] = {"event": event, **fields}
        get_logger("ragops.evaluation.metrics").warning("metrics.defensive", **payload)
    except Exception:  # noqa: BLE001 - a logger must never break a metric
        return


def _as_finite_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, or ``None`` when it is not usable data.

    ``bool`` is accepted deliberately: a per-query hit flag is a real 1/0
    observation, not missing data. Anything else that is not a real number —
    ``None``, NaN, ±inf, a string, a missing value — comes back as ``None`` so
    the caller can skip it instead of poisoning a mean.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        candidate = float(value)
    else:
        # Covers Decimal, numpy scalars and numeric strings arriving from JSON.
        try:
            candidate = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return candidate if math.isfinite(candidate) else None


def _clamp01(value: float) -> float:
    """Clamp to ``[0.0, 1.0]``, mapping non-finite input to ``0.0``.

    The ``isfinite`` check must come first: every comparison against NaN is
    False, so a naive ``min``/``max`` clamp can return NaN.
    """
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _top_k(retrieved: Sequence[str], k: int) -> tuple[str, ...]:
    """The prefix of the ranking that counts. A non-positive ``k`` is empty."""
    if k <= 0:
        return ()
    return tuple(retrieved[:k])


def _first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant document, or ``None`` if there is none."""
    if not relevant:
        return None
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return rank
    return None


def _relevant_in_top_k(retrieved: Sequence[str], relevant: set[str], k: int) -> int:
    """Count of *distinct* relevant documents inside the top-k window.

    Deduping is what keeps Precision@k and NDCG@k at or below 1.0 when a
    retriever repeats a document instead of returning new ones.
    """
    seen: set[str] = set()
    for doc_id in _top_k(retrieved, k):
        if doc_id in relevant and doc_id not in seen:
            seen.add(doc_id)
    return len(seen)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` slots holding a relevant document.

    ``|Rel ∩ top-k| / k`` — divided by ``k``, not by the number of documents
    actually returned, so returning less than ``k`` documents lowers the score
    exactly as the Järvelin & Kekäläinen (2002) definition requires.
    """
    if k <= 0 or not relevant or not retrieved:
        return 0.0
    return _clamp01(_relevant_in_top_k(retrieved, relevant, k) / k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents found within the top ``k``.

    ``|Rel ∩ top-k| / |Rel|``. Use this, not precision, to answer "did the
    retriever find the right things at all".
    """
    if k <= 0 or not relevant or not retrieved:
        return 0.0
    return _clamp01(_relevant_in_top_k(retrieved, relevant, k) / len(relevant))


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of :func:`precision_at_k` and :func:`recall_at_k`.

    ``2PR / (P + R)``, and ``0.0`` when ``P + R == 0`` — which happens for an
    empty relevant set, an empty ranking, or a ranking that found nothing.
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    total = precision + recall
    if total <= 0.0:
        return 0.0
    return _clamp01((2.0 * precision * recall) / total)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant document, or ``0.0`` if there is none.

    Rank 1 gives 1.0, rank 2 gives 0.5, and so on. This is the per-query term
    that :func:`mrr` averages, and the reason MRR rewards *where* the first hit
    landed rather than how many hits there were.
    """
    rank = _first_relevant_rank(retrieved, relevant)
    if rank is None:
        return 0.0
    return _clamp01(1.0 / rank)


def _as_ranking_list(retrieved: Sequence[str] | Sequence[Sequence[str]]) -> list[list[str]]:
    """Accept either one ranking or a batch of them, so MRR is computable from both.

    MRR is by definition a mean across queries, but the published signature is
    ``mrr(retrieved, relevant)`` and the evaluation layer scores one query at a
    time. A bare list of document ids is therefore treated as a single ranking
    (for which MRR *is* the reciprocal rank) and a list of rankings as a batch.
    """
    if not retrieved:
        return []
    raw: Any = retrieved
    if isinstance(raw[0], str):
        return [[str(doc_id) for doc_id in raw]]
    return [[str(doc_id) for doc_id in ranking] for ranking in raw]


def mrr(retrieved: list[str] | Sequence[Sequence[str]], relevant: set[str]) -> float:
    """Mean Reciprocal Rank (Voorhees 2001).

    Pass one ranking and this equals :func:`reciprocal_rank`; pass several
    rankings for the same query and it is their mean, which is the quantity the
    TREC question-answering task originally defined.
    """
    rankings = _as_ranking_list(retrieved)
    if not rankings:
        return 0.0
    total = sum(reciprocal_rank(ranking, relevant) for ranking in rankings)
    return _clamp01(total / len(rankings))


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
    grades: dict[str, int] | None = None,
) -> float:
    """Normalised discounted cumulative gain at ``k``.

    ``DCG@k = Σ gain_i / log2(i + 1)`` for ``i = 1..k`` over the top-k window;
    ``IDCG@k`` is the same sum over the ``k`` most valuable relevant documents
    available, which is what makes the score 1.0 for a perfect ordering.

    Gain is ``grades.get(id, 1)`` when ``grades`` is supplied (floored at zero)
    and ``1`` otherwise; a document outside ``relevant`` contributes nothing.
    Each document is credited at its first appearance only, so repeating a hit
    cannot push DCG above IDCG.
    """

    def gain(doc_id: str) -> float:
        if doc_id not in relevant:
            return 0.0
        if grades is None:
            return 1.0
        return float(max(0, grades.get(doc_id, 1)))

    if k <= 0 or not relevant or not retrieved:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(_top_k(retrieved, k), start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        value = gain(doc_id)
        if value > 0.0:
            dcg += value / math.log2(rank + 1)

    # Sorting makes the sum independent of set iteration order, so the result is
    # bit-for-bit reproducible across processes.
    ideal_gains = sorted((gain(doc_id) for doc_id in relevant), reverse=True)[:k]
    idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal_gains, 1))
    if idcg <= 0.0:
        return 0.0
    return _clamp01(dcg / idcg)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> bool:
    """Whether the top ``k`` contains at least one relevant document.

    Returned as a ``bool``, not a float, because it is a yes/no observation and
    the per-query table in the UI shows it as one. :func:`aggregate_metrics`
    averages it like any other number, giving a hit rate in ``[0.0, 1.0]``.
    """
    if k <= 0 or not relevant or not retrieved:
        return False
    return any(doc_id in relevant for doc_id in _top_k(retrieved, k))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

#: Metrics that are ratios and therefore bounded by 1.0. Only these are clamped
#: by :func:`aggregate_metrics`; clamping a latency or cost mean would corrupt it.
BOUNDED_METRICS: Final[frozenset[str]] = frozenset(
    {
        "precision",
        "precision_at_k",
        "recall",
        "recall_at_k",
        "f1",
        "f1_at_k",
        "mrr",
        "rr",
        "reciprocal_rank",
        "ndcg",
        "ndcg_at_k",
        "hit_rate",
        "hit_rate_at_k",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "citation_coverage",
        "unsupported_claim_ratio",
        "zero_result_rate",
        "error_rate",
        "duplicate_ratio",
    }
)


def aggregate_metrics(per_query: list[dict[str, float]], k: int) -> dict[str, float]:
    """Mean each numeric key across ``per_query`` and add ``count``.

    Keys are emitted in sorted order so two runs of the same evaluation produce
    identical dictionaries, which keeps stored ``metrics`` blobs comparable.
    Non-numeric, NaN and infinite values are dropped from their key's mean and
    reported; a key with no usable value at all comes back ``0.0`` with a
    warning so the result shape stays stable for the schema that consumes it.

    ``k`` is accepted because the published signature carries it, and is used
    only to tag the warnings; it is not echoed in the result, since the means
    are already the values at the caller's ``k``.
    """
    if not per_query:
        return {"count": 0}

    seen_keys: set[str] = set()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}

    for row in per_query:
        if not isinstance(row, dict):
            _log_defensive("aggregate.row_not_a_mapping", k=k, row_type=type(row).__name__)
            continue
        for key, raw in row.items():
            seen_keys.add(key)
            value = _as_finite_float(raw)
            if value is None:
                dropped[key] = dropped.get(key, 0) + 1
                continue
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    result: dict[str, float] = {}
    for key in sorted(seen_keys):
        kept = counts.get(key, 0)
        if kept == 0:
            _log_defensive(
                "aggregate.no_usable_values",
                metric=key,
                k=k,
                rows=len(per_query),
            )
            result[key] = 0.0
            continue
        if key in dropped:
            _log_defensive(
                "aggregate.values_dropped",
                metric=key,
                k=k,
                dropped=dropped[key],
                kept=kept,
            )
        mean = totals[key] / kept
        result[key] = _clamp01(mean) if key in BOUNDED_METRICS else mean

    result["count"] = len(per_query)
    return result


# ---------------------------------------------------------------------------
# Metric directions — consumed by app.evaluation.regression
# ---------------------------------------------------------------------------

#: Which way is good for each metric. Regression detection cannot compare two
#: runs without this, and it must not guess: a rising cost is a regression while
#: a rising recall is progress, and both are the same sign change.
METRIC_DIRECTIONS: Final[dict[str, str]] = {
    # -- Retrieval and answer quality: bigger is better ---------------------
    "precision_at_k": HIGHER_IS_BETTER,
    "recall_at_k": HIGHER_IS_BETTER,
    "f1_at_k": HIGHER_IS_BETTER,
    "mrr": HIGHER_IS_BETTER,
    "ndcg_at_k": HIGHER_IS_BETTER,
    "hit_rate_at_k": HIGHER_IS_BETTER,
    # Common aliases so a run recorded under a short name still resolves.
    "precision": HIGHER_IS_BETTER,
    "recall": HIGHER_IS_BETTER,
    "f1": HIGHER_IS_BETTER,
    "rr": HIGHER_IS_BETTER,
    "reciprocal_rank": HIGHER_IS_BETTER,
    "ndcg": HIGHER_IS_BETTER,
    "hit_rate": HIGHER_IS_BETTER,
    "faithfulness": HIGHER_IS_BETTER,
    "answer_relevance": HIGHER_IS_BETTER,
    "context_relevance": HIGHER_IS_BETTER,
    "citation_coverage": HIGHER_IS_BETTER,
    "quality_score": HIGHER_IS_BETTER,
    "token_efficiency": HIGHER_IS_BETTER,
    "retrieval_score": HIGHER_IS_BETTER,
    # -- Cost, latency and volume: smaller is better ------------------------
    "estimated_cost": LOWER_IS_BETTER,
    "avg_cost_per_1k_tokens": LOWER_IS_BETTER,
    "avg_latency_ms": LOWER_IS_BETTER,
    "duration_ms": LOWER_IS_BETTER,
    "avg_tokens_per_request": LOWER_IS_BETTER,
    "cost_per_1k_tokens": LOWER_IS_BETTER,
    "cost_per_request": LOWER_IS_BETTER,
    "cost_per_user": LOWER_IS_BETTER,
    "cost_per_application": LOWER_IS_BETTER,
    "total_cost": LOWER_IS_BETTER,
    "p50_latency_ms": LOWER_IS_BETTER,
    "p95_latency_ms": LOWER_IS_BETTER,
    "p99_latency_ms": LOWER_IS_BETTER,
    "max_latency_ms": LOWER_IS_BETTER,
    "p95_duration_ms": LOWER_IS_BETTER,
    "unsupported_claim_ratio": LOWER_IS_BETTER,
    "zero_result_rate": LOWER_IS_BETTER,
    "error_rate": LOWER_IS_BETTER,
    "duplicate_ratio": LOWER_IS_BETTER,
    "potential_waste_pct": LOWER_IS_BETTER,
    "wasted_tokens": LOWER_IS_BETTER,
}


def metric_direction(metric: str) -> str:
    """Direction of ``metric``: higher, lower, or ``neutral`` if we have no opinion.

    Regression detection asks this about every metric that appears in both runs,
    including ones this file has never seen. Returning ``neutral`` for an
    unknown metric is the honest answer — inventing a direction would flag a
    healthy metric as a regression.
    """
    return METRIC_DIRECTIONS.get(metric, NEUTRAL)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class HybridScorer:
    """Deterministic Reciprocal Rank Fusion over several candidate rankings.

    RRF fuses ranked lists without requiring their scores to be commensurable,
    which is exactly the situation when combining BM25 (an unbounded
    IDF-weighted sum), dense vector search (a cosine, so bounded but on a
    different scale), and an optional cross-encoder logit. The only parameter
    is how sharply ranks are discounted:

        ``score(d) = Σ over rankings r of 1 / (k + rank_r(d))``

    with ``rank_r(d)`` 1-based. ``k = 60`` is the value from Cormack, G. V.,
    Clarke, C. L. A. & Buettcher, S. (2009), "Reciprocal rank fusion outperforms
    Condorcet and individual rank learning methods", SIGIR '09, 758-759, where
    it was found to be insensitive across roughly two orders of magnitude —
    the property that keeps the fusion stable when one retriever occasionally
    returns nothing at all.

    Only a document's first appearance within a ranking contributes: a repeat is
    not independent evidence, and counting it twice would let a retriever that
    echoes a hit out-rank one that surfaces something new.

    Scores are deliberately *not* clamped to ``[0.0, 1.0]``. They are small
    positive numbers (about ``1/k`` per list) meant to be summed and compared,
    not shown as a quality percentage.
    """

    __slots__ = ()

    @staticmethod
    def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
        """Fuse ``rankings`` and return ``{document_id: rrf_score}``.

        Empty rankings are skipped, so a retriever that found nothing
        contributes nothing instead of poisoning the result. A negative ``k`` is
        meaningless (it would divide by zero) and is coerced to 0 with a warning.
        """
        effective_k = k
        if effective_k < 0:
            _log_defensive("rrf.negative_k", k=k)
            effective_k = 0

        scores: dict[str, float] = {}
        for ranking in rankings:
            if not ranking:
                continue
            seen: set[str] = set()
            for index, doc_id in enumerate(ranking):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (effective_k + index + 1)
        return scores

    @staticmethod
    def rank(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
        """The fused ranking as ``(document_id, score)``, best first.

        Ties break on document id so the ordering is total and reproducible —
        an unstable fusion would make two identical runs look like a regression.
        """
        scores = HybridScorer.rrf(rankings, k)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
