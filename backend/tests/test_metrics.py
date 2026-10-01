"""Information-retrieval metrics.

Every expected number below is computed by hand from the definition, not read
off a previous run. For NDCG the derivation is spelled out in full because it
is the one metric that is easy to get subtly wrong and hard to eyeball:

    DCG@k  = Σ gain_i / log2(i + 1),  i = 1..k
    IDCG@k = the same sum over the k most valuable relevant documents
    NDCG@k = DCG / IDCG

§35 forbids computing Precision/Recall/MRR with an LLM, and these tests exist
partly to prove the deterministic implementations are actually right -- an
"obviously correct" metric that is off by one rank is worse than no metric,
because it will be trusted.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    HybridScorer,
    aggregate_metrics,
    f1_at_k,
    hit_rate_at_k,
    mrr,
    metric_direction,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_one_of_two_slots_relevant(self) -> None:
        # Only "a" is relevant, and it sits at rank 1 of 2 -> 1/2
        assert precision_at_k(["a", "z"], {"a"}, 2) == 0.5

    def test_zero_precision(self) -> None:
        assert precision_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_denominator_is_k_not_the_number_returned(self) -> None:
        # K=5 but only 2 documents retrieved. Precision@k is |Rel ∩ top-k| / k
        # by the Järvelin & Kekäläinen definition, so returning less than k
        # *lowers* the score. Dividing by len(retrieved) instead would reward a
        # retriever that returns two documents instead of five.
        assert precision_at_k(["a", "z"], {"a"}, 5) == pytest.approx(1 / 5)

    def test_only_the_top_k_counts(self) -> None:
        # "a" is relevant but sits at rank 3, outside K=2.
        assert precision_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
        assert precision_at_k(["x", "y", "a"], {"a"}, 3) == pytest.approx(1 / 3)

    def test_duplicate_retrievals_count_once_each(self) -> None:
        # "a" appears twice. Only the first counts as a relevant slot, so a
        # degenerate retriever that echoes cannot reach 1.0 by repetition.
        # ("a","a","z") over K=3 -> 2 distinct relevant ("a" and "z") / 3.
        assert precision_at_k(["a", "a", "z"], {"a", "z"}, 3) == pytest.approx(2 / 3)

    def test_k_of_zero_is_zero_not_an_error(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_retrieval_is_zero(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c", "d"], {"a", "b"}, 4) == 1.0

    def test_half_recall(self) -> None:
        # found 1 of 2 relevant -> 1/2
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_zero_recall(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_recall_denominator_is_the_relevant_set_not_k(self) -> None:
        # 5 relevant, retrieved all 5 within K=5, but K=10 -- still 1.0.
        assert recall_at_k(["a", "b", "c", "d", "e"], {"a", "b", "c", "d", "e"}, 10) == 1.0

    def test_recall_exceeds_precision_when_k_is_generous(self) -> None:
        retrieved, relevant = ["a", "x", "y", "z"], {"a"}
        assert precision_at_k(retrieved, relevant, 4) == 0.25
        assert recall_at_k(retrieved, relevant, 4) == 1.0


class TestF1:
    def test_perfect(self) -> None:
        assert f1_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_harmonic_mean_of_the_two(self) -> None:
        # P = 1/2, R = 1/2 -> F1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert f1_at_k(["a", "z"], {"a", "b"}, 2) == pytest.approx(0.5)

    def test_zero_when_nothing_found(self) -> None:
        assert f1_at_k(["x"], {"a"}, 1) == 0.0

    def test_f1_sits_between_precision_and_recall(self) -> None:
        retrieved, relevant = ["a", "x", "y", "z"], {"a", "b"}
        p = precision_at_k(retrieved, relevant, 4)
        r = recall_at_k(retrieved, relevant, 4)
        f = f1_at_k(retrieved, relevant, 4)
        assert min(p, r) <= f <= max(p, r)


class TestReciprocalRankAndMRR:
    def test_relevant_at_rank_one(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0

    def test_relevant_at_rank_two(self) -> None:
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_relevant_at_rank_ten(self) -> None:
        # 9 irrelevant documents then "a": 1/10
        ranking = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "a"]
        assert reciprocal_rank(ranking, {"a"}) == pytest.approx(0.1)

    def test_never_relevant_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_ranking_is_zero(self) -> None:
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_relevant_set_is_empty_is_zero(self) -> None:
        assert reciprocal_rank(["a"], set()) == 0.0

    def test_repeated_document_counts_at_its_first_position(self) -> None:
        # "a" first appears at rank 3; the repeat at rank 4 is ignored.
        ranking = ["x", "y", "a", "a"]
        assert reciprocal_rank(ranking, {"a"}) == pytest.approx(1 / 3)

    def test_mrr_of_a_single_ranking_equals_its_reciprocal_rank(self) -> None:
        assert mrr([["x", "a", "y"]], {"a"}) == pytest.approx(0.5)

    def test_mrr_averages_across_queries(self) -> None:
        # (1/1 + 1/2 + 0/3) / 3 = 0.5
        assert mrr([["a"], ["x", "a"], ["x", "y", "z"]], {"a"}) == pytest.approx(0.5)

    def test_mrr_of_no_queries_is_zero(self) -> None:
        assert mrr([], {"a"}) == 0.0

    def test_mrr_stays_in_the_unit_interval(self) -> None:
        rankings = [["a"], ["b"], ["x", "y"], []]
        assert 0.0 <= mrr(rankings, {"a", "b"}) <= 1.0


class TestNDCG:
    def test_perfect_binary_ordering_is_one(self) -> None:
        # Retrieved in the same order as relevant, so DCG == IDCG.
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_two_item_reversal_is_not_penalised(self) -> None:
        # Worth pinning down, because it looks like a bug and is not. With
        # binary gain and exactly two relevant documents, both permutations
        # place one item at rank 1 and one at rank 2, so DCG is identical and
        # NDCG is 1.0 either way. NDCG only discriminates ordering once three
        # or more positions are in play.
        #
        #   reversed DCG = 1/log2(3) + 1/log2(2) = 0.63093 + 1 = 1.63093
        #   IDCG         = 1/log2(2) + 1/log2(3) = 1       + 0.63093 = 1.63093
        assert ndcg_at_k(["b", "a"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_three_item_permutation_is_hand_computable(self) -> None:
        # Retrieved x,b,a against relevant {a,b}: one junk document at rank 1
        # pushes both real hits down by a position.
        #   DCG  = 1/log2(3) + 1/log2(4) = 0.630930 + 0.5 = 1.130930
        #   IDCG = 1/log2(2) + 1/log2(3) = 1        + 0.630930 = 1.630930
        #   NDCG = 1.130930 / 1.630930 = 0.693426
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (
            1 / math.log2(2) + 1 / math.log2(3)
        )
        assert expected == pytest.approx(0.693426, abs=1e-6)
        assert ndcg_at_k(["x", "b", "a"], {"a", "b"}, 3) == pytest.approx(expected)

    def test_both_relevant_found_at_ranks_1_and_2_is_perfect(self) -> None:
        # The complement of the case above: a leading junk document is what
        # costs score, not the mere presence of one.
        assert ndcg_at_k(["b", "a", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_three_item_ordering_matters_when_it_should(self) -> None:
        # The counterexample to the two-item case above: three relevant docs
        # and a gap in the middle must score worse than a clean ordering.
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)
        assert ndcg_at_k(["b", "x", "c"], {"a", "b", "c"}, 3) < 1.0

    def test_graded_relevance_prefers_the_high_grade_first(self) -> None:
        # Three documents, so the discount actually bites:
        #   best  [a(2), b(1), c(1)]: DCG = 2/1 + 1/1.58496 + 1/2 = 3.130930
        #   IDCG                      = same                         = 3.130930 -> 1.0
        #   worst [c(1), b(1), a(2)]: DCG = 1/1 + 1/1.58496 + 2/2 = 2.630930
        #   IDCG                      = 3.130930                     -> 0.840303
        grades = {"a": 2, "b": 1, "c": 1}
        best = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3, grades)
        worst = ndcg_at_k(["c", "b", "a"], {"a", "b", "c"}, 3, grades)
        assert best == pytest.approx(1.0)
        assert worst == pytest.approx(0.840303, abs=1e-5)
        assert best > worst, "graded NDCG must reward putting gain=2 first"

    def test_grade_magnitudes_change_the_score(self) -> None:
        # Same ranking, different grades -> a different number, which is the
        # whole point of accepting a grades mapping.
        assert ndcg_at_k(["c", "b", "a"], {"a", "b", "c"}, 3, {"a": 2, "b": 1, "c": 1}) != (
            ndcg_at_k(["c", "b", "a"], {"a", "b", "c"}, 3)
        )

    def test_graded_relevance_can_still_penalise(self) -> None:
        # gain=3 doc buried at rank 3 behind two irrelevant docs
        value = ndcg_at_k(["x", "y", "a"], {"a"}, 3, {"a": 3})
        assert value < 1.0
        assert value == pytest.approx((3 / math.log2(4)) / (3 / math.log2(2)))
    def test_repeating_a_hit_cannot_exceed_the_ideal(self) -> None:
        # Without first-appearance dedup, ten copies of "a" at ranks 1..10
        # would drive DCG above IDCG and the score above 1.0.
        spam = ["a"] * 10
        assert ndcg_at_k(spam, {"a"}, 10) <= 1.0
        assert ndcg_at_k(spam, {"a"}, 10) == pytest.approx(1.0)

    def test_degenerate_inputs_are_zero(self) -> None:
        assert ndcg_at_k([], {"a"}, 5) == 0.0
        assert ndcg_at_k(["a"], set(), 5) == 0.0
        assert ndcg_at_k(["a"], {"a"}, 0) == 0.0

    def test_is_reproducible_across_set_iteration_order(self) -> None:
        # IDCG is built by sorting the gains, so a differently-ordered set
        # (different hash seed, different platform) cannot change the answer.
        relevant = {"a", "b", "c", "d", "e"}
        first = ndcg_at_k(["e", "a", "d", "b", "c"], relevant, 5)
        second = ndcg_at_k(["c", "b", "d", "a", "e"], relevant, 5)
        assert first == second


class TestHitRate:
    def test_hit_when_relevant_is_present(self) -> None:
        assert hit_rate_at_k(["x", "a"], {"a"}, 2) is True

    def test_miss_when_absent(self) -> None:
        assert hit_rate_at_k(["x", "y"], {"a"}, 2) is False

    def test_returns_a_bool_not_a_number(self) -> None:
        # The per-query table renders this as a yes/no, so a 1.0 here would
        # leak into the UI as a score.
        assert isinstance(hit_rate_at_k(["a"], {"a"}, 1), bool)

    def test_any_relevant_document_is_enough(self) -> None:
        # Unlike Precision@K and NDCG@K, Hit Rate@K asks only whether *one*
        # relevant document appears in the window -- it does not require the
        # whole relevant set to be found.
        assert hit_rate_at_k(["a"], {"a", "b", "c"}, 1) is True

    def test_degenerate_inputs_are_misses_not_errors(self) -> None:
        assert hit_rate_at_k([], {"a"}, 5) is False
        assert hit_rate_at_k(["a"], set(), 5) is False
        assert hit_rate_at_k(["a"], {"a"}, 0) is False


class TestAggregateMetrics:
    """Averaging across queries, with the shape the storage layer depends on."""

    def _rows(self) -> list[dict[str, float]]:
        return [
            {"precision": 1.0, "recall": 1.0, "f1": 1.0, "reciprocal_rank": 1.0,
             "ndcg": 1.0, "hit": 1.0, "num_documents": 5},
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "reciprocal_rank": 0.0,
             "ndcg": 0.0, "hit": 0.0, "num_documents": 5},
        ]

    def test_means_over_all_queries(self) -> None:
        out = aggregate_metrics(self._rows(), k=5)
        for field in ("precision", "recall", "f1", "reciprocal_rank", "ndcg", "hit"):
            assert out[field] == pytest.approx(0.5), field

    def test_reports_the_query_count(self) -> None:
        assert aggregate_metrics(self._rows(), k=5)["count"] == 2

    def test_keys_are_sorted_so_two_runs_are_byte_identical(self) -> None:
        # Stored metrics blobs are compared across runs to detect regressions.
        # Non-deterministic key order would make every comparison a diff.
        keys = list(aggregate_metrics(self._rows(), k=5))
        assert [k for k in keys if k != "count"] == sorted(k for k in keys if k != "count")

    def test_bounded_metrics_are_clamped(self) -> None:
        # A key listed in BOUNDED_METRICS is a ratio and cannot exceed 1.0.
        # Without the clamp a buggy per-query value would propagate into the
        # dashboard as a "140% precision" card.
        rows = [{"precision": 4.0}, {"precision": 0.0}]
        # The mean is 2.0, which is not a possible ratio, so it clamps to 1.0
        # rather than surfacing as a "200% precision" card.
        assert aggregate_metrics(rows, k=1)["precision"] == 1.0

    def test_unbounded_metrics_are_not_clamped(self) -> None:
        # Clamping a latency mean would corrupt it -- 3000 ms is a real
        # measurement, not an out-of-range ratio.
        rows = [{"avg_latency_ms": 3000.0}, {"avg_latency_ms": 1000.0}]
        assert aggregate_metrics(rows, k=1)["avg_latency_ms"] == pytest.approx(2000.0)

    def test_nan_and_inf_values_are_dropped_not_propagated(self) -> None:
        rows = [{"recall": float("nan")}, {"recall": 1.0}, {"recall": float("inf")}]
        out = aggregate_metrics(rows, k=5)
        # Only the one usable value contributes, and it does not poison the mean.
        assert out["recall"] == pytest.approx(1.0)
        assert math.isfinite(out["recall"])

    def test_a_key_with_no_usable_value_becomes_zero(self) -> None:
        # The result shape must stay stable for the schema that consumes it.
        out = aggregate_metrics([{"recall": float("nan")}], k=5)
        assert out["recall"] == 0.0

    def test_non_dict_rows_are_skipped(self) -> None:
        out = aggregate_metrics([{"recall": 1.0}, "not-a-row", None], k=5)
        assert out["recall"] == pytest.approx(1.0)
        assert out["count"] == 3, "count reports rows supplied, not rows usable"

    def test_empty_input_reports_zero_queries(self) -> None:
        out = aggregate_metrics([], k=5)
        assert out == {"count": 0}

    def test_every_value_is_finite(self) -> None:
        out = aggregate_metrics(self._rows(), k=5)
        for key, value in out.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} is {value}"


class TestMetricDirection:
    """Regression detection needs to know which way is 'better'."""

    def test_higher_is_better_metrics(self) -> None:
        for metric in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k",
                       "f1_at_k", "faithfulness", "citation_coverage",
                       "token_efficiency", "retrieval_score"):
            assert metric_direction(metric) == "higher_is_better", metric

    def test_lower_is_better_metrics(self) -> None:
        for metric in ("estimated_cost", "avg_latency_ms", "p95_latency_ms",
                       "p99_latency_ms", "error_rate", "duration_ms",
                       "total_cost", "wasted_tokens", "duplicate_ratio"):
            assert metric_direction(metric) == "lower_is_better", metric

    def test_short_aliases_resolve_the_same_way(self) -> None:
        for alias, full in (("precision", "precision_at_k"), ("recall", "recall_at_k"),
                            ("ndcg", "ndcg_at_k"), ("rr", "reciprocal_rank")):
            assert metric_direction(alias) == metric_direction(full)

    def test_unknown_metric_is_neutral(self) -> None:
        # Guessing a direction for an unknown metric would silently invert a
        # regression, so the honest answer is "we do not know".
        assert metric_direction("something_new") == "neutral"
        assert metric_direction("") == "neutral"


class TestReciprocalRankFusion:
    def test_agreement_between_rankings_promotes_a_document(self) -> None:
        bm25, dense = ["a", "b", "c"], ["a", "d", "e"]
        scores = HybridScorer.rrf([bm25, dense], k=60)
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["d"]

    def test_a_document_in_one_ranking_only_still_scores(self) -> None:
        scores = HybridScorer.rrf([["a"], ["z"]], k=60)
        assert scores["a"] > 0.0
        assert scores["z"] > 0.0

    def test_scores_sum_correctly_for_a_single_ranking(self) -> None:
        scores = HybridScorer.rrf([["a", "b"]], k=60)
        # Both at rank 1 and 2 in one list: 1/(60+1) and 1/(60+2).
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)

    def test_empty_rankings_produce_no_scores(self) -> None:
        assert HybridScorer.rrf([], k=60) == {}
        assert HybridScorer.rrf([[], []], k=60) == {}

    def test_different_k_changes_the_flattening(self) -> None:
        # A larger k flattens the contribution difference between rank 1 and
        # rank 2. Asserting this pins the parameter's actual effect.
        sharp = HybridScorer.rrf([["a", "b"]], k=1)
        flat = HybridScorer.rrf([["a", "b"]], k=1000)
        assert sharp["a"] - sharp["b"] > flat["a"] - flat["b"]
