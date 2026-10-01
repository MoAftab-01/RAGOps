"""Regression detection between two recorded evaluation runs.

§35 lists *"fabricate root causes"* as a thing not to do, and §9 requires
regression detection that reports what changed without inventing why. The
behaviour under test is therefore mostly about **restraint**:

* a metric the platform has no direction for is reported but never flagged;
* a metric missing from one run is skipped, not treated as a drop to zero;
* the summary names movements and settings, and never links one to the other.

The arithmetic itself is checked against hand-computed percentages, because a
regression verdict built on a mis-signed percentage is worse than no verdict.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.regression import (
    DEFAULT_REGRESSION_THRESHOLD_PCT,
    compare_runs,
    config_differences,
    deltas,
    factual_changes,
    is_regression,
    numeric_metric,
    regressions_of,
    summarise,
)


def by_metric(rows: list) -> dict:
    """Index a delta list by metric name, the way the route body wants it."""
    return {row.metric: row for row in rows}


class TestNumericMetric:
    def test_numbers_pass_through(self) -> None:
        assert numeric_metric(0.5) == 0.5
        assert numeric_metric(3) == 3.0

    def test_booleans_are_not_numbers(self) -> None:
        # ``True`` is an int in Python. Reading a recorded flag as 1.0 would put
        # a phantom numeric delta into the comparison.
        assert numeric_metric(True) is None
        assert numeric_metric(False) is None

    def test_nested_mappings_are_skipped(self) -> None:
        # Runs store a ``by_k`` mapping beside the flat metrics.
        assert numeric_metric({"5": 0.4, "10": 0.5}) is None

    def test_none_and_strings_are_skipped(self) -> None:
        assert numeric_metric(None) is None
        assert numeric_metric("0.5") is None

    def test_nan_and_inf_are_skipped(self) -> None:
        # A NaN that reached the report would render as a blank card and poison
        # any downstream mean.
        assert numeric_metric(float("nan")) is None
        assert numeric_metric(math.inf) is None


class TestIsRegression:
    def test_a_drop_in_a_higher_is_better_metric_is_a_regression(self) -> None:
        assert is_regression("higher_is_better", -0.2, -20.0, 5.0) is True

    def test_a_rise_in_a_higher_is_better_metric_is_not(self) -> None:
        assert is_regression("higher_is_better", 0.2, 20.0, 5.0) is False

    def test_a_rise_in_a_lower_is_better_metric_is_a_regression(self) -> None:
        # Cost and latency both run this way, so a silent inversion here would
        # mark spending more as an improvement.
        assert is_regression("lower_is_better", 0.2, 20.0, 5.0) is True

    def test_a_drop_in_a_lower_is_better_metric_is_not(self) -> None:
        assert is_regression("lower_is_better", -0.2, -20.0, 5.0) is False

    def test_a_neutral_metric_is_never_flagged_either_way(self) -> None:
        # The whole point of a neutral direction: refusing to guess is what
        # stops a healthy metric being reported as broken.
        assert is_regression("neutral", -99.0, -5000.0, 5.0) is False
        assert is_regression("neutral", 99.0, 5000.0, 5.0) is False

    def test_movement_inside_the_threshold_is_not_a_regression(self) -> None:
        # 4% is below the default 5%.
        assert is_regression("higher_is_better", -0.04, -4.0, 5.0) is False

    def test_movement_exactly_at_the_threshold_is_not_a_regression(self) -> None:
        # The comparison is strict, so the threshold is the largest movement
        # still accepted. Off-by-one here is a coin flip, not a judgement.
        assert is_regression("higher_is_better", -0.05, -5.0, 5.0) is False
        assert is_regression("higher_is_better", -0.0501, -5.01, 5.0) is True

    def test_zero_threshold_flags_any_movement(self) -> None:
        assert is_regression("higher_is_better", -0.0001, -0.01, 0.0) is True

    def test_negative_threshold_is_read_as_a_magnitude(self) -> None:
        # Callers should pass a magnitude; a negative one must not invert the
        # comparison and flag every improvement.
        assert is_regression("higher_is_better", 0.2, 20.0, -5.0) is False
        assert is_regression("higher_is_better", -0.2, -20.0, -5.0) is True

    def test_drop_from_a_zero_baseline_is_judged_on_absolute_size(self) -> None:
        # There is no meaningful percentage change from zero, and refusing to
        # judge it would freeze a metric at the value it was introduced with.
        assert is_regression("higher_is_better", -0.001, None, 5.0) is True
        assert is_regression("higher_is_better", 0.001, None, 5.0) is False
        # Same zero baseline on a cost metric: only a rise is a regression.
        assert is_regression("lower_is_better", 0.001, None, 5.0) is False
        assert is_regression("lower_is_better", -0.001, None, 5.0) is False


class TestDeltas:
    def test_hand_computed_relative_change(self) -> None:
        # recall 0.80 -> 0.60 is a change of -0.20 on a baseline of 0.80,
        # which is -25% exactly. Not -20%, which is what reading the change as
        # a percentage would give.
        found = by_metric(deltas({"recall": 0.80}, {"recall": 0.60}))
        row = found["recall"]
        assert row.baseline == 0.80
        assert row.candidate == 0.60
        assert row.absolute_change == pytest.approx(-0.20)
        assert row.relative_change_pct == pytest.approx(-25.0)
        assert row.is_regression is True

    def test_percentages_are_computed_on_the_absolute_baseline(self) -> None:
        # -2 on a baseline of -100 is a 2% improvement in magnitude terms.
        found = by_metric(deltas({"x": -100.0}, {"x": -102.0}))
        assert found["x"].relative_change_pct == pytest.approx(-2.0)

    def test_a_metric_missing_from_the_candidate_is_skipped(self) -> None:
        # Treating the missing side as zero would register as a 100% collapse.
        found = deltas({"recall": 0.8, "mrr": 0.7}, {"recall": 0.8})
        assert [row.metric for row in found] == ["recall"]

    def test_a_metric_missing_from_the_baseline_is_skipped(self) -> None:
        found = deltas({"recall": 0.8}, {"recall": 0.8, "mrr": 0.7})
        assert [row.metric for row in found] == ["recall"]

    def test_non_numeric_metrics_are_skipped_not_zeroed(self) -> None:
        found = deltas(
            {"recall": 0.8, "by_k": {"5": 0.4}, "label": "run-a"},
            {"recall": 0.8, "by_k": {"5": 0.4}, "label": "run-b"},
        )
        assert [row.metric for row in found] == ["recall"]

    def test_output_is_sorted_for_a_stable_report(self) -> None:
        found = deltas(
            {"recall": 0.8, "mrr": 0.7, "precision": 0.6},
            {"recall": 0.8, "mrr": 0.7, "precision": 0.6},
        )
        assert [row.metric for row in found] == ["mrr", "precision", "recall"]

    def test_identical_runs_produce_no_regressions(self) -> None:
        found = deltas({"recall": 0.8, "cost": 1.0}, {"recall": 0.8, "cost": 1.0})
        assert found, "unchanged metrics must still be reported"
        assert regressions_of(found) == []

    def test_direction_is_carried_from_the_metrics_registry(self) -> None:
        found = by_metric(
            deltas(
                {"estimated_cost": 1.0, "p95_latency_ms": 100.0, "mystery": 1.0},
                {"estimated_cost": 2.0, "p95_latency_ms": 200.0, "mystery": 2.0},
            )
        )
        assert found["estimated_cost"].direction == "lower_is_better"
        assert found["p95_latency_ms"].direction == "lower_is_better"
        assert found["mystery"].direction == "neutral"

    def test_both_a_cost_rise_and_a_quality_drop_are_flagged(self) -> None:
        found = deltas(
            {"estimated_cost": 1.0, "recall": 0.80},
            {"estimated_cost": 2.0, "recall": 0.60},
        )
        flagged = {row.metric for row in regressions_of(found)}
        assert flagged == {"estimated_cost", "recall"}

    def test_a_custom_threshold_is_honoured(self) -> None:
        base, cand = {"recall": 1.0}, {"recall": 0.95}
        # -5% exactly: accepted at the default 5%, flagged at 1%.
        assert regressions_of(deltas(base, cand, 5.0)) == []
        assert [r.metric for r in regressions_of(deltas(base, cand, 1.0))] == ["recall"]

    def test_default_threshold_is_five_percent(self) -> None:
        assert DEFAULT_REGRESSION_THRESHOLD_PCT == 5.0


class TestConfigDifferences:
    def test_a_changed_setting_is_reported(self) -> None:
        found = config_differences({"rerank": True, "top_k": 5}, {"rerank": False, "top_k": 5})
        assert [(d.key, d.baseline_value, d.candidate_value) for d in found] == [
            ("rerank", True, False)
        ]

    def test_identical_configs_report_nothing(self) -> None:
        assert config_differences({"a": 1}, {"a": 1}) == []

    def test_a_key_missing_from_one_side_is_not_a_difference(self) -> None:
        # "the candidate did not record this" is not the same statement as "the
        # candidate set it differently".
        assert config_differences({"a": 1}, {}) == []
        assert config_differences({}, {"a": 1}) == []

    def test_nested_mappings_are_flattened_one_level(self) -> None:
        found = config_differences(
            {"retrieval": {"top_k": 5, "rerank": True}},
            {"retrieval": {"top_k": 10, "rerank": True}},
        )
        assert [(d.key, d.baseline_value, d.candidate_value) for d in found] == [
            ("retrieval.top_k", 5, 10)
        ]

    def test_a_mapping_against_a_scalar_is_a_difference_not_a_crash(self) -> None:
        found = config_differences({"a": {"x": 1}}, {"a": 5})
        assert [(d.key, d.baseline_value, d.candidate_value) for d in found] == [
            ("a", {"x": 1}, 5)
        ]

    def test_output_is_sorted(self) -> None:
        found = config_differences({"b": 1, "a": 1}, {"b": 2, "a": 2})
        assert [d.key for d in found] == ["a", "b"]

    def test_values_keep_their_types(self) -> None:
        # repr-preserving rendering matters downstream: False vs "False" vs 0
        # are different settings to whoever reads the report.
        found = config_differences({"a": None}, {"a": 0})
        assert found[0].baseline_value is None
        assert found[0].candidate_value == 0


class TestNoFabricatedCauses:
    """The wording of a report must not assert a causal link."""

    def test_change_sentences_state_the_setting_not_a_consequence(self) -> None:
        differences = config_differences({"rerank": True}, {"rerank": False})
        sentence = factual_changes(differences)[0]
        assert "rerank" in sentence and "True" in sentence and "False" in sentence
        banned = ("because", "caused", "due to", "which lowered", "resulted in", "led to")
        assert not any(word in sentence.lower() for word in banned), sentence

    def test_summary_names_metrics_and_settings_without_linkage(self) -> None:
        rows = deltas({"recall": 0.8}, {"recall": 0.5})
        flagged = regressions_of(rows)
        differences = config_differences({"rerank": True}, {"rerank": False})
        summary = summarise(flagged, differences)
        assert summary is not None
        assert "recall" in summary
        assert "1 recorded setting(s)" in summary
        banned = ("because", "caused", "due to", "explains")
        assert not any(word in summary.lower() for word in banned), summary

    def test_summary_is_none_when_nothing_regressed(self) -> None:
        rows = deltas({"recall": 0.8}, {"recall": 0.8})
        assert summarise(regressions_of(rows), []) is None

    def test_summary_omits_the_settings_clause_when_none_differ(self) -> None:
        rows = deltas({"recall": 0.8}, {"recall": 0.5})
        summary = summarise(regressions_of(rows), [])
        assert "setting" not in summary


class TestCompareRuns:
    def test_end_to_end_report_shape(self) -> None:
        report = compare_runs(
            {"metrics": {"recall": 0.80, "estimated_cost": 1.00}, "config": {"rerank": True}},
            {"metrics": {"recall": 0.60, "estimated_cost": 1.50}, "config": {"rerank": False}},
        )
        assert report["has_regression"] is True
        assert len(report["deltas"]) == 2
        assert len(report["config_differences"]) == 1
        assert report["contributing_config_changes"][0].startswith("Recorded setting")
        assert "recall" in (report["regression_summary"] or "")

    def test_an_improvement_is_not_a_regression(self) -> None:
        report = compare_runs(
            {"metrics": {"recall": 0.60}, "config": {}},
            {"metrics": {"recall": 0.80}, "config": {}},
        )
        assert report["has_regression"] is False
        assert report["regression_summary"] is None

    def test_missing_sections_default_to_empty(self) -> None:
        # The route always supplies both, but a hand-built comparison should not
        # need the caller to remember that.
        report = compare_runs({}, {})
        assert report["deltas"] == []
        assert report["config_differences"] == []
        assert report["has_regression"] is False

    def test_null_sections_are_treated_as_empty(self) -> None:
        report = compare_runs({"metrics": None, "config": None}, {"metrics": None, "config": None})
        assert report["has_regression"] is False

    def test_thresholds_flow_through(self) -> None:
        base = {"metrics": {"recall": 0.95}, "config": {}}
        assert compare_runs(base, base, 1.0)["has_regression"] is False
        assert compare_runs(base, base, 0.0)["has_regression"] is False

    def test_metrics_and_config_are_read_from_their_own_sections(self) -> None:
        # Reading the wrong section returns empty lists rather than raising, so
        # a report can look perfectly clean while comparing nothing. Assert the
        # counts, not just the verdict.
        report = compare_runs(
            {"metrics": {"recall": 0.80, "estimated_cost": 1.0}, "config": {"rerank": True}},
            {"metrics": {"recall": 0.60, "estimated_cost": 1.0}, "config": {"rerank": True}},
        )
        assert len(report["deltas"]) == 2
        assert [d.metric for d in report["deltas"]] == ["estimated_cost", "recall"]
        assert report["config_differences"] == []

    def test_a_metric_and_a_config_key_of_the_same_name_do_not_bleed(self) -> None:
        # "recall" as a metric and as a setting are different facts; merging the
        # two sections would let one stand in for the other.
        report = compare_runs(
            {"metrics": {"recall": 0.80}, "config": {"recall": 3}},
            {"metrics": {"recall": 0.60}, "config": {"recall": 5}},
        )
        assert [d.metric for d in report["deltas"]] == ["recall"]
        assert [d.key for d in report["config_differences"]] == ["recall"]
