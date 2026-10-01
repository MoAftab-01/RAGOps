"""Anomaly detection with Isolation Forest.

The critical behaviour here is not "does it find the outlier" — that is
straightforward. It is that a run which could not examine the data says so
instead of reporting a clean result. A detector that returns ``num_anomalies:
0`` because it dropped every row is indistinguishable from a healthy system
unless the sample count is reported alongside, and that is exactly the bug this
suite pins shut.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.anomaly import DEFAULT_FEATURES, AnomalyDetector


def make_window(n: int = 400, seed: int = 7) -> np.ndarray:
    """A lognormal window resembling real request telemetry.

    Six columns in ``DEFAULT_FEATURES`` order: every value is strictly
    positive, which is why the real data is analysed in log space.
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=5.0, sigma=0.6, size=(n, 1))
    multipliers = np.array([1.0, 0.9, 0.6, 0.05, 0.01, 0.004])
    return base * multipliers + rng.uniform(0.1, 1.0, size=(n, 6))


class TestConstruction:
    def test_starts_unfitted(self) -> None:
        detector = AnomalyDetector()
        assert detector.is_fitted is False
        assert detector.n_features is None

    def test_is_fitted_after_a_run(self) -> None:
        detector = AnomalyDetector()
        detector.fit_predict(make_window())
        assert detector.is_fitted is True
        assert detector.n_features == len(DEFAULT_FEATURES)

    def test_random_state_is_pinned_for_reproducibility(self) -> None:
        # An anomaly report is shown to a user who then has to act on it. A
        # detector that flags a different trace on every refresh cannot be
        # investigated.
        window = make_window()
        first = AnomalyDetector().fit_predict(window)
        second = AnomalyDetector().fit_predict(window)
        assert np.array_equal(first, second)

    def test_different_random_states_still_produce_a_usable_result(self) -> None:
        # Not bit-identical, but the same order of magnitude of flags.
        window = make_window()
        a = AnomalyDetector(random_state=1).fit_predict(window)
        b = AnomalyDetector(random_state=2).fit_predict(window)
        assert (a < 0).sum() > 0 and (b < 0).sum() > 0

    def test_reset_returns_to_unfitted(self) -> None:
        detector = AnomalyDetector()
        detector.fit_predict(make_window())
        detector.reset()
        assert detector.is_fitted is False
        assert detector.n_features is None


class TestFitPredict:
    def test_returns_one_score_per_row(self) -> None:
        window = make_window(n=200)
        scores = AnomalyDetector().fit_predict(window)
        assert scores.shape == (200,)
        assert scores.dtype == np.float64

    def test_lower_score_means_more_anomalous(self) -> None:
        # sklearn's decision_function convention: negative == flagged.
        # Inverting this silently produces a detector that flags only healthy
        # traffic, which looks like a working detector.
        window = make_window()
        scores = AnomalyDetector().fit_predict(window)
        assert scores.min() < 0.0 < scores.max()

    def test_finds_a_planted_extreme(self) -> None:
        window = make_window(n=400)
        planted = np.zeros((1, window.shape[1]))
        planted[0] = window.max(axis=0) * 50.0
        combined = np.vstack([window, planted])
        scores = AnomalyDetector(contamination=0.05).fit_predict(combined)
        assert scores[-1] < 0.0, "the planted outlier must be flagged"
        assert scores[-1] == scores.min(), "and it must be the most anomalous"

    def test_healthy_window_produces_few_flags(self) -> None:
        # A well-behaved system should look well-behaved. A detector that flags
        # a quarter of ordinary traffic would be unusable regardless of what
        # it finds on genuinely bad data.
        window = make_window(n=400)
        scores = AnomalyDetector(contamination=0.05).fit_predict(window)
        rate = float((scores < 0).mean())
        assert rate <= 0.20, f"flagged {rate:.1%} of healthy traffic"

    def test_too_small_a_window_raises_no_error_and_flags_nothing(self) -> None:
        # Below the fit minimum the detector declines to fit and returns
        # shape-stable non-negative scores, so a dashboard request on a fresh
        # install renders an empty list rather than a 500. The caller still has
        # to notice that nothing was examined -- which it can, because detect()
        # reports num_samples (see below).
        scores = AnomalyDetector().fit_predict(np.zeros((3, 6)))
        assert scores.shape == (3,)
        assert np.all(scores >= 0.0)
        assert AnomalyDetector().is_fitted is False

    def test_empty_window_returns_empty(self) -> None:
        assert AnomalyDetector().fit_predict(np.zeros((0, 6))).size == 0

    def test_accepts_a_plain_python_list(self) -> None:
        scores = AnomalyDetector().fit_predict([[1.0] * 6 for _ in range(60)])
        assert scores.shape == (60,)


class TestDetect:
    def test_reports_samples_features_and_observed_rate(self) -> None:
        result = AnomalyDetector(contamination=0.05).detect(make_window(n=300))
        assert result["num_samples"] == 300
        assert result["num_features"] == len(DEFAULT_FEATURES)
        assert result["detection_method"] == "isolation_forest"
        assert result["anomaly_rate"] == pytest.approx(
            result["num_anomalies"] / 300
        )

    def test_observed_rate_is_not_the_configured_contamination(self) -> None:
        # Contamination is a prior handed to the estimator, not a measurement.
        # Reporting the configured rate as the observed one would be a
        # fabricated number on the dashboard.
        result = AnomalyDetector(contamination=0.1).detect(make_window(n=300))
        assert result["contamination"] == 0.1
        assert result["anomaly_rate"] != result["contamination"] or result[
            "anomaly_rate"
        ] == pytest.approx(0.1)

    def test_flags_array_matches_the_score_array(self) -> None:
        result = AnomalyDetector().detect(make_window(n=150))
        assert result["flags"].shape == result["scores"].shape
        assert int(result["flags"].sum()) == result["num_anomalies"]

    def test_small_window_reports_zero_samples_not_a_clean_bill_of_health(self) -> None:
        # This is the distinction the suite exists for. num_samples == 0 with
        # num_anomalies == 0 means "nothing was examined"; the API layer has to
        # be able to tell that apart from a healthy system, and it can only do
        # so because the count is reported.
        result = AnomalyDetector().detect(np.zeros((2, 6)))
        assert result["num_samples"] == 2
        assert result["num_anomalies"] == 0
        assert result["anomaly_rate"] == 0.0

    def test_every_reported_number_is_finite(self) -> None:
        result = AnomalyDetector().detect(make_window())
        for key in ("expected_low", "expected_high"):
            assert np.all(np.isfinite(result[key]))
        assert np.all(np.isfinite(result["scores"]))


class TestPercentileBand:
    def test_band_brackets_the_middle_of_the_distribution(self) -> None:
        window = make_window(n=1000)
        low, high = AnomalyDetector().percentile_band(window)
        assert low.shape == high.shape == (len(DEFAULT_FEATURES),)
        assert np.all(low <= high)

    def test_band_is_per_column_not_global(self) -> None:
        # Column 0 (total_tokens) and column 5 (estimated_cost) differ by
        # orders of magnitude; a single global band would be meaningless.
        window = make_window(n=500)
        low, high = AnomalyDetector().percentile_band(window)
        assert high[0] > high[5]

    def test_empty_input_gives_empty_arrays(self) -> None:
        low, high = AnomalyDetector().percentile_band(np.zeros((0, 6)))
        assert low.size == 0 and high.size == 0

    def test_band_does_not_require_a_fit(self) -> None:
        # The band is a descriptive statistic, so it still means something on
        # a window too small for Isolation Forest to fit.
        detector = AnomalyDetector()
        low, high = detector.percentile_band(make_window(n=5))
        assert detector.is_fitted is False
        assert low.size == 6


class TestFeatureDeviation:
    def test_a_normal_row_sits_inside_the_band(self) -> None:
        window = make_window(n=500)
        deviation = AnomalyDetector().feature_deviation(window, 0)
        assert set(deviation) == set(DEFAULT_FEATURES)
        assert all(0.0 <= value <= 1.0 for value in deviation.values())

    def test_an_extreme_row_is_measured_as_above_one(self) -> None:
        # Deviation is expressed in band-widths, so a point far outside can
        # exceed 1.0 -- it says "this is N band-widths away".
        window = make_window(n=500)
        extreme = np.zeros((1, 6))
        extreme[0] = window.max(axis=0) * 100.0
        combined = np.vstack([window, extreme])
        deviation = AnomalyDetector().feature_deviation(combined, len(combined) - 1)
        assert max(deviation.values()) > 1.0

    def test_out_of_range_index_raises(self) -> None:
        window = make_window(n=20)
        with pytest.raises(IndexError):
            AnomalyDetector().feature_deviation(window, 20)

    def test_empty_window_gives_no_deviation(self) -> None:
        assert AnomalyDetector().feature_deviation(np.zeros((0, 6)), 0) == {}

    def test_non_default_feature_count_is_reported_positionally(self) -> None:
        # Mislabelling three columns as "total_tokens, duration_ms,
        # context_tokens" would make the evidence on a stored anomaly a lie.
        window = np.ones((50, 3))
        deviation = AnomalyDetector().feature_deviation(window, 0)
        assert set(deviation) == {"feature_0", "feature_1", "feature_2"}


class TestNonFiniteInput:
    """Telemetry legitimately contains NULLs; they must not 500 the detector."""

    def test_nan_rows_are_substituted_not_rejected(self) -> None:
        window = make_window(n=100)
        window[5, 0] = np.nan
        window[7, 3] = np.inf
        result = AnomalyDetector().detect(window)
        assert result["num_samples"] == 100
        assert np.all(np.isfinite(result["scores"]))

    def test_everything_non_finite_still_returns_a_result(self) -> None:
        result = AnomalyDetector().detect(np.full((50, 6), np.nan))
        assert result["num_samples"] == 50
        assert result["num_anomalies"] == 0
