"""Unsupervised anomaly detection over recorded telemetry features.

IsolationForest rather than a supervised model or a hand-written z-score
threshold, for one reason: there are no labels. Nobody has annotated which
traces were "abnormal", and inferring them from the very data the detector will
score is circular. Isolation Forest also does not assume a distribution — the
token and latency columns in a real RAG workload are multi-modal by construction
(a summarisation call and a one-token classification call are not two draws from
one distribution), so a mean±σ band would flag an entire healthy traffic class.

This module is pure: numpy in, numpy out. No database, no network, no import
side effects, no model file. Everything it reports is derived from the feature
matrix handed to it, which is what makes it unit-testable against
hand-computed expectations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
from sklearn.ensemble import IsolationForest

from app.config import settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = ["AnomalyDetector", "DEFAULT_FEATURES"]

# Feature order used when the caller does not name columns. Kept aligned with
# ``app.schemas.insights.AnomalyDetectRequest.features`` so the request schema
# and this default cannot drift apart silently.
DEFAULT_FEATURES: Final[tuple[str, ...]] = (
    "total_tokens",
    "duration_ms",
    "context_tokens",
    "num_documents",
    "agent_iterations",
    "estimated_cost",
)

# Below this many rows Isolation Forest cannot fit anything useful; the honest
# answer is "anomaly rate 0, detection not attempted", not a confident 0.0 from
# a model fitted on four points.
_MIN_FIT_SAMPLES: Final = 10

# Percentiles defining the "expected" band shown next to an anomaly. 5/95
# catches the visible tail without letting a single outlier widen the band until
# it stops being an outlier.
_LOW_PERCENTILE: Final = 5.0
_HIGH_PERCENTILE: Final = 95.0


def _as_matrix(features: Any) -> NDArray[np.float64]:
    """Coerce input into a finite 2-D float64 matrix, or raise a clear error.

    Non-finite values are replaced with zeros rather than rejected. Telemetry
    legitimately contains ``NULL`` cost and ``NULL`` latency, and a single NaN
    would otherwise make sklearn raise and take down a whole detection run. The
    substitution is recorded so a caller can tell it happened.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 1-D or 2-D, got {matrix.ndim}-D")
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=np.float64)
    if not np.isfinite(matrix).all():
        logger.warning(
            "ml.anomaly.non_finite_features",
            non_finite=int((~np.isfinite(matrix)).sum()),
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


class AnomalyDetector:
    """IsolationForest wrapper with an explicit, non-fit-dependent band.

    ``contamination`` is the *expected fraction* of anomalies handed to the
    estimator. It is a prior, not a measurement: IsolationForest then uses it to
    place its decision threshold, and the realised anomaly rate on any given
    window will differ. The detector reports both, so the dashboard never shows
    the configured rate as if it were the observed one.
    """

    def __init__(
        self,
        *,
        contamination: float | None = None,
        random_state: int = 42,
        n_estimators: int = 100,
        max_samples: float | int = 1.0,
    ) -> None:
        # random_state is pinned: an anomaly report is shown to a user, and a
        # detector that flags a different trace on every page refresh cannot be
        # investigated.
        self.contamination = float(
            contamination if contamination is not None else settings.anomaly_contamination
        )
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self._model: IsolationForest | None = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit_predict` has been called on this instance."""
        return self._model is not None

    @property
    def n_features(self) -> int | None:
        """Feature width the model was fitted on."""
        return self._n_features

    @property
    def min_samples(self) -> int:
        """Minimum rows before a fit is attempted."""
        return max(_MIN_FIT_SAMPLES, int(settings.anomaly_min_samples) // 5)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit_predict(self, features: Any) -> NDArray[np.float64]:
        """Fit on ``features`` and return the per-row anomaly score.

        The returned array is ``decision_function`` output: **lower is more
        anomalous**, negative means "flagged by the fitted threshold". The sign
        convention is sklearn's and is stated here because inverting it by
        accident produces a detector that flags only the *healthy* traffic.

        Returns an empty array (rather than raising) for empty or too-small
        input, so a dashboard request on a fresh install renders an empty list
        instead of a 500.
        """
        matrix = _as_matrix(features)
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)

        if matrix.shape[0] < self.min_samples:
            logger.info(
                "ml.anomaly.insufficient_samples",
                samples=int(matrix.shape[0]),
                min_samples=self.min_samples,
            )
            return np.zeros(matrix.shape[0], dtype=np.float64)

        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            # 'entropy' is the modern alias for the old 'auto': it picks
            # max_features=1.0 when there are many rows, which is the behaviour
            # that gives the best detection on a handful of correlated columns.
            max_features=1.0,
        )
        model.fit(matrix)
        self._model = model
        self._n_features = int(matrix.shape[1])

        scores = np.asarray(model.decision_function(matrix), dtype=np.float64)
        flags = np.asarray(model.predict(matrix) == -1, dtype=bool)
        logger.info(
            "ml.anomaly.fitted",
            samples=int(matrix.shape[0]),
            features=self._n_features,
            contamination=self.contamination,
            anomalies=int(flags.sum()),
        )
        return scores

    def detect(self, features: Any) -> dict[str, Any]:
        """Fit, score, and package the band and the flagged rows together.

        This is the shape the anomaly service persists and the dashboard reads,
        so it is produced in one place rather than reassembled from three method
        calls at each layer.
        """
        matrix = _as_matrix(features)
        scores = self.fit_predict(matrix)

        low, high = self.percentile_band(matrix)
        flags = scores < 0.0 if scores.size else np.zeros(0, dtype=bool)
        num_anomalies = int(flags.sum())

        return {
            "num_samples": int(matrix.shape[0]),
            "num_features": int(matrix.shape[1]),
            "contamination": self.contamination,
            "detection_method": "isolation_forest",
            "num_anomalies": num_anomalies,
            # The observed rate, not the configured prior. These differ, and
            # conflating them is exactly the kind of fabricated number the
            # project forbids.
            "anomaly_rate": (num_anomalies / matrix.shape[0]) if matrix.shape[0] else 0.0,
            "expected_low": low,
            "expected_high": high,
            "scores": scores,
            "flags": flags,
        }

    def percentile_band(
        self, features: Any
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the ``(low, high)`` percentile band, per feature column.

        The band is a *descriptive* statistic of the same window, computed with
        ``numpy.percentile`` and nothing else. It is not derived from the fitted
        model, so it still means something on a window too small to fit, and it
        cannot be contaminated by the detector's own threshold.

        Empty input yields two empty arrays rather than an error, which keeps
        ``detect`` a single code path.
        """
        matrix = _as_matrix(features)
        if matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        low = np.percentile(matrix, _LOW_PERCENTILE, axis=0)
        high = np.percentile(matrix, _HIGH_PERCENTILE, axis=0)
        return low.astype(np.float64), high.astype(np.float64)

    def feature_deviation(
        self, features: Any, row_index: int
    ) -> dict[str, float]:
        """Per-feature distance of one row from the band, as a 0-1 fraction.

        Reported as ``feature_importance_proxy`` in the API schema, and named
        accordingly here: it measures *how far outside the band* each column
        sits, which is evidence for the finding. It is not a fitted importance
        score, and calling it one would misdescribe what IsolationForest did.
        """
        matrix = _as_matrix(features)
        if matrix.shape[0] == 0:
            return {}
        if not 0 <= row_index < matrix.shape[0]:
            raise IndexError(f"row_index {row_index} out of range for {matrix.shape[0]} rows")

        low, high = self.percentile_band(matrix)
        row = matrix[row_index]
        span = np.where((high - low) > 0, high - low, 1.0)

        below = np.clip((low - row) / span, 0.0, None)
        above = np.clip((row - high) / span, 0.0, None)
        deviation = np.maximum(below, above)

        names = DEFAULT_FEATURES
        if len(deviation) != len(names):
            # Feature count differs from the default column set: report by
            # position so the value is still usable, just not mislabelled.
            names = tuple(f"feature_{i}" for i in range(len(deviation)))

        return {name: round(float(value), 4) for name, value in zip(names, deviation)}

    def reset(self) -> None:
        """Drop the fitted model, so the instance can score a new window."""
        self._model = None
        self._n_features = None
