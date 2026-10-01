"""Run-to-run comparison: which recorded metrics moved, and by how much.

§35 forbids fabricating root causes, and the hardest part of honouring that is
resisting the pull to explain a regression. This module therefore reports two
things and nothing else:

* **deltas** — for each metric present in *both* runs, the recorded movement and
  whether it is a regression given that metric's direction; and
* **configuration differences** — the settings the two runs literally recorded
  differently.

A human may join those two lists into a causal story. This module does not, and
the wording of :func:`factual_changes` and :func:`summarise` is chosen so that
even the summary cannot smuggle a cause in: it says ``rerank changed from true
to false``, never ``recall dropped because rerank was disabled``.

Like :mod:`app.evaluation.metrics`, this is a pure module — no I/O, no clock,
no database, no model. The route loads two runs and calls in here.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.evaluation.metrics import metric_direction
from app.schemas.evaluation import ConfigDifference, MetricDelta

__all__ = [
    "DEFAULT_REGRESSION_THRESHOLD_PCT",
    "compare_runs",
    "config_differences",
    "deltas",
    "factual_changes",
    "is_regression",
    "numeric_metric",
    "regressions_of",
    "summarise",
]

#: Default size of a drop worth calling a regression, as a relative percentage.
#: This is a threshold on the recorded change, not a tolerance the report
#: adjusts to reach a verdict: set it to 0 to flag any movement at all, which
#: is what you want when the baseline is a published number other people rely on.
DEFAULT_REGRESSION_THRESHOLD_PCT = 5.0

#: Tolerance on the threshold comparison, in percentage points. Sized to sit
#: well above the representation error of a single divide while staying far
#: below any movement a reader would call a movement.
_EPSILON = 1e-9

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
NEUTRAL = "neutral"


def numeric_metric(value: Any) -> float | None:
    """A stored metric as a float, or ``None`` if it is not a number.

    Runs store a nested ``by_k`` mapping alongside the flat metrics; those keys
    are dictionaries, and averaging over them would raise rather than report.
    Booleans are excluded deliberately — ``True`` is an ``int`` in Python, and
    treating a recorded flag as ``1.0`` would report a phantom numeric delta.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return number


def is_regression(
    direction: str,
    change: float,
    relative_pct: float | None,
    threshold_pct: float,
) -> bool:
    """Whether a movement counts as a regression, per the metric's direction.

    A drop from a zero baseline is judged on its absolute size instead: there is
    no meaningful percentage change from nothing, and refusing to judge it would
    let a metric appear frozen forever at the value it was introduced with.

    A ``neutral`` direction is never a regression. Refusing to guess which way
    is worse is the point — guessing wrong flags a healthy metric as broken.

    The comparison carries a small epsilon. ``0.95`` against ``1.0`` is a drop of
    ``-5.000000000000004`` percent once binary floating point has had its way
    with it, and ``-5.000000000000004 < -5.0`` is true — so a run that moved by
    exactly the threshold would be reported as a regression while a run that
    moved by a hair more would not. Deciding a verdict on representation error
    is not a decision.
    """
    if relative_pct is None:
        return change < 0 and direction == HIGHER_IS_BETTER
    if direction == HIGHER_IS_BETTER:
        return relative_pct < -abs(threshold_pct) - _EPSILON
    if direction == LOWER_IS_BETTER:
        return relative_pct > abs(threshold_pct) + _EPSILON
    return False


def deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold_pct: float = DEFAULT_REGRESSION_THRESHOLD_PCT,
) -> list[MetricDelta]:
    """Delta every metric present in *both* runs, in a stable (sorted) order.

    A metric missing from either run is skipped rather than treated as zero: one
    side not having recorded it means the comparison cannot be made, and zero is
    a value that would immediately register as a catastrophic drop.

    Direction comes from :func:`app.evaluation.metrics.metric_direction`, which
    returns ``neutral`` for a metric it has never heard of.
    """
    found: list[MetricDelta] = []
    for metric in sorted(set(baseline) & set(candidate)):
        base_value = numeric_metric(baseline[metric])
        cand_value = numeric_metric(candidate[metric])
        if base_value is None or cand_value is None:
            continue
        change = cand_value - base_value
        # None when the baseline was zero: a percentage change from nothing is
        # not a number, and 0.0 or 100.0 would both be lies. The schema keeps a
        # float, so it is normalised below only once is_regression has seen it.
        relative = (change / abs(base_value) * 100.0) if base_value else None
        direction = metric_direction(metric)
        found.append(
            MetricDelta(
                metric=metric,
                baseline=base_value,
                candidate=cand_value,
                absolute_change=change,
                relative_change_pct=relative if relative is not None else 0.0,
                is_regression=is_regression(direction, change, relative, threshold_pct),
                direction=direction,
            )
        )
    return found


def config_differences(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[ConfigDifference]:
    """Keys whose recorded values differ, sorted, flattened one level deep.

    Only keys present in both configs and actually unequal. A key missing from
    one side is not reported as a difference, because "the candidate did not
    record this" is not the same statement as "the candidate set it
    differently", and conflating them would put a phantom change into the list
    of things that changed.
    """
    differences: list[ConfigDifference] = []
    for key in sorted(set(baseline) & set(candidate)):
        base_value, cand_value = baseline[key], candidate[key]
        if isinstance(base_value, dict) and isinstance(cand_value, dict):
            differences.extend(
                ConfigDifference(
                    key=f"{key}.{nested}",
                    baseline_value=base_value[nested],
                    candidate_value=cand_value[nested],
                )
                for nested in sorted(set(base_value) & set(cand_value))
                if base_value[nested] != cand_value[nested]
            )
        elif base_value != cand_value:
            differences.append(
                ConfigDifference(
                    key=key, baseline_value=base_value, candidate_value=cand_value
                )
            )
    return differences


def factual_changes(differences: Sequence[ConfigDifference]) -> list[str]:
    """Each recorded change as a sentence about settings, not about causes.

    Deliberately phrased as what the two configs say. The temptation on a
    regression page is to write "reranking was disabled, which lowered recall" —
    and that causal link is a hypothesis that may be wrong even when it feels
    obvious, which is exactly why it does not belong in an automated report.
    """
    return [
        f"Recorded setting `{difference.key}` changed from "
        f"`{difference.baseline_value!r}` to `{difference.candidate_value!r}`."
        for difference in differences
    ]


def regressions_of(all_deltas: Sequence[MetricDelta]) -> list[MetricDelta]:
    """Just the flagged movements, in the order they were reported."""
    return [delta for delta in all_deltas if delta.is_regression]


def summarise(
    regression_list: Sequence[MetricDelta], differences: Sequence[ConfigDifference]
) -> str | None:
    """One sentence naming the recorded regressions, or ``None`` if there are none."""
    if not regression_list:
        return None
    moved = ", ".join(
        f"{delta.metric} {delta.baseline:g} to {delta.candidate:g}"
        for delta in regression_list
    )
    context = (
        f" {len(differences)} recorded setting(s) also differ between these runs."
        if differences
        else ""
    )
    return f"Recorded regression in {moved}.{context}"


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold_pct: float = DEFAULT_REGRESSION_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Compare two recorded runs: the deltas, the config differences, the verdict.

    Returns a plain dict rather than the API's :class:`RegressionReport` because
    this module does not know the two run ids or names — the route supplies
    those. Keeping that out means the comparison can be tested with no
    database, which is the only way it gets tested at all.

    Both arguments are a ``{"metrics": ..., "config": ...}`` mapping. The two
    sections are read explicitly rather than iterated over together, so a
    config key can never be mistaken for a metric and both comparisons cannot
    silently come back empty.
    """
    found = deltas(baseline.get("metrics") or {}, candidate.get("metrics") or {}, threshold_pct)
    differences = config_differences(
        baseline.get("config") or {}, candidate.get("config") or {}
    )
    flagged = regressions_of(found)
    return {
        "deltas": found,
        "config_differences": differences,
        "has_regression": bool(flagged),
        "regression_summary": summarise(flagged, differences),
        "contributing_config_changes": factual_changes(differences),
        "threshold_pct": threshold_pct,
    }
