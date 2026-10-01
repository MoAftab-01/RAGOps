"""Anomaly list, detection, and resolution.

Detection fits an :class:`~app.ml.anomaly.AnomalyDetector` over a feature matrix
built from recorded columns. Nothing here interprets *why* an outlier is an
outlier: the stored ``evidence`` is the observed value, the band the detector
measured, and the peer statistics behind it. A cause is a claim, and a detector
that writes one into the database will eventually present a coincidence as a
finding.

Two sign conventions from the detector are load-bearing and are not guessed at:
``decision_function`` output is *lower is more anomalous* with negatives
flagged, and the expected band is one value **per feature column**. The route
carries both through rather than summarising them into a single scalar, because
a window-wide band compared against a single trace's tokens would be a
comparison between quantities measured over different columns.

TODO(anomaly_repo): ``app/repositories/anomaly_repo.py`` does not exist, so the
list statement is built here and the feature matrix is assembled in
:func:`_feature_rows`. Both move into the repository when it lands.
"""

from __future__ import annotations

import time
import uuid
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import (
    DbSession,
    KnownScope,
    Pagination,
    TenantPrincipal,
    WriteGuard,
    not_found,
    require_known_application,
)
from app.config import settings
from app.core.logging import get_logger
from app.ml.anomaly import DEFAULT_FEATURES, AnomalyDetector
from app.models import Anomaly, Application, Trace, utcnow
from app.repositories.common import (
    apply_organization_filter,
    apply_pagination,
    apply_tenant_filter,
    count_rows,
)
from app.schemas.common import Page
from app.schemas.insights import AnomalyDetectRequest, AnomalyDetectResponse, AnomalyOut
from app.services.window import AnalyticsScope, resolve_application_and_window

logger = get_logger(__name__)

#: Features that are derived at ingest time and live in ``Trace.metadata`` rather
#: than in a column. Keyed feature name -> the metadata object holding it.
#: Kept next to the query that reads them so the two cannot drift apart; the
#: ingest side is :meth:`app.telemetry.ingest_service.IngestService._store_analysis`.
_METADATA_FEATURES: dict[str, str] = {
    "num_documents": "retrieval_analysis",
}

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


class AnomalyResolveRequest(BaseModel):
    """Resolve or reopen one anomaly.

    A single field because resolution is the only mutable thing about a detected
    anomaly: the measurement that produced it is a record, and editing it would
    make the evidence disagree with the row it was meant to justify.
    """

    is_resolved: bool = Field(description="true resolves the anomaly, false reopens it")


@router.get("", response_model=Page[AnomalyOut], summary="List anomalies")
async def list_anomalies(
    session: DbSession,
    scope: KnownScope,
    pagination: Pagination,
    anomaly_type: Annotated[str | None, Query(description="Filter by anomaly type")] = None,
    severity: Annotated[str | None, Query(description="low | medium | high")] = None,
    is_resolved: Annotated[bool | None, Query(description="Filter by resolution")] = None,
) -> Page[AnomalyOut]:
    """Detected anomalies in the window, newest first.

    ``is_resolved`` left unset means both, which is what an unfiltered list
    should show: resolved anomalies are history, not noise to hide by default.
    """
    # ``Anomaly`` carries an application FK but no relationship, so the name is
    # joined rather than eager-loaded. An outer join, because a platform-wide
    # anomaly has no application at all and an inner one would drop it.
    stmt = (
        select(Anomaly, Application.name)
        .join(Application, Anomaly.application_id == Application.id, isouter=True)
        .where(Anomaly.detected_at >= scope.start, Anomaly.detected_at <= scope.end)
        .order_by(Anomaly.detected_at.desc(), Anomaly.id.desc())
    )
    stmt = apply_tenant_filter(stmt, scope.window_filter(), Anomaly.application_id)
    if anomaly_type:
        stmt = stmt.where(Anomaly.anomaly_type == anomaly_type)
    if severity:
        stmt = stmt.where(Anomaly.severity == severity)
    if is_resolved is not None:
        stmt = stmt.where(Anomaly.is_resolved.is_(is_resolved))

    total = int((await session.execute(count_rows(stmt))).scalar_one())
    rows = (
        await session.execute(apply_pagination(stmt, pagination.page, pagination.page_size))
    ).all()
    return Page[AnomalyOut].build(
        items=[_to_out(anomaly, application_name) for anomaly, application_name in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.post(
    "/detect",
    response_model=AnomalyDetectResponse,
    summary="Run anomaly detection over recent telemetry",
    dependencies=[WriteGuard],
)
async def detect_anomalies(
    session: DbSession, payload: AnomalyDetectRequest
) -> AnomalyDetectResponse:
    """Fit IsolationForest over recorded traces and record what it flagged.

    Reports the fit honestly when it could not run: a window holding fewer
    traces than the detector's floor returns zero anomalies with an explicit
    note, rather than a clean result that would read as "this application's
    telemetry is normal" when in fact nothing was examined.
    """
    end = utcnow()
    scope = await resolve_application_and_window(
        session,
        application_name=payload.application,
        window=payload.window,
        start=end - timedelta(hours=payload.lookback_hours),
        end=end,
    )
    # An unknown application name here is a write, so it must not degrade to a
    # platform-wide fit: the caller would get a clean "0 anomalies" report for
    # the whole system and record it as the result for the application they
    # asked about. Same check the read routes use, called directly because this
    # scope comes from the request body rather than a query parameter.
    scope = await require_known_application(scope)
    features = _resolve_features(payload.features)
    contamination = (
        payload.contamination
        if payload.contamination is not None
        else settings.anomaly_contamination
    )
    rows = await _feature_rows(session, scope, features)
    detector = AnomalyDetector(contamination=contamination)

    started = time.perf_counter()
    if len(rows) < detector.min_samples:
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "anomalies.detect.skipped",
            samples=len(rows),
            required=detector.min_samples,
            application=scope.application_name,
        )
        return _empty_fit(scope, features, contamination, len(rows), duration_ms)

    matrix = [row["features"] for row in rows]
    result = detector.detect(matrix)
    flagged = [index for index, is_flagged in enumerate(result["flags"]) if is_flagged]
    anomalies = await _record_anomalies(
        session, scope, rows, matrix, features, detector, result, flagged, payload.persist
    )
    duration_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "anomalies.detect.completed",
        application=scope.application_name,
        samples=int(result["num_samples"]),
        anomalies=int(result["num_anomalies"]),
        persisted=payload.persist,
    )
    return AnomalyDetectResponse(
        application_id=scope.application_id,
        application_name=scope.application_name,
        num_samples=int(result["num_samples"]),
        num_features=int(result["num_features"]),
        contamination=float(result["contamination"]),
        num_anomalies=int(result["num_anomalies"]),
        # The observed rate, which is what the detector actually found, not the
        # configured prior it was asked to look for.
        anomaly_rate=float(result["anomaly_rate"]),
        feature_importance_proxy=_mean_deviation(detector, matrix, features, flagged),
        anomalies=anomalies,
        duration_ms=duration_ms,
    )


def _empty_fit(
    scope: AnalyticsScope,
    features: list[str],
    contamination: float,
    num_samples: int,
    duration_ms: float,
) -> AnomalyDetectResponse:
    """A detect response for a window too small to fit.

    Zero anomalies and zero deviations, with the reason stated in the log rather
    than buried in the payload: ``num_samples`` below the detector's floor is
    visible in the response, and that is what the caller needs to tell "nothing
    looked wrong" apart from "nothing could be looked at".
    """
    return AnomalyDetectResponse(
        application_id=scope.application_id,
        application_name=scope.application_name,
        num_samples=num_samples,
        num_features=len(features),
        contamination=contamination,
        num_anomalies=0,
        anomaly_rate=0.0,
        feature_importance_proxy={feature: 0.0 for feature in features},
        anomalies=[],
        duration_ms=duration_ms,
    )


def _resolve_features(requested: list[str]) -> list[str]:
    """The columns to fit over, filtered to the ones that exist.

    An unknown name is dropped rather than passed through, because the matrix is
    built by attribute lookup: a typo would produce a matrix silently missing a
    column and a fit reporting ``num_features`` one lower than the caller asked
    for, without saying which column went missing.
    """
    available = set(DEFAULT_FEATURES)
    return [
        feature for feature in (requested or list(DEFAULT_FEATURES)) if feature in available
    ]


async def _feature_rows(
    session: DbSession, scope: AnalyticsScope, features: list[str]
) -> list[dict[str, Any]]:
    """Traces in the window, each with its recorded feature values.

    A trace missing any requested feature is skipped rather than zero-filled. A
    null ``context_tokens`` padded with ``0.0`` becomes the most extreme row in
    that column and gets flagged as an outlier for being unrecorded.
    """
    stmt = select(Trace).where(Trace.start_time >= scope.start, Trace.start_time <= scope.end)
    stmt = apply_tenant_filter(stmt, scope.window_filter(), Trace.application_id)
    rows = (await session.execute(stmt)).scalars().all()

    usable: list[dict[str, Any]] = []
    for trace in rows:
        values = [_feature_value(trace, feature) for feature in features]
        if any(value is None for value in values):
            continue
        usable.append({"trace": trace, "features": [float(value) for value in values]})
    return usable


def _feature_value(trace: Trace, feature: str) -> float | None:
    """One anomaly feature for one trace, or None when it was never recorded.

    Most features are plain columns. ``num_documents`` is not: it is *derived* at
    ingest time from the retrieved-document rows and stored under
    ``metadata->'retrieval_analysis'``. Reading it with ``getattr`` returns None
    for every single trace, the caller-side null filter then drops every row, and
    the run reports ``num_samples: 0`` with zero anomalies -- which reads exactly
    like "this application's telemetry is normal" for an application whose
    telemetry was never examined at all. So the derived feature is read from
    where it is actually stored.

    Anything genuinely absent stays None and is skipped upstream, as before.
    """
    if feature in _METADATA_FEATURES:
        analysis = (trace.extra_metadata or {}).get(_METADATA_FEATURES[feature])
        if not isinstance(analysis, dict):
            return None
        value = analysis.get(feature)
        return None if value is None else float(value)
    value = getattr(trace, feature, None)
    return None if value is None else float(value)


def _mean_deviation(
    detector: AnomalyDetector,
    matrix: list[list[float]],
    features: list[str],
    flagged: list[int],
) -> dict[str, float]:
    """Mean per-feature deviation across the flagged rows.

    The detector's own :meth:`~app.ml.anomaly.AnomalyDetector.feature_deviation`
    — how far each column sits outside its band, as a 0-1 fraction — averaged
    over the rows it flagged. The schema calls this ``feature_importance_proxy``
    and the docstring there says why: it measures how unusual each recorded
    field was, not what the model learned to weight.

    The values are re-keyed onto ``features`` by position. The detector names a
    deviation after the default column order, which only coincides with the
    request's column order when the request used the defaults; a caller who asked
    for two columns would otherwise get back a ``total_tokens`` and a
    ``duration_ms`` reading for whatever it had actually measured.

    No flagged rows means no deviation was measured for any column, so every
    column is reported at zero — the same "nothing was outside the band" reading
    a clean run produces, rather than a missing key the UI would have to handle.
    """
    if not flagged:
        return {feature: 0.0 for feature in features}
    per_feature: dict[str, list[float]] = {feature: [] for feature in features}
    for index in flagged:
        positional = _deviation_by_position(detector, matrix, index)
        for position, feature in enumerate(features):
            per_feature[feature].append(positional[position])
    return {
        feature: round(sum(values) / len(values), 6)
        for feature, values in per_feature.items()
        if values
    }


def _deviation_by_position(
    detector: AnomalyDetector, matrix: list[list[float]], index: int
) -> list[float]:
    """One row's per-column deviations, ordered by column position.

    ``feature_deviation`` returns a dict keyed by the default feature names when
    the width matches, and by ``feature_{i}`` when it does not. Sorting the keys
    would scramble the order in the second case, so the width decides: the
    default names are positional already, and otherwise every key is positional
    by construction.
    """
    deviation = detector.feature_deviation(matrix, index)
    if len(deviation) == len(DEFAULT_FEATURES):
        return [deviation[name] for name in DEFAULT_FEATURES]
    return [deviation[key] for key in sorted(deviation, key=_positional_index)]


def _positional_index(key: str) -> int:
    """Sort key putting ``feature_{i}`` labels in numeric column order.

    Lexicographic order would put ``feature_10`` before ``feature_2``; the
    deviation for a thirteenth column would then be reported as a second
    column's. Only reached when the matrix width differs from the defaults, and
    it returns 0 for a key of either shape otherwise.
    """
    if key in DEFAULT_FEATURES:
        return DEFAULT_FEATURES.index(key)
    return int(key.rsplit("_", 1)[-1]) if key.rsplit("_", 1)[-1].isdigit() else 0


async def _record_anomalies(
    session: DbSession,
    scope: AnalyticsScope,
    rows: list[dict],
    matrix: list[list[float]],
    features: list[str],
    detector: AnomalyDetector,
    result: dict,
    flagged: list[int],
    persist: bool,
) -> list[AnomalyOut]:
    """Persist one row per flagged trace, or return the projections unpersisted.

    The row's ``expected_low``/``expected_high`` are the band of the *single
    column that strayed most* for that trace, which is the only column its
    ``observed_value`` is comparable to. Every column's band and deviation go
    into ``evidence`` so the row can be read without re-deriving them.

    No cause is written anywhere: the fit produced an outlier score, and nothing
    in it produced a reason.
    """
    outputs: list[AnomalyOut] = []
    low, high = detector.percentile_band(matrix)

    for index in flagged:
        row = rows[index]
        trace: Trace = row["trace"]
        deviation = _deviation_by_position(detector, matrix, index)
        # Ties resolve to the earliest column in the request's own order, so two
        # runs over the same data attribute the same row the same way. Comparing
        # the raw observed values instead would be meaningless: these columns
        # are incommensurable (tokens, milliseconds, dollars), and the largest
        # one is not the strangest one.
        peak = max(range(len(deviation)), key=lambda position: (deviation[position], -position))
        score = result["scores"][index]

        anomaly = Anomaly(
            application_id=scope.application_id,
            trace_id=trace.trace_id,
            anomaly_type="feature_outlier",
            severity=_severity(deviation[peak]),
            metric=features[peak],
            observed_value=row["features"][peak],
            expected_low=float(low[peak]),
            expected_high=float(high[peak]),
            anomaly_score=None if score is None else float(score),
            evidence={
                "detection_method": result["detection_method"],
                "num_samples": int(result["num_samples"]),
                "features_examined": list(features),
                "feature_deviation": {
                    feature: deviation[column] for column, feature in enumerate(features)
                },
                "window_band": {
                    feature: {"low": float(low[column]), "high": float(high[column])}
                    for column, feature in enumerate(features)
                },
            },
            detected_at=utcnow(),
            detection_method=str(result["detection_method"]),
        )
        if persist:
            session.add(anomaly)
            await session.flush()
        outputs.append(_to_out(anomaly, scope.application_name))
    return outputs


def _severity(deviation: float) -> str:
    """Severity from how far outside its band the dominant column sat.

    ``deviation`` is the detector's 0-1 fraction of the column's own band
    width, so the thresholds mean the same thing for a token count and for a
    duration. A deviation of zero is not an outlier at all and is rated ``low``.
    """
    if deviation >= 1.0:
        return "high"
    if deviation >= 0.5:
        return "medium"
    return "low"


def _to_out(anomaly: Anomaly, application_name: str | None = None) -> AnomalyOut:
    """Project a stored anomaly, naming its application when it was joined in."""
    return AnomalyOut(
        id=anomaly.id,
        application_id=anomaly.application_id,
        application_name=application_name,
        trace_id=anomaly.trace_id,
        anomaly_type=anomaly.anomaly_type,
        severity=anomaly.severity,
        metric=anomaly.metric,
        observed_value=anomaly.observed_value,
        expected_low=anomaly.expected_low,
        expected_high=anomaly.expected_high,
        anomaly_score=anomaly.anomaly_score,
        peer_median=anomaly.peer_median,
        peer_p95=anomaly.peer_p95,
        evidence=anomaly.evidence,
        detected_at=anomaly.detected_at,
        detection_method=anomaly.detection_method,
        is_resolved=anomaly.is_resolved,
    )


@router.patch(
    "/{anomaly_id}",
    response_model=AnomalyOut,
    summary="Resolve or reopen an anomaly",
    dependencies=[WriteGuard],
)
async def resolve_anomaly(
    session: DbSession,
    anomaly_id: uuid.UUID,
    payload: AnomalyResolveRequest,
    principal: TenantPrincipal,
) -> AnomalyOut:
    """Mark one anomaly resolved, or reopen it.

    The measurement is untouched. Resolving records that a human has seen it,
    not that the trace was different.

    Scoped to the caller's organization, and 404 rather than 403: this writes to
    a row, and a 403 would tell a tenant that someone else's anomaly exists
    before refusing to let it touch it.
    """
    stmt = (
        select(Anomaly, Application.name)
        .join(Application, Anomaly.application_id == Application.id)
        .where(Anomaly.id == anomaly_id)
    )
    # The join is inner, not ``isouter=True`` as the listing route uses. An
    # anomaly with no application belongs to no company; the outer join exists
    # there only to name a platform-wide anomaly for the platform principal, and
    # here it would let a tenant reach one by id.
    stmt = apply_organization_filter(
        stmt, principal.organization_id, Application.organization_id
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        raise await not_found(f"No anomaly with id {anomaly_id}.")
    anomaly, application_name = row
    anomaly.is_resolved = payload.is_resolved
    logger.info("anomalies.resolved", anomaly_id=str(anomaly_id), resolved=payload.is_resolved)
    return _to_out(anomaly, application_name)