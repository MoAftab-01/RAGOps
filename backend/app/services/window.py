"""Window + application resolution shared by every analytics service.

Every analytics question in RAGOps has the same two qualifiers — *which time
window* and *which application* — so resolving them in one place is what stops
the seven analytics services from drifting apart on how a request is scoped.
Getting that wrong is not a cosmetic bug: an unnoticed application filter turns
"this application's p95 latency" into "the whole platform's p95 latency" and
the dashboard quietly shows the wrong number.

The contract is intentionally narrow:

* ``window`` is one of ``1h|24h|7d|30d|90d``; supplying ``start``/``end``
  switches to a custom range and wins over ``window``.
* ``application_name`` of ``None`` means *all applications in the requesting
  organization* and resolves to ``application_id is None`` — that is a
  deliberate scope, not a missing filter, so it is never confused with "not
  found". With no organization (the platform principal) it is every
  application, which is the pre-tenancy behaviour.
* A name that does not exist resolves to ``application_id is None`` too, but
  with ``found=False`` and a warning log, so the caller can distinguish
  "no filter" from "you asked about an application that is not instrumented".
  Silently answering a bad name with platform-wide numbers would be a lie.
* ``organization_id`` is the tenant confinement. It bounds the name lookup as
  well as the queries, so ``?application=<someone else's name>`` does not
  resolve at all rather than resolving and then being filtered.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.repositories.common import WindowFilter, get_application_by_name
from app.utils.pricing import resolve_window

logger = get_logger(__name__)

__all__ = [
    "AnalyticsScope",
    "build_window_filter",
    "cache_parts",
    "ensure_aware",
    "resolve_application_and_window",
]

# Bucket width that keeps a window readable: hourly for a day or less, daily
# beyond that. Mirrors docs/API_CONTRACT.md ("One bucket per hour for 1h/24h,
# per day otherwise") so the chart resolution matches what the UI promised.
HOURLY_WINDOWS: frozenset[str] = frozenset({"1h", "24h", "custom"})
DEFAULT_WINDOW = "7d"


def ensure_aware(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, convert an aware one to UTC.

    The API accepts ISO-8601 datetimes and Pydantic will happily hand back a
    naive one when the caller omitted the offset. Comparing that against a
    ``timestamptz`` column would either raise or silently use the server's
    local timezone, so every bound is normalised here once instead of at each
    of the dozen query sites.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class AnalyticsScope:
    """A resolved read scope: the window bounds, the application, and its label.

    Carries both the presentation fields (the label the UI echoes back) and the
    query fields (the bounds and ids the repositories filter on) so a service
    cannot render one window while querying another.

    ``organization_id`` is the tenant this scope is confined to, or ``None`` for
    the platform principal. It is threaded all the way into
    :class:`WindowFilter` so that "no application filter" reads as *all of this
    company's applications* rather than *all applications*. Both being ``None``
    is the platform-wide case, reachable only through the auth escape hatch.
    """

    start: datetime
    end: datetime
    label: str
    application_id: uuid.UUID | None = None
    application_name: str | None = None
    found: bool = True
    organization_id: uuid.UUID | None = None

    @property
    def window(self) -> str:
        """Alias so callers can pass ``scope.window`` straight into a schema."""
        return self.label

    @property
    def is_hourly(self) -> bool:
        """True when time-series buckets should be hourly rather than daily."""
        return self.label in HOURLY_WINDOWS

    @property
    def bucket_unit(self) -> str:
        """SQL ``date_trunc`` unit for this scope's time series."""
        return "hour" if self.is_hourly else "day"

    def window_filter(self) -> WindowFilter:
        """The repository-side filter object for this scope.

        Note this copies ``organization_id`` automatically. ``applications.py``
        rebuilds its scope with ``dataclasses.replace`` to add an application
        filter, and the organization survives that, which is exactly what keeps
        a tenant's per-application stats from escaping to the platform scope.
        """
        return WindowFilter(
            start=self.start,
            end=self.end,
            application_id=self.application_id,
            organization_id=self.organization_id,
        )

    def context(self) -> dict[str, Any]:
        """Log-safe summary attached to every service log line.

        Includes the organization because this is the one structured log that
        fires for every analytics read: if a cross-tenant leak ever happens,
        this line is where the answer to "which org was it looking at?" is.
        """
        return {
            "window": self.label,
            "window_start": self.start.isoformat(),
            "window_end": self.end.isoformat(),
            "application_id": str(self.application_id) if self.application_id else None,
            "application_name": self.application_name,
            "organization_id": (
                str(self.organization_id) if self.organization_id else None
            ),
        }


async def resolve_application_and_window(
    session: AsyncSession,
    application_name: str | None = None,
    window: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    organization_id: uuid.UUID | None = None,
) -> AnalyticsScope:
    """Resolve ``(application, window, start, end, organization)`` into one scope.

    Shared entry point for every analytics service so a route's query parameters
    mean the same thing everywhere. Performs exactly one database round trip —
    and only when an application name was supplied, so the common
    "all applications" dashboard load issues no lookup at all.

    A name that matches no application resolves to a scope with
    ``found=False``. Callers that can surface a 404 should check that flag;
    callers that cannot must still refuse to invent a number for it.

    ``organization_id`` is the tenant confinement, and it is applied to the
    name lookup rather than only to the result: a tenant asking for another
    company's application name gets ``found=False`` and an empty dashboard, not
    that company's numbers. The scope returned on that path keeps the
    organization, so the fallback is still confined.
    """
    resolved_start, resolved_end, label = resolve_window(window, start, end)
    resolved_start = ensure_aware(resolved_start)
    resolved_end = ensure_aware(resolved_end)

    if not application_name:
        return AnalyticsScope(
            start=resolved_start,
            end=resolved_end,
            label=label,
            application_id=None,
            application_name=None,
            found=True,
            organization_id=organization_id,
        )

    application = await get_application_by_name(session, application_name, organization_id)
    if application is None:
        logger.warning(
            "analytics.application_not_found",
            application_name=application_name,
            window=label,
            organization_id=str(organization_id) if organization_id else None,
        )
        # found=False, not an exception: an unknown filter yields an empty
        # dashboard rather than a stack trace, and the flag lets a route turn
        # it into a 404 when that is the better answer.
        return AnalyticsScope(
            start=resolved_start,
            end=resolved_end,
            label=label,
            application_id=None,
            application_name=application_name,
            found=False,
            organization_id=organization_id,
        )

    return AnalyticsScope(
        start=resolved_start,
        end=resolved_end,
        label=label,
        application_id=application.id,
        application_name=application.name,
        found=True,
        organization_id=organization_id,
    )


def build_window_filter(scope: AnalyticsScope) -> WindowFilter:
    """Convenience alias so services do not import ``WindowFilter`` directly."""
    return scope.window_filter()


def cache_parts(scope: AnalyticsScope, endpoint: str) -> dict[str, Any]:
    """Cache-key parts for :func:`app.core.cache.cached_call`.

    The bucket *width* is part of the key because the same window can be
    charted hourly or daily depending on the label; without it a coarse view
    could serve a fine-grained series from cache.

    ``endpoint`` is required, and it is the fix for a real bug rather than
    ceremony. Every analytics route passes the same window, start, end,
    application and bucket, so without a discriminator ``/dashboard/overview``
    and ``/analytics/tokens`` hash to the *same* Redis key: whichever ran first
    cached its payload, and the other then validated that payload against its
    own response model and raised a 500. All seven call sites had it wrong
    identically, because nothing in the signature said the name was needed.
    Making it a positional argument means a new route cannot reuse a
    neighbour's key by omission.

    ``organization_id`` is here for the same reason and is not optional to
    think about: two companies querying the identical window with no
    application filter would otherwise share one entry, and the second would be
    served the first company's dashboard -- a cross-tenant leak that no
    response-model validation would catch, because the *shape* matches. The
    empty string stands for the platform principal, and it is a distinct value
    from any real uuid, so platform and tenant never collide either.

    (This function used to take ``*extra`` and merge it with
    ``dict.update(extra)``, which can only ever work for a sequence of
    key-value *pairs* -- no caller passed one, and the line was dead code
    hiding behind a signature that looked plausible.)
    """
    return {
        "endpoint": endpoint,
        "window": scope.label,
        "start": scope.start.isoformat(),
        "end": scope.end.isoformat(),
        "application_id": str(scope.application_id) if scope.application_id else "",
        "organization_id": str(scope.organization_id) if scope.organization_id else "",
        "bucket": scope.bucket_unit,
    }