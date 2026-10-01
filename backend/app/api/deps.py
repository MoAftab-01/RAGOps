"""Shared FastAPI dependencies for the v1 API.

Every analytics endpoint in RAGOps is scoped the same way — a time window and
optionally one application — and every list endpoint pages the same way. Those
two concerns are resolved once here rather than in thirteen routers, because a
route that parses its own window is a route that will eventually parse it
differently from its neighbour, and the visible symptom is a dashboard that
shows two applications' p95 latency side by side on the same chart.

Nothing in this module computes a metric. It turns query parameters into an
:class:`~app.services.window.AnalyticsScope` and a validated page window, and
leaves every number to the repository and service layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid
from typing import Annotated, Any, TypeVar

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy import Select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.core.security import (
    AuthDependency,
    Principal,
    TenantPrincipal,
    require_api_key,
)
from app.repositories.common import (
    MAX_PAGE_SIZE,
    apply_pagination,
    count_rows,
    get_application_by_name,
)
from app.schemas.common import Page
from app.services.window import AnalyticsScope, resolve_application_and_window

T = TypeVar("T")

logger = get_logger(__name__)

__all__ = [
    "AuthDependency",
    "KnownScope",
    "PageParams",
    "Pagination",
    "Principal",
    "TenantPrincipal",
    "WindowParams",
    "WriteGuard",
    "DbSession",
    "not_found",
    "page_params",
    "paginate",
    "require_known_application",
    "resolve_application_or_404",
    "window_params",
    "window_scope",
]

#: The window vocabulary the contract promises. Validated here rather than left
#: to ``resolve_window``, which treats an unknown label as ``7d`` — a silent
#: coercion that answers a typo with a confident wrong window.
ALLOWED_WINDOWS: frozenset[str] = frozenset({"1h", "24h", "7d", "30d", "90d"})

#: Contract default when the caller supplies neither a window nor a range.
DEFAULT_WINDOW = "7d"

DbSession = Annotated[AsyncSession, Depends(get_db)]

#: Route-level write guard, for use in a router's ``dependencies=[...]`` list.
#: :data:`AuthDependency` is the parameter-annotation form of the same check;
#: both delegate to :func:`app.core.security.require_api_key`, and neither
#: re-implements any part of the comparison.
WriteGuard = Depends(require_api_key)


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowParams:
    """The three query parameters every analytics endpoint shares.

    Raw, unvalidated input. :func:`window_scope` is what turns it into a
    resolved :class:`AnalyticsScope` against the database.
    """

    window: str = DEFAULT_WINDOW
    start: datetime | None = None
    end: datetime | None = None
    application: str | None = None


def _validate_window(value: str) -> str:
    if value not in ALLOWED_WINDOWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unknown window {value!r}. Expected one of: "
                f"{', '.join(sorted(ALLOWED_WINDOWS))}."
            ),
        )
    return value


def window_params(
    window: Annotated[
        str, Query(description="Relative window; ignored when start/end are given")
    ] = DEFAULT_WINDOW,
    start: Annotated[
        datetime | None, Query(description="ISO-8601 start; enables a custom range")
    ] = None,
    end: Annotated[datetime | None, Query(description="ISO-8601 end of a custom range")] = None,
    application: Annotated[
        str | None, Query(description="Application name; null means all applications")
    ] = None,
) -> WindowParams:
    """Parse the shared analytics query parameters.

    A custom range still has to name a window for the *label* the UI echoes
    back, so the incoming label is validated even when ``start``/``end`` win:
    a request carrying both a custom range and a nonsense window is a client
    bug worth reporting rather than half-honouring.
    """
    _validate_window(window)
    if start is not None and end is not None and start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start must be earlier than end.",
        )
    return WindowParams(window=window, start=start, end=end, application=application)


WindowQuery = Annotated[WindowParams, Depends(window_params)]


async def window_scope(
    session: DbSession,
    params: WindowQuery,
    principal: TenantPrincipal,
) -> AnalyticsScope:
    """Resolve the shared parameters into a queryable scope.

    The application name is resolved to an id here, so no router has to do a
    lookup of its own. An unknown name resolves to a scope with
    ``found=False``; read routes pass that through as an empty result rather
    than inventing numbers, and routes that can 404 check the flag.

    The principal is bound here, once, rather than at every route: this is the
    single funnel every analytics read passes through, so it is the right place
    to pin ``organization_id`` into the scope. A route cannot forget it, and
    neither can the repository layer that reads ``scope.window_filter()``.

    ``current_principal`` resolves an absent key to the platform principal, so
    the local demo and the shared ``RAGOPS_API_KEY`` keep working unchanged.
    """
    scope = await resolve_application_and_window(
        session,
        application_name=params.application,
        window=params.window,
        start=params.start,
        end=params.end,
        organization_id=principal.organization_id,
    )
    logger.debug("analytics.scope_resolved", **scope.context())
    return scope


ScopeQuery = Annotated[AnalyticsScope, Depends(window_scope)]


async def require_known_application(scope: ScopeQuery) -> AnalyticsScope:
    """404 when ``?application=`` named something that does not exist.

    The resolver deliberately does not raise, and this is the one place that acts
    on its ``found`` flag — so no router has to re-implement the check, and none
    can forget it and silently answer with platform-wide numbers.

    The parameter is annotated ``ScopeQuery`` rather than ``AnalyticsScope`` on
    purpose: a bare dataclass annotation is a *body* field to FastAPI, so
    declaring it directly would make every route using this dependency demand a
    JSON body and reject the ``GET`` that only ever carries query parameters.
    """
    if not scope.found:
        raise await not_found(f"No application named {scope.application_name!r}.")
    return scope


#: The same scope, but 404s if ``?application=`` named something unknown. Used by
#: every read route whose queries are scoped by application: the resolver
#: deliberately degrades an unknown filter to a platform-wide scope rather than
#: raising, which is right for a dashboard going blank and wrong for an API,
#: where "no data for this application" and "no such application" are different
#: answers and a caller cannot tell them apart.
KnownScope = Annotated[AnalyticsScope, Depends(require_known_application)]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageParams:
    """1-based page and page size, clamped to the contract's ceiling."""

    page: int
    page_size: int


def page_params(
    page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
    page_size: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description=f"Items per page (max {MAX_PAGE_SIZE})")
    ] = 25,
) -> PageParams:
    return PageParams(page=page, page_size=page_size)


Pagination = Annotated[PageParams, Depends(page_params)]


async def paginate(
    session: AsyncSession,
    stmt: Select[Any],
    params: PageParams,
    *,
    schema: type[T],
) -> Page[T]:
    """Run a filtered statement paged, and count the unpaginated total.

    ``schema`` is the Pydantic model the rows serialise into. The count runs
    against the same filtered statement with the sort stripped (see
    :func:`~app.repositories.common.count_rows`), so ``total`` reflects the
    filters rather than the current page — the number a paginator needs in
    order to work at all.

    Two round trips rather than a window function on purpose: ``COUNT(*) OVER
    ()`` returns nothing at all when the page is past the end, which is exactly
    the request whose total a client most needs.
    """
    total = int((await session.execute(count_rows(stmt))).scalar_one())
    result = await session.execute(apply_pagination(stmt, params.page, params.page_size))
    return Page[T].build(
        items=[schema.model_validate(row) for row in result.scalars().all()],
        total=total,
        page=params.page,
        page_size=params.page_size,
    )


# ---------------------------------------------------------------------------
# Lookups shared by several routers
# ---------------------------------------------------------------------------


async def not_found(detail: str) -> HTTPException:
    """Build the 404 every "this id does not exist" path returns.

    A factory rather than a constant so the message can name what was missing,
    which is the difference between a client that can fix its bug and one that
    retries forever.
    """
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


async def resolve_application_or_404(
    session: AsyncSession, name: str | None, principal: Principal
) -> uuid.UUID | None:
    """Application id for a request-supplied name, or ``None`` when unscoped.

    ``None`` in, ``None`` out: no name means "every application I can see",
    which is a legitimate request.

    A name that does not resolve is a 404, which is the difference between
    these routes and the read endpoints. Every caller here *writes* rows that
    will later be listed beside that application's other runs, experiments or
    recommendations -- so a typo silently becoming a platform-wide run would
    corrupt the comparison that run exists to feed. A read route degrades to an
    empty panel; a write route has to refuse.

    It lives here, not in the repository layer, because raising
    ``HTTPException`` is a transport concern. This replaced four byte-identical
    private copies -- one each in ``answer_eval``, ``experiments``,
    ``optimization`` and ``retrieval`` -- and the copies had to go: a tenant
    predicate added to one and not the other three would be indistinguishable
    from correct code, which is the same failure mode
    :func:`~app.repositories.common.apply_tenant_filter` prevents on the read
    side.

    A name belonging to another organization resolves to 404, not 403. A 403
    would confirm the application exists, and "does application
    ``customer-support-bot`` exist somewhere in this deployment?" is exactly the
    question a tenant must not be able to ask.
    """
    if not name:
        return None
    application = await get_application_by_name(session, name, principal.organization_id)
    if application is None:
        raise await not_found(f"No application named {name!r}.")
    return application.id


def matches_search(columns, term: str):
    """``ILIKE`` across several text columns, for the trace list's free-text filter.

    The term is wrapped in the wildcards here and bound as a parameter, so a
    user searching for ``%`` matches everything rather than producing a syntax
    error or an injection.
    """
    pattern = f"%{term.strip()}%"
    return or_(*(column.ilike(pattern) for column in columns))
