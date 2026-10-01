"""Shared repository primitives: the time window, and entity upserts.

Every read path in RAGOps is scoped by the same pair of questions — *which
time window* and *which application*. Rather than let each repository invent
its own argument shape (and quietly forget the application filter on one of
them), those two concerns live here as a single value object plus two helpers
that are applied through SQLAlchemy expressions.

The upsert helpers exist for the telemetry write path. Instrumentation is
multi-tenant by nature: two processes may discover ``customer-support-bot`` at
the same moment, so the create path has to be race-safe. A
``SELECT`` followed by an ``INSERT`` is not; ``INSERT ... ON CONFLICT DO
NOTHING ... RETURNING id`` followed by a re-read is. That is why these return
the ORM object instead of just an id — callers need ``.id`` for the child rows
they are about to write.

Nothing here touches a connection at import time.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.logging import get_logger
from app.models import Application, User

logger = get_logger(__name__)

T = TypeVar("T")

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500


class ApplicationOwnedByAnotherOrganization(Exception):
    """An application name already belongs to a different company.

    Distinct from a generic lookup miss because the two need different answers
    from the caller: a miss means "send again, it will be created", while this
    means "stop, you cannot have this name". Folding it into a 404 would tell a
    company to keep retrying an ingest that can never succeed.

    The owner's id is carried rather than resolved to a name. Naming the other
    company would turn a tenancy error into an enumeration channel -- the whole
    point of the check is that two companies cannot discover each other, and
    ``organizations.name`` is not secret.
    """

    def __init__(self, *, name: str, owner: uuid.UUID) -> None:
        self.name = name
        self.owner = owner
        super().__init__(
            f"Application {name!r} already exists and belongs to another "
            "organization. Application names are unique across the platform; "
            "choose a different name, or ask that organization to share it."
        )


@dataclass(frozen=True, slots=True)
class WindowFilter:
    """A resolved analytics scope: half-open UTC range plus optional application.

    ``start`` is inclusive and ``end`` is inclusive, matching the ``>=``/``<=``
    comparisons in :func:`apply_window`. Both must be timezone-aware; callers get
    them from :func:`app.utils.pricing.resolve_window`, which is the single
    place window strings are turned into datetimes.

    The two id fields answer different questions, and the pair together answers
    "whose data":

    * ``application_id`` — one application, or ``None`` for all of them.
    * ``organization_id`` — the owning company, or ``None`` for the whole
      platform.

    ``application_id`` of ``None`` no longer means *everything*. It means *all
    applications within ``organization_id``*, and platform-wide is now only the
    case where **both** are ``None``. That combination is reachable solely via
    the auth escape hatch (no key presented, or an empty ``RAGOPS_API_KEY``),
    which is the pre-tenancy behaviour by design.

    This matters because "no application filter" and "no tenancy filter" are
    different questions with the same SQL. A tenant reading a dashboard with no
    ``?application=`` must still see only its own company; if ``None`` meant
    everything, every unscoped read would quietly cross the tenant boundary and
    no individual predicate would look wrong. Every site therefore applies both
    through :func:`apply_tenant_filter`, so "unscoped" has one definition.
    """

    start: datetime
    end: datetime
    application_id: uuid.UUID | None = None
    organization_id: uuid.UUID | None = None

    def describe(self) -> dict[str, Any]:
        """Log-safe summary. Used as structured log context, never as a query."""
        return {
            "window_start": self.start.isoformat(),
            "window_end": self.end.isoformat(),
            "application_id": str(self.application_id) if self.application_id else None,
            "organization_id": (
                str(self.organization_id) if self.organization_id else None
            ),
        }


def apply_window(
    stmt: Select[Any], column: ColumnElement[Any] | None, window: WindowFilter | None
) -> Select[Any]:
    """Constrain ``stmt`` to ``window`` on a timestamp ``column``.

    Returns the statement unchanged when either the column or the window is
    ``None`` so callers can pass an optional column (for example
    ``LLMCall.created_at``) without branching at every call site.
    """
    if column is None or window is None:
        return stmt
    return stmt.where(column >= window.start, column <= window.end)


def apply_application_filter(
    stmt: Select[Any], column: ColumnElement[Any] | None, application_id: uuid.UUID | None
) -> Select[Any]:
    """Constrain ``stmt`` to one application, or leave it unscoped when ``None``."""
    if column is None or application_id is None:
        return stmt
    return stmt.where(column == application_id)


def tenant_predicates(
    application_id: uuid.UUID | None,
    organization_id: uuid.UUID | None,
    application_id_column: Any,
) -> list[ColumnElement[bool]]:
    """The application and organization predicates for one column, as a list.

    Takes the two ids rather than a :class:`WindowFilter` because three of the
    sites have no window to scope by at all -- a user roster, a set of
    pre-built ``where`` clauses -- and a version that forced them to
    manufacture a dummy window to reach the shared logic would be a version
    they eventually stop using.

    ``None`` for both ids means the platform principal, and adds *no* predicate
    rather than a permissive one. That asymmetry is the whole point: a tenant's
    ``organization_id`` is never ``None``, so there is no spelling of this call
    that means "everything" for a tenant.

    The organization predicate is a semi-join -- ``col IN (SELECT id FROM
    applications WHERE organization_id = :org)`` -- rather than a join. That
    choice is load-bearing: a real join would add ``applications`` to the FROM
    list of every aggregate query and perturb how SQLAlchemy and PostgreSQL
    infer ``GROUP BY``, changing the *shape* of results that were correct
    before. A subquery adds no column to the outer FROM and cannot do that.

    ``application_id_column`` is the nullable ``application_id`` FK on the
    model being filtered (``Trace``, ``LLMCall``, ...). Applications with
    ``organization_id IS NULL`` -- rows that predate the migration, or an
    application whose organization was deleted (``ondelete="SET NULL"``) --
    belong to no company and are therefore invisible to every tenant. That is
    the safe direction: an orphaned row is withheld, not handed to whichever
    company asked.

    The same ``IN`` has a second consequence that matters at the sites whose
    column is nullable: a row with ``application_id IS NULL`` never satisfies
    ``IN (...)``, because SQL's three-valued logic cannot match ``NULL``.
    ``llm_calls``, ``retrieval_calls``, ``anomalies``, ``experiments``,
    ``evaluation_runs`` and ``optimization_recommendations`` all allow an
    unattached call, and for a tenant those rows are correctly withheld. It
    would *not* be right for the platform principal, and it is not: platform
    scope leaves both ids ``None``, which adds no predicate at all.
    """
    predicates: list[ColumnElement[bool]] = []
    if application_id is not None:
        predicates.append(application_id_column == application_id)
    if organization_id is not None:
        predicates.append(
            application_id_column.in_(
                select(Application.id).where(
                    Application.organization_id == organization_id
                )
            )
        )
    return predicates


def window_predicates(
    window: WindowFilter | None, application_id_column: Any
) -> list[ColumnElement[bool]]:
    """:func:`tenant_predicates` for a scope that arrives as a ``WindowFilter``.

    Split out because not every filter site constrains a ``Select``.
    :mod:`app.services.recommendation_service` builds ``where`` *clauses* and
    splats them into several statements, and a site that could not reach the
    shared predicate builder would be a site where the organization predicate
    is one forgotten line away.
    """
    if window is None:
        return []
    return tenant_predicates(
        window.application_id, window.organization_id, application_id_column
    )


def apply_tenant_filter(
    stmt: Select[Any], window: WindowFilter | None, application_id_column: Any
) -> Select[Any]:
    """Constrain ``stmt`` to one application and/or one organization.

    The common shape, and the reason the :class:`WindowFilter` docstring can
    claim that "unscoped" has exactly one meaning. Every filter site in the
    read path calls this instead of testing ``application_id`` on its own,
    because a site that filters by application but forgets the organization is
    a cross-tenant leak that reads as correct.

    ``application_id_column`` must be a column holding **application** ids.
    Passing ``Application.organization_id`` looks reasonable and is a silent
    leak: the organization predicate would compile to ``organization_id IN
    (SELECT id FROM applications WHERE organization_id = :org)``, comparing an
    organization id against application ids, which matches nothing -- a tenant
    sees an empty list rather than an error, so the bug survives a manual test
    and is read as "this company has no applications". Rows whose
    ``organization_id`` is ``NULL`` (pre-migration rows, or an application whose
    organization was deleted) belong to no company and are withheld.

    For a query *on* the applications table itself, use
    :func:`apply_organization_filter`, which is the one shape this cannot
    express.

    ``None`` for both scope fields adds no predicate, so a platform-wide query
    compiles to exactly the SQL it did before this helper existed -- which is
    what keeps the pre-tenancy output byte-identical.
    """
    predicates = window_predicates(window, application_id_column)
    return stmt.where(*predicates) if predicates else stmt


def apply_organization_filter(
    stmt: Select[Any], organization_id: uuid.UUID | None, organization_id_column: Any
) -> Select[Any]:
    """Constrain a query on ``applications`` to one organization.

    The applications table carries the organization directly, so it is the one
    site where the tenant seam is *not* a ``application_id`` FK, and the one
    shape :func:`apply_tenant_filter` cannot express -- its organization
    predicate is always expressed as a semi-join back onto this same table.

    Kept beside it rather than inlined at the two call sites so that the reason
    the two differ is written down once, next to the helper someone will
    otherwise "simplify" by calling the wrong one.

    ``None`` means the platform principal and adds no predicate, which is how a
    local demo with no organizations configured keeps listing every application.
    """
    if organization_id is None:
        return stmt
    return stmt.where(organization_id_column == organization_id)


def apply_pagination(
    stmt: Select[Any], page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
) -> Select[Any]:
    """Apply LIMIT/OFFSET, clamping ``page_size`` so a caller cannot ask for
    the whole table by accident."""
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    return stmt.limit(page_size).offset((page - 1) * page_size)


def count_rows(stmt: Select[Any]) -> Select[int]:
    """Wrap a filtered statement so the caller can read the unpaginated total.

    ``order_by(None)`` is required: PostgreSQL cannot ``COUNT(*)`` over a
    subquery that still carries a sort, and the ordering is meaningless for a
    count.
    """
    return select(func.count()).select_from(stmt.order_by(None).subquery())


async def get_application_by_name(
    session: AsyncSession, name: str, organization_id: uuid.UUID | None = None
) -> Application | None:
    """Look up an application by its name, optionally within one organization.

    ``organization_id`` of ``None`` is platform-wide and is only reached through
    the auth escape hatch. A tenant passes its own id, so ``?application=<name>``
    cannot be used to read another company's dashboard: the name simply does not
    resolve, and the caller gets ``found=False`` (a 404 on the routes that can
    express one) rather than somebody else's numbers.

    The name is globally unique (``ix_applications_name``), so this cannot
    return ``MultipleResultsFound``; the predicate is a narrowing, not a
    second key. That global uniqueness is a deliberate product limitation --
    see the note in :func:`get_or_create_application`.
    """
    stmt = select(Application).where(Application.name == name)
    if organization_id is not None:
        stmt = stmt.where(Application.organization_id == organization_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_or_create_application(
    session: AsyncSession,
    name: str,
    *,
    description: str | None = None,
    environment: str = "development",
    extra_metadata: dict[str, Any] | None = None,
    organization_id: uuid.UUID | None = None,
) -> Application:
    """Fetch an application by name, creating it when this is the first sighting.

    Uses ``ON CONFLICT DO NOTHING ... RETURNING id`` so two collectors racing on
    the same brand-new application cannot both insert; the loser re-reads the
    row the winner wrote. Callers own the transaction, so this does not commit.

    ``organization_id`` is the writing tenant, and it is checked on the way in
    rather than on the way out. ``applications.name`` is globally unique, so a
    name already owned by another company cannot be *created* by this one -- the
    insert would conflict -- but it can still be *found*, and returning another
    company's row to a caller who just named it is exactly the silent
    misplacement this refuses: the trace would then be written into a
    competitor's application, where they would see it and this company would
    not. So a name that exists under a different organization is an error, not a
    hit. The error is deliberately loud and names the owner, because the caller
    has to change something -- a different application name, or a different
    account -- and a quiet empty dashboard would be indistinguishable from
    "this application is simply not ingesting".

    ``None`` is the platform principal and skips the check, which is what keeps
    the local demo and the shared key working on names that already exist.
    """
    existing = await get_application_by_name(session, name)
    if existing is not None:
        _assert_same_organization(existing, name, organization_id)
        return existing

    insert_stmt = (
        pg_insert(Application)
        .values(
            name=name,
            description=description,
            environment=environment,
            extra_metadata=extra_metadata,
            organization_id=organization_id,
        )
        .on_conflict_do_nothing(index_elements=[Application.name])
        .returning(Application.id)
    )
    new_id: uuid.UUID | None = (await session.execute(insert_stmt)).scalar_one_or_none()

    if new_id is None:
        # Lost the race: another writer inserted this name first.
        logger.debug("application.create.race_resolved", application_name=name)
        resolved = await get_application_by_name(session, name)
        if resolved is None:  # pragma: no cover - only if the row was deleted mid-flight
            raise RuntimeError(f"Application {name!r} conflicted but could not be re-read")
        # The race is not a bypass. Somebody else may have won it precisely
        # because this name belongs to their organization, so the same check
        # runs again on the row that was actually resolved.
        _assert_same_organization(resolved, name, organization_id)
        return resolved

    logger.info(
        "application.created",
        application_name=name,
        environment=environment,
        organization_id=str(organization_id) if organization_id else None,
    )
    created = await session.get(Application, new_id)
    if created is None:  # pragma: no cover - the INSERT just returned this id
        raise RuntimeError(f"Application {name!r} was inserted but could not be re-read")
    return created


def _assert_same_organization(
    application: Application, name: str, organization_id: uuid.UUID | None
) -> None:
    """Refuse an application name that already belongs to another company.

    An application with no organization is treated as matching everyone. That is
    the pre-tenancy state -- every row created before this column existed -- and
    refusing it would strand the demo data and break the local setup. It is also
    the safe direction: an unowned application holds no other company's data, so
    a tenant claiming it narrows nobody's view.
    """
    if organization_id is None:
        return
    if application.organization_id in (None, organization_id):
        return
    raise ApplicationOwnedByAnotherOrganization(
        name=name,
        owner=application.organization_id,
    )


async def get_or_create_user(
    session: AsyncSession,
    application_id: uuid.UUID,
    external_id: str,
    *,
    display_name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> User:
    """Fetch a user by ``(application_id, external_id)``, creating on first sight.

    Keyed on the composite unique constraint rather than the external id alone
    so two applications can legitimately use the same ``user-42``.
    """
    existing = await get_user_for_external_id(session, application_id, external_id)
    if existing is not None:
        return existing

    insert_stmt = (
        pg_insert(User)
        .values(
            application_id=application_id,
            external_id=external_id,
            display_name=display_name,
            extra_metadata=extra_metadata,
        )
        .on_conflict_do_nothing(index_elements=[User.application_id, User.external_id])
        .returning(User.id)
    )
    new_id: uuid.UUID | None = (await session.execute(insert_stmt)).scalar_one_or_none()

    if new_id is None:
        logger.debug(
            "user.create.race_resolved",
            application_id=str(application_id),
            external_id=external_id,
        )
        resolved = await get_user_for_external_id(session, application_id, external_id)
        if resolved is None:  # pragma: no cover - only if the row was deleted mid-flight
            raise RuntimeError(
                f"User {external_id!r} conflicted but could not be re-read"
            )
        return resolved

    created = await session.get(User, new_id)
    if created is None:  # pragma: no cover - the INSERT just returned this id
        raise RuntimeError(f"User {external_id!r} was inserted but could not be re-read")
    return created


async def get_user_for_external_id(
    session: AsyncSession, application_id: uuid.UUID, external_id: str
) -> User | None:
    """Single-row lookup of a user by its natural key. ``None`` when unknown."""
    result = await session.execute(
        select(User).where(
            User.application_id == application_id, User.external_id == external_id
        )
    )
    return result.scalar_one_or_none()
