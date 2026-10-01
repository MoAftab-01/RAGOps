"""Shared pytest fixtures.

Two kinds of test live in this suite and they are kept apart deliberately:

* **Unit tests** exercise pure functions -- the token counter, the pricing
  table, the IR metrics, the anomaly detector. They need nothing running and
  must never touch the network, a database, or a model.
* **Integration tests** exercise the API against a real PostgreSQL. They are
  marked ``integration`` and skip themselves when no database is reachable, so
  ``pytest`` on a bare checkout is a meaningful signal rather than an
  error storm.

There is no mocking library in the dependency set. Rather than add one for a
handful of call sites, the tests build real objects and pass real callables;
the two places that genuinely need an HTTP double use ``httpx``'s own
transport interface, which ships with the project.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

# The app reads configuration from the environment at import time. Tests must
# not inherit a developer's local .env, and they must not require one.
os.environ.setdefault("RAGOPS_ENVIRONMENT", "test")
os.environ.setdefault("RAGOPS_API_KEY", "")

#: Prefix that marks a trace as belonging to a test run. Integration tests
#: assert on counts scoped to their own application name, never on totals, so a
#: populated demo database cannot make them flaky.
TEST_APP_PREFIX = "pytest-"

#: The shared key tenant tests present. A fixed literal in a test file, so the
#: legacy "presented key matches the shared key" branch of the resolver is
#: taken by exactly the tests that mean to take it and cannot drift between
#: runs. Not a secret and deliberately shaped unlike a real key: it is asserted
#: so in ``test_tenancy_organizations``, because a test file is the one place a
#: credential ends up in a CI log.
TEST_SHARED_KEY = "test-shared-key"


def _database_is_configured() -> bool:
    """Whether a database is configured, by the same rules the app uses.

    This consults ``app.config.settings`` rather than ``os.environ`` alone, and
    that distinction is the whole reason the tenant-isolation tests used to skip
    silently. ``settings.database_url`` carries a working default, so the app
    reached PostgreSQL and served a dashboard while this hook -- reading the
    environment only -- saw nothing and skipped every ``integration`` test. A
    green run that had never executed a single assertion is worse than a red
    one: it reports coverage that does not exist.

    The environment is still checked first, so a shell that exports
    ``DATABASE_URL`` is honoured without importing settings, and so an explicit
    variable keeps winning over the default exactly as it does in the app.
    """
    if os.environ.get("RAGOPS_DATABASE_URL") or os.environ.get("DATABASE_URL"):
        return True
    from app.config import settings

    return bool(settings.database_url)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip integration tests when no database is configured.

    A missing database is not a test failure -- it means this checkout has not
    been set up yet. Reporting it as an error would train people to ignore red.
    """
    if _database_is_configured():
        return
    skip = pytest.mark.skip(
        reason=(
            "No database configured; integration tests need a live PostgreSQL. "
            "Set DATABASE_URL (or RAGOPS_DATABASE_URL)."
        )
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Session-scoped loop for the session-scoped async fixtures below."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def unique_app_name(request: pytest.FixtureRequest) -> str:
    """A per-test application name that will never collide with real data."""
    return f"{TEST_APP_PREFIX}{request.node.name}"[:128]


class RecordingTransport:
    """A minimal async HTTP transport for provider tests.

    ``httpx`` ships its own ``AsyncBaseTransport`` interface, so exercising the
    provider client against a canned response needs no new dependency. Every
    request is recorded so a test can assert on what was actually sent --
    which is how the token-counter parity test proves the SDK and the server
    compute the same number instead of merely assuming it.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.requests: list[Any] = []
        self._responses = list(responses)

    async def handle_async_request(self, request: Any) -> Any:  # pragma: no cover
        import httpx

        self.requests.append(request)
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {request.url}")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return httpx.Response(
            status_code=getattr(nxt, "status_code", 200),
            json=getattr(nxt, "json", {}),
            request=request,
        )


@pytest.fixture
def recording_transport() -> type[RecordingTransport]:
    return RecordingTransport


@pytest.fixture
async def db_session(
    unique_app_name: str,
) -> AsyncIterator[Any]:
    """A real session against the configured database.

    Everything the test writes is removed afterwards. The deletion is scoped by
    application id rather than truncating tables: truncating would destroy a
    developer's demo dataset, and an integration test has no business doing
    that to someone's machine.
    """
    from app.core.database import dispose_engine, session_scope
    from app.models import Application
    from sqlalchemy import delete

    async with session_scope() as session:
        application = Application(
            name=unique_app_name,
            description="created by pytest",
            environment="test",
            is_active=True,
        )
        session.add(application)
        await session.commit()
        application_id = application.id

    try:
        async with session_scope() as session:
            yield session
    finally:
        async with session_scope() as session:
            await session.execute(
                delete(Application).where(Application.id == application_id)
            )
            await session.commit()
        await dispose_engine()


# ---------------------------------------------------------------------------
# Endpoint-level fixtures (integration)
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_key_enabled(monkeypatch: Any) -> None:
    """Give the suite a non-empty shared key so presented keys resolve.

    ``RAGOPS_API_KEY`` is set to ``""`` above, and that empty string is the
    deliberate escape hatch: with no shared key configured, *any* presented
    ``X-API-Key`` resolves to the platform principal without a database lookup.
    Right for the local demo, fatal for tenant tests -- every request would be
    platform-scoped, every cross-tenant assertion would pass for the wrong
    reason, and a test asserting that an unknown key is rejected would fail
    truthfully about a config the test suite itself put in place.

    The hatch is not removed and not weakened. It is switched off for the tests
    that are specifically about tenancy being enforced, which request this
    fixture by name; the tests that assert the hatch works set the empty value
    back themselves.

    It is deliberately *not* autouse. An autouse fixture here would silently
    invert a setting the unit suite depends on, and the next test file to be
    added would inherit a tenancy config it never asked for.
    """
    from app.core import security

    monkeypatch.setattr(security.settings, "ragops_api_key", TEST_SHARED_KEY)


@pytest.fixture
async def api_client(db_session: Any) -> AsyncIterator[Any]:
    """An ``httpx`` client bound to the app in-process.

    ``ASGITransport`` skips the socket entirely, so these tests need no running
    server and cannot pass by accidentally exercising a stale one -- a real
    hazard here, since a long-lived ``uvicorn`` on :8000 is usually running
    during development and predates whatever the working tree now says.

    ``get_db`` is overridden with the session :func:`db_session` manages, and
    that is what makes teardown possible at all: the rows a test wrote are the
    rows the fixture deletes. The override is installed and removed around the
    yield rather than at session scope, so a test that forgets to clean up still
    cannot leak the override into the next one.
    """
    import httpx

    from app.core.database import get_db
    from app.main import app

    async def _override() -> AsyncIterator[Any]:
        yield db_session

    app.dependency_overrides[get_db] = _override
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://ragops.test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def org_factory(db_session: Any) -> AsyncIterator[Any]:
    """Create organizations, each with one ingest-and-read API key.

    Yields a callable taking a name and returning ``(organization, plaintext)``.
    The plaintext exists in memory for the length of one test and is dropped
    with it: never stored, never returned by a fixture that could print it, and
    never written into a file as a literal.

    Two details here were bugs before they were comments. ``_make`` is async
    because an un-awaited ``flush()`` on an AsyncSession is a coroutine that is
    never run -- the INSERT silently does not happen, ``organization.id`` stays
    ``None``, and the ApiKey that references it fails the NOT NULL constraint
    several statements later with an error pointing at the key rather than at
    the missing await. And teardown deletes applications before organizations,
    because deleting an organization CASCADEs its keys but only SET NULLs its
    applications, leaving orphans that quietly crowd a paginated list off page
    one on the next run.
    """
    from sqlalchemy import delete

    from app.core.security import generate_api_key
    from app.models import ApiKey, Application, Organization

    created: list[Any] = []

    async def _make(name: str, *, with_key: bool = True) -> tuple[Any, str | None]:
        organization = Organization(
            name=f"{name}-{uuid.uuid4().hex[:8]}",
            slug=f"test-{uuid.uuid4().hex[:12]}",
            is_active=True,
        )
        db_session.add(organization)
        await db_session.flush()

        plaintext: str | None = None
        if with_key:
            plaintext, key_hash = generate_api_key()
            db_session.add(
                ApiKey(
                    organization_id=organization.id,
                    name="test-key",
                    key_prefix=plaintext[:12],
                    key_hash=key_hash,
                    scopes="ingest,read",
                )
            )
            await db_session.flush()

        created.append(organization.id)
        return organization, plaintext

    yield _make

    # Right for production -- deleting a company must not delete a company's
    # telemetry -- and wrong for a test suite, where every run would leave more
    # rows behind than the last until a paginated list no longer fits the
    # current test's own fixtures on page one.
    for organization_id in created:
        await db_session.execute(
            delete(Application).where(Application.organization_id == organization_id)
        )
        await db_session.execute(
            delete(Organization).where(Organization.id == organization_id)
        )
    await db_session.commit()
