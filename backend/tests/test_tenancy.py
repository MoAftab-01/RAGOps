"""Endpoint tests for tenant isolation.

This is the first file in the repo to drive the API over HTTP. Everything before
it tested pure functions or called repositories directly, which is a real gap:
the leaks steps 5 and 6 close are all in the *routing* layer -- a route that
forgets to take a principal, or takes one and never uses it -- and a repository
test cannot see any of them.

Two rules this file follows without exception:

* **No key is ever printed, logged, or asserted on as a literal.** Keys are
  generated in-process, handed straight to the client, and compared by hash. A
  test suite is the one place where a credential escapes into CI logs, so the
  rule is structural: there is no statement in this file that could write a key
  out.
* **Isolation is asserted negatively.** The strongest statement is "the other
  org's row is not in the response", because it fails both when the predicate is
  missing and when it is inverted.

Everything written here is namespaced with a uuid and torn down by fixture, so
a populated demo database cannot make this file flaky.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

pytestmark = [
    pytest.mark.integration,
    # Requested explicitly rather than autouse. ``shared_key_enabled`` inverts
    # the empty-shared-key escape hatch the unit suite relies on, so it must
    # apply to the tests asserting tenancy is enforced and to nothing else --
    # ``test_the_platform_escape_hatch_still_sees_everything`` below sets the
    # empty value back for its own duration.
    pytest.mark.usefixtures("shared_key_enabled"),
]

#: The shape :func:`~app.core.security.generate_api_key` promises. Asserted
#: rather than restated from the implementation: a key that silently stopped
#: matching its own scheme would still authenticate, and this is the only
#: place the scheme is written down twice.
API_KEY_PATTERN = re.compile(r"^rag_[A-Za-z0-9_-]{43}$")


async def _make_application(
    session: Any, organization_id: Any, prefix: str
) -> Any:
    """An application owned by one organization."""
    from app.models import Application

    application = Application(
        name=f"pytest-{prefix}-{uuid.uuid4().hex[:6]}",
        organization_id=organization_id,
        environment="test",
        is_active=True,
    )
    session.add(application)
    await session.commit()
    return application


async def _all_application_names(
    client: Any, headers: dict[str, str] | None = None
) -> set[str]:
    """Every application name the API will show, across all pages.

    The list route paginates at 25 by default and orders by creation, so a
    test that reads one page is asserting against a *window* onto a shared
    database rather than against the tenant predicate it means to test. A
    populated demo database -- or a previous run that failed before its own
    teardown -- pushes the current test's freshly created applications off the
    end, and the assertion then fails on a row that is present and correctly
    scoped. Walking the pages keeps the test honest about what it is checking.
    """
    names: set[str] = set()
    page = 1
    while True:
        response = await client.get(
            "/api/applications", params={"page": page, "page_size": 100}, headers=headers
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        names.update(item["name"] for item in payload["items"])
        if not payload["has_next"]:
            return names
        page += 1


async def _make_trace(session: Any, application_id: Any) -> Any:
    """A minimal trace row, so there is something to protect."""
    from app.models import Trace

    trace = Trace(
        application_id=application_id,
        trace_id=f"pytest-{uuid.uuid4().hex[:10]}",
        name="pytest-tenancy",
        kind="chat",
        status="success",
        start_time=datetime.now(timezone.utc),
        duration_ms=1,
    )
    session.add(trace)
    await session.commit()
    return trace


async def _llm_call_count(session: Any) -> int:
    from sqlalchemy import func, select

    from app.models import LLMCall

    return int(
        (await session.execute(select(func.count(LLMCall.id)))).scalar_one()
    )


# ---------------------------------------------------------------------------
# Read isolation
# ---------------------------------------------------------------------------


async def test_a_tenant_sees_only_its_own_applications(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """``GET /api/applications`` is confined; org names are not public."""
    org_a, key_a = await org_factory("Acme A")
    org_b, key_b = await org_factory("Globex B")
    app_a = await _make_application(db_session, org_a.id, "a")
    app_b = await _make_application(db_session, org_b.id, "b")

    names = await _all_application_names(api_client, {"X-API-Key": key_a})
    assert app_a.name in names, "org A cannot see its own application"
    assert app_b.name not in names, "org A can enumerate org B's applications"

    other_names = await _all_application_names(api_client, {"X-API-Key": key_b})
    assert app_b.name in other_names
    assert app_a.name not in other_names, "org B can enumerate org A's applications"


async def test_a_tenant_cannot_read_another_orgs_trace(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """``GET /api/traces/{id}`` returns full prompts, so it is 404 not 403.

    404 rather than 403 because a 403 confirms the id exists -- which is the
    enumeration channel. Both directions are asserted: fixing only the forward
    case leaves the symmetric hole open.
    """
    org_a, key_a = await org_factory("Reader A")
    org_b, key_b = await org_factory("Reader B")
    app_a = await _make_application(db_session, org_a.id, "own")
    app_b = await _make_application(db_session, org_b.id, "foreign")
    trace_a = await _make_trace(db_session, app_a.id)
    trace_b = await _make_trace(db_session, app_b.id)

    mine = await api_client.get(
        f"/api/traces/{trace_a.trace_id}", headers={"X-API-Key": key_a}
    )
    assert mine.status_code == 200, mine.text

    cross = await api_client.get(
        f"/api/traces/{trace_b.trace_id}", headers={"X-API-Key": key_a}
    )
    assert cross.status_code == 404, (
        f"cross-tenant trace read returned {cross.status_code}, not 404"
    )

    reverse = await api_client.get(
        f"/api/traces/{trace_a.trace_id}", headers={"X-API-Key": key_b}
    )
    assert reverse.status_code == 404, (
        f"reverse cross-tenant read returned {reverse.status_code}, not 404"
    )


async def test_a_tenant_cannot_read_another_orgs_application_stats(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """``GET /api/applications/{id}/stats`` 404s rather than reporting totals.

    The read that a bare id selects directly, with no ``?application=`` to
    confuse -- the shape that most invites a missing predicate.
    """
    org_a, key_a = await org_factory("Stats A")
    org_b, _ = await org_factory("Stats B")
    app_b = await _make_application(db_session, org_b.id, "target")

    response = await api_client.get(
        f"/api/applications/{app_b.id}/stats?window=90d", headers={"X-API-Key": key_a}
    )
    assert response.status_code == 404, (
        f"cross-tenant stats read returned {response.status_code}, not 404"
    )


async def test_a_tenant_dashboard_shows_only_its_own_totals(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """The aggregate is the leak that per-row checks miss.

    Even if every individual row were correctly withheld, a shared ``GROUP BY``
    would still add another company's trace counts and token totals into one
    number. Asserting on the *contents* of ``application_usage`` catches that
    even though the total is shared.
    """
    org_a, key_a = await org_factory("Dash A")
    org_b, _ = await org_factory("Dash B")
    app_a = await _make_application(db_session, org_a.id, "visible")
    app_b = await _make_application(db_session, org_b.id, "hidden")
    await _make_trace(db_session, app_a.id)
    await _make_trace(db_session, app_b.id)

    response = await api_client.get(
        "/api/dashboard/overview?window=90d", headers={"X-API-Key": key_a}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    breakdown = {row["application"] for row in payload["application_usage"]}
    assert app_b.name not in breakdown, (
        "org A's dashboard breakdown includes org B's application"
    )
    assert payload["total_traces"] == 1, (
        f"org A counted {payload['total_traces']} traces, expected only its own 1"
    )


# ---------------------------------------------------------------------------
# Write isolation
# ---------------------------------------------------------------------------


async def test_a_second_tenant_cannot_take_an_existing_application_name(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """A name collision is a 409, and no trace is stored against the victim's app.

    The worst possible failure here is a 201: org B posts ``support-bot``,
    org A already owns it, and the trace lands in org A's data. Org B sees a
    success and an empty dashboard; org A sees a trace it never sent. Both are
    quietly wrong, which is why the status code and the ``LLMCall``/``Trace``
    count are both asserted rather than just the response body.

    A second trace under a *free* name still works, so the rejection is
    specific to the collision and not a blanket refusal of org B's writes.
    """
    from sqlalchemy import func, select

    from app.models import Application, Trace

    org_a, key_a = await org_factory("Name Owner")
    org_b, key_b = await org_factory("Name Squatter")
    taken = f"pytest-taken-{uuid.uuid4().hex[:8]}"
    owned = await api_client.post(
        "/api/traces",
        headers={"X-API-Key": key_a},
        json={"application": taken, "trace_id": f"pytest-{uuid.uuid4().hex[:10]}"},
    )
    assert owned.status_code == 201, owned.text

    traces_before = int(
        (await db_session.execute(select(func.count(Trace.id)))).scalar_one()
    )

    collision = await api_client.post(
        "/api/traces",
        headers={"X-API-Key": key_b},
        json={"application": taken, "trace_id": f"pytest-{uuid.uuid4().hex[:10]}"},
    )
    assert collision.status_code == 409, (
        f"a name owned by another organization returned {collision.status_code}, "
        "not 409 -- a 201 here means org B's telemetry was written into org A's "
        "application"
    )
    # The error must not name the other company: two companies discovering each
    # other's existence through an error message is the enumeration channel the
    # 404 decisions elsewhere in this file exist to close.
    assert org_a.name not in collision.text, "the error discloses the owning company"
    assert int(
        (await db_session.execute(select(func.count(Trace.id)))).scalar_one()
    ) == traces_before, "a rejected write still stored a trace"

    free = await api_client.post(
        "/api/traces",
        headers={"X-API-Key": key_b},
        json={"application": f"pytest-free-{uuid.uuid4().hex[:8]}", "trace_id": f"pytest-{uuid.uuid4().hex[:10]}"},
    )
    assert free.status_code == 201, free.text
    created = (
        await db_session.execute(
            select(Application).where(Application.name.startswith("pytest-free-"))
        )
    ).scalars().all()
    assert [row.organization_id for row in created] == [org_b.id], (
        "org B's own write was not attributed to org B"
    )


async def test_a_tenant_cannot_append_to_another_orgs_trace(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """The highest-severity regression guarded here.

    ``_application_for`` feeds three write routes; before step 5 any key could
    resolve any trace's owning application and append a generation to it. The
    assertion that matters is not only the 404 -- it is that no ``LLMCall`` row
    appeared, because a rejected write that still committed would be worse than
    no guard at all.
    """
    org_a, key_a = await org_factory("Writer A")
    org_b, _ = await org_factory("Writer B")
    app_b = await _make_application(db_session, org_b.id, "victim")
    trace_b = await _make_trace(db_session, app_b.id)

    calls_before = await _llm_call_count(db_session)
    response = await api_client.post(
        f"/api/traces/{trace_b.trace_id}/generation",
        headers={"X-API-Key": key_a},
        json={"model": "llama3.2:1b", "input_tokens": 10, "output_tokens": 5},
    )
    assert response.status_code == 404, (
        f"cross-tenant write returned {response.status_code}, not 404"
    )
    assert await _llm_call_count(db_session) == calls_before, (
        "a rejected cross-tenant write still created an LLMCall"
    )


async def test_a_tenant_writes_land_in_its_own_organization(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """The positive half: a tenant's own write works and is attributed to it.

    Pinned because the write path is scoped too. A guard that rejected every
    cross-tenant write would pass the test above, and this is what catches it --
    "no data ever lands" and "data lands in the right place" are different
    guarantees, and only this one distinguishes them.
    """
    from sqlalchemy import select

    from app.models import Application

    org_a, key_a = await org_factory("Owner")
    application_name = f"pytest-own-{uuid.uuid4().hex[:8]}"

    response = await api_client.post(
        "/api/traces",
        headers={"X-API-Key": key_a},
        json={
            "application": application_name,
            "trace_id": f"pytest-{uuid.uuid4().hex[:10]}",
            "name": "own-write",
            "kind": "chat",
        },
    )
    assert response.status_code == 201, response.text

    application = (
        await db_session.execute(
            select(Application).where(Application.name == application_name)
        )
    ).scalar_one_or_none()
    assert application is not None, "the write created no application to own"
    assert application.organization_id == org_a.id, (
        "the new application was not attributed to the writing organization"
    )


async def test_a_revoked_key_is_rejected_and_writes_nothing(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """Revocation is enforced at the write, and the 401 is not a 403.

    A 403 would say "this key exists and is revoked", which is a small but real
    oracle. The key also must not be able to create the trace it was refused.
    """
    from sqlalchemy import select

    from app.models import ApiKey

    organization, plaintext = await org_factory("Revoked Inc")
    key_row = (
        await db_session.execute(
            select(ApiKey).where(ApiKey.organization_id == organization.id)
        )
    ).scalar_one()
    key_row.revoked_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await db_session.commit()

    calls_before = await _llm_call_count(db_session)
    response = await api_client.post(
        "/api/traces",
        headers={"X-API-Key": plaintext},
        json={
            "application": f"pytest-revoked-{uuid.uuid4().hex[:8]}",
            "trace_id": f"pytest-{uuid.uuid4().hex[:10]}",
            "name": "revoked",
        },
    )
    assert response.status_code == 401, (
        f"a revoked key was accepted ({response.status_code})"
    )
    assert await _llm_call_count(db_session) == calls_before


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


async def test_the_plaintext_key_is_never_persisted(
    db_session: Any, org_factory: Any
) -> None:
    """The stored row holds the hash; no column anywhere holds the plaintext.

    Asserted over every string column of every key row rather than by naming
    ``key_hash``, because a future column -- a ``last_seen_key``, an audit note
    -- would leak the key just as badly and a narrower assertion would not see
    it.
    """
    from sqlalchemy import select

    from app.core.security import hash_api_key
    from app.models import ApiKey

    _organization, plaintext = await org_factory("Hash Check")
    assert plaintext is not None

    rows = (await db_session.execute(select(ApiKey))).scalars().all()
    matching = [row for row in rows if row.key_hash == hash_api_key(plaintext)]
    assert matching, "the created key is not stored under its own hash"

    for row in rows:
        for column in ApiKey.__table__.columns:
            value = getattr(row, column.name, None)
            if isinstance(value, str):
                assert plaintext not in value, (
                    f"plaintext present in api_keys.{column.name}"
                )


async def test_generated_keys_match_their_documented_shape() -> None:
    """The scheme is part of the contract, so it is asserted rather than assumed.

    A pure unit test with no database and no key material: only the *shape* is
    checked, so this cannot fail by leaking anything.
    """
    from app.core.security import generate_api_key

    plaintext, key_hash = generate_api_key()
    assert API_KEY_PATTERN.match(plaintext), "generated key does not match its scheme"
    assert len(key_hash) == 64, "the hash is not a hex sha256 digest"
    assert key_hash != plaintext, "the key was stored in the clear"
    assert generate_api_key()[0] != plaintext, "two generated keys collided"


# ---------------------------------------------------------------------------
# The escape hatch
# ---------------------------------------------------------------------------


async def test_the_platform_escape_hatch_still_sees_everything(
    api_client: Any, db_session: Any, org_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key presented resolves to platform, and platform still sees both.

    Asserted *positively*. "The tenant tests passed" is also consistent with
    "every request 404s", and only this test tells the difference -- it is the
    guard against tightening the escape hatch until the local demo stops
    booting.

    The autouse fixture sets a non-empty shared key, so this test puts the
    empty one back. That is the point of it: an *absent* header must still
    resolve to platform even on a deployment that has key resolution turned
    on, which is a separate promise from the one the other tests check.
    """
    from app.core import security

    monkeypatch.setattr(security.settings, "ragops_api_key", "")

    org_a, _ = await org_factory("Open A", with_key=False)
    org_b, _ = await org_factory("Open B", with_key=False)
    app_a = await _make_application(db_session, org_a.id, "open-a")
    app_b = await _make_application(db_session, org_b.id, "open-b")

    names = await _all_application_names(api_client)
    assert app_a.name in names and app_b.name in names, (
        "the platform principal no longer sees both organizations -- the "
        "escape hatch is broken and the local demo will not boot"
    )


async def test_a_presented_key_beats_an_empty_shared_key(
    api_client: Any, db_session: Any, org_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty-key hatch applies to *any* presented key -- stated, not hidden.

    With ``ragops_api_key`` empty, a key that resolves to no organization is
    accepted as platform rather than rejected. That is the pre-tenancy
    behaviour and it is what makes ``clone && docker compose up`` work with no
    key at all, so it is kept deliberately.

    It is pinned here because the alternative is a silent surprise: a
    deployment with an empty shared key that also has real organizations
    believes it is authenticated, and every one of its keys is being ignored.
    The behaviour is correct; the *documentation* of it is the fragile part,
    and this is the test that keeps the two in step.
    """
    from app.core import security

    monkeypatch.setattr(security.settings, "ragops_api_key", "")

    org_a, key_a = await org_factory("Hatch A")
    app_a = await _make_application(db_session, org_a.id, "hatch")
    await _make_trace(db_session, app_a.id)

    names = await _all_application_names(api_client, {"X-API-Key": key_a})
    assert app_a.name in names, "a real key stopped resolving once orgs exist"

    unknown = await api_client.get(
        "/api/applications", headers={"X-API-Key": "rag_not-a-real-key-at-all"}
    )
    assert unknown.status_code == 200, (
        "an empty shared key no longer accepts any presented key -- the local "
        "demo with no key configured would stop working"
    )


async def test_an_unknown_key_is_rejected_rather_than_degraded(
    db_session: Any, api_client: Any
) -> None:
    """A *presented* wrong key is 401; only an *absent* header is the escape hatch.

    This is the distinction the whole design rests on: omitting the header keeps
    reads open for a local demo, but presenting a key that resolves to nothing
    must never silently widen to platform-wide, or the auth is decorative.
    """
    response = await api_client.get(
        "/api/applications", headers={"X-API-Key": "rag_definitely-not-a-real-key"}
    )
    assert response.status_code == 401, (
        f"an unknown key was accepted ({response.status_code})"
    )


# ---------------------------------------------------------------------------
# Cache keying
# ---------------------------------------------------------------------------


def test_cache_keys_differ_across_organizations() -> None:
    """Two organizations reading the same window must not share a cache entry.

    A pure unit test, no database and no Redis: ``make_key`` hashes the parts,
    so this asserts on the keys themselves. It lives here because the failure it
    guards is invisible in every other test in this file -- two tenants reading
    one window would each see stale, plausible, *wrong* numbers rather than an
    error.
    """
    from app.core.cache import make_key
    from app.services.window import AnalyticsScope, cache_parts

    def scope_for(organization_id: uuid.UUID | None) -> AnalyticsScope:
        now = datetime.now(timezone.utc)
        return AnalyticsScope(
            start=now,
            end=now,
            label="90d",
            application_id=None,
            application_name=None,
            found=True,
            organization_id=organization_id,
        )

    tenant_a = make_key(
        "analytics", **cache_parts(scope_for(uuid.uuid4()), "dashboard-overview")
    )
    tenant_b = make_key(
        "analytics", **cache_parts(scope_for(uuid.uuid4()), "dashboard-overview")
    )
    platform = make_key("analytics", **cache_parts(scope_for(None), "dashboard-overview"))

    assert tenant_a != tenant_b, "two organizations share one cache key"
    assert platform not in (tenant_a, tenant_b), (
        "the platform key collides with a tenant key"
    )


# ---------------------------------------------------------------------------
# Structural invariant
# ---------------------------------------------------------------------------


def test_no_bare_application_id_filter_sites_remain() -> None:
    """Re-assert the step-4 guard here so the two copies cannot drift.

    ``test_scoping`` owns the predicate; this copy exists so that deleting the
    check from there fails *this* file loudly rather than losing it silently.
    """

    from tests.test_scoping import APP_ROOT, TestFilterSiteGuard

    bare = [
        (file, line)
        for file, line in TestFilterSiteGuard._walk(APP_ROOT)
        if file != "repositories/common.py"
    ]
    assert not bare, f"bare application_id filter sites reappeared: {bare}"
