"""Endpoint tests for the organizations router -- creating companies and their keys.

Split from :mod:`tests.test_tenancy` rather than appended to it, and the split is
about what these tests are defending. ``test_tenancy`` asks "does one company see
another's data?". This file asks "can an untrusted caller mint itself a
company?" -- the same surface, the opposite direction. Keeping them apart means
a failure names the question that broke.

The rules from ``test_tenancy`` carry over unchanged: no key is ever printed,
logged, or written down as a literal. Every plaintext here comes out of a
response, is used immediately, and is dropped when the test ends. The keys that
authenticate a request are generated in-process or minted over HTTP, never
typed into a file where a future reader could mistake one for a real credential.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import pytest

from tests.conftest import TEST_SHARED_KEY
from tests.test_tenancy import API_KEY_PATTERN

#: ``shared_key_enabled`` is requested by *name* rather than imported. An
#: autouse fixture applies only to the module that defines it, so a second file
#: driving tenant endpoints must ask for it explicitly -- otherwise every
#: request here resolves through the empty-shared-key escape hatch to the
#: platform principal, and the privilege assertions below would pass against a
#: platform caller while claiming to have tested a tenant one.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("shared_key_enabled")]

#: The shared key as a header, spelled out once so no test here can omit it by
#: accident and read a platform answer as if it had come from a tenant key.
PLATFORM_HEADERS = {"X-API-Key": TEST_SHARED_KEY}


def _created(response: Any) -> dict[str, Any]:
    """Assert a 201 from a creation route and return its payload."""
    assert response.status_code == 201, response.text
    return response.json()


async def _mint(client: Any, organization_id: Any, **payload: Any) -> dict[str, Any]:
    """POST one key over HTTP and return the response body."""
    body = {"name": "key", **payload}
    return _created(
        await client.post(
            f"/api/organizations/{organization_id}/api-keys",
            json=body,
            headers=PLATFORM_HEADERS,
        )
    )


# ---------------------------------------------------------------------------
# The one response that carries a credential
# ---------------------------------------------------------------------------


async def test_minting_a_key_returns_the_plaintext_exactly_once(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """The creation response carries the key; no other shape even has a field.

    The load-bearing assertion is the second half. ``ApiKeyOut`` has no
    ``key`` attribute, so re-serving the plaintext is not a code path that
    exists -- the schema, rather than this route's care, is what prevents it.
    """
    from sqlalchemy import select

    from app.models import ApiKey

    # ``with_key=False`` so the list below holds exactly the key this test
    # minted. The factory's own key is real and would be a second row, making
    # "the list has no plaintext field" an assertion about whichever row happened
    # to sort first.
    organization, _ = await org_factory("Key Mint", with_key=False)
    body = _created(
        await api_client.post(
            f"/api/organizations/{organization.id}/api-keys",
            json={"name": "CI"},
            headers=PLATFORM_HEADERS,
        )
    )
    plaintext = body["key"]

    assert re.match(API_KEY_PATTERN, plaintext), "the minted key does not match its scheme"
    assert body["key_prefix"] == plaintext[:12], "the prefix is not the head of the key"
    assert body["is_active"] is True
    assert body["scopes"] == ["ingest", "read"], "the default grant is not both scopes"
    assert body["last_used_at"] is None, "a key is born already used"

    listed = await api_client.get(
        f"/api/organizations/{organization.id}/api-keys", headers=PLATFORM_HEADERS
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()
    assert len(items) == 1
    assert "key" not in items[0], "the list response exposes a plaintext field"
    assert set(items[0]) == {
        "id",
        "organization_id",
        "name",
        "key_prefix",
        "scopes",
        "is_active",
        "last_used_at",
        "revoked_at",
        "expires_at",
        "created_at",
        "updated_at",
    }

    rows = (await db_session.execute(select(ApiKey))).scalars().all()
    assert rows, "the key row was not persisted at all"
    for row in rows:
        assert row.key_hash != plaintext, "the plaintext was stored verbatim"
        assert plaintext not in row.key_hash, "the plaintext survives in the digest column"


async def test_a_key_minted_over_http_actually_authenticates(
    api_client: Any, db_session: Any, org_factory: Any
) -> None:
    """The end-to-end claim: a mint that stores an unusable digest fails here.

    Every other test in this file either checks the response shape or compares
    digests, and both would pass against a key that could never be presented
    successfully. Only sending telemetry with the returned plaintext proves the
    stored hash is the hash of the value that was handed out.
    """
    from sqlalchemy import select

    from app.models import Application

    organization, _ = await org_factory("Round Trip")
    body = await _mint(api_client, organization.id, name="minted")
    application = f"pytest-minted-{uuid.uuid4().hex[:8]}"

    written = await api_client.post(
        "/api/traces",
        json={"application": application, "kind": "chat", "status": "running"},
        headers={"X-API-Key": body["key"]},
    )
    assert written.status_code == 201, written.text

    row = (
        await db_session.execute(select(Application).where(Application.name == application))
    ).scalar_one()
    assert row.organization_id == organization.id, (
        "the trace landed under an organization the key does not belong to"
    )


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scopes", "expected"),
    [
        (None, ["ingest", "read"]),
        (["read"], ["read"]),
        (["read", "ingest"], ["ingest", "read"]),
        (["ingest", "read"], ["ingest", "read"]),
    ],
)
async def test_key_scopes_are_normalised_not_passed_through(
    api_client: Any, org_factory: Any, scopes: Any, expected: list[str]
) -> None:
    """``read,ingest`` and ``ingest,read`` are the same grant, stored once.

    Persisting them as different strings would make two rows that mean the same
    thing compare unequal in an audit, and would let a client believe it had
    been given something it had not.
    """
    organization, _ = await org_factory(f"Scope{uuid.uuid4().hex[:4]}")
    payload = {"name": "scoped"}
    if scopes is not None:
        payload["scopes"] = scopes
    body = _created(
        await api_client.post(
            f"/api/organizations/{organization.id}/api-keys",
            json=payload,
            headers=PLATFORM_HEADERS,
        )
    )
    assert body["scopes"] == expected


async def test_an_unknown_scope_is_refused_rather_than_stored(
    api_client: Any, org_factory: Any
) -> None:
    """A typo is a 422, not a key that quietly cannot do anything.

    Storing ``wirte`` would produce a credential that authenticates perfectly
    and writes nothing, and the only symptom would be telemetry that never
    arrives.
    """
    organization, _ = await org_factory("Typo Scope")
    response = await api_client.post(
        f"/api/organizations/{organization.id}/api-keys",
        json={"name": "typo", "scopes": ["ingest", "wirte"]},
        headers=PLATFORM_HEADERS,
    )
    assert response.status_code == 422, response.text
    assert "wirte" in response.text


@pytest.mark.parametrize("scopes", [[], [""], ["  "]])
async def test_an_empty_scope_list_is_refused(
    api_client: Any, org_factory: Any, scopes: list[str]
) -> None:
    """A key with no scopes is a credential that looks valid and is not."""
    organization, _ = await org_factory(f"Empty{uuid.uuid4().hex[:4]}")
    response = await api_client.post(
        f"/api/organizations/{organization.id}/api-keys",
        json={"name": "empty", "scopes": scopes},
        headers=PLATFORM_HEADERS,
    )
    assert response.status_code == 422, response.text


async def test_a_read_only_key_cannot_ingest(api_client: Any, org_factory: Any) -> None:
    """Scopes are enforced on use, not merely recorded.

    A scope column nothing reads is decoration. This is the assertion that makes
    the column load-bearing.
    """
    organization, _ = await org_factory("Reader")
    body = await _mint(api_client, organization.id, name="reader", scopes=["read"])
    assert body["scopes"] == ["read"]

    response = await api_client.post(
        "/api/traces",
        json={
            "application": f"pytest-reader-{uuid.uuid4().hex[:8]}",
            "kind": "chat",
            "status": "running",
        },
        headers={"X-API-Key": body["key"]},
    )
    assert response.status_code == 403, response.text
    assert "ingest" in response.text


# ---------------------------------------------------------------------------
# Who may manage organizations
# ---------------------------------------------------------------------------


async def test_a_tenant_key_cannot_manage_organizations(
    api_client: Any, org_factory: Any
) -> None:
    """The decisive privilege check: a tenant key is 403 on this surface.

    If this ever returns 201, a customer key can mint itself a company and a
    fresh credential for it, which undoes the tenancy model from one route.
    """
    _, tenant_key = await org_factory("Would-Be Boss")

    created = await api_client.post(
        "/api/organizations",
        json={"name": f"self-minted-{uuid.uuid4().hex[:8]}"},
        headers={"X-API-Key": tenant_key},
    )
    assert created.status_code == 403, created.text

    listed = await api_client.get("/api/organizations", headers={"X-API-Key": tenant_key})
    assert listed.status_code == 403, listed.text


async def test_a_tenant_key_cannot_mint_a_second_key_for_its_own_organization(
    api_client: Any, org_factory: Any
) -> None:
    """Self-service minting would defeat revocation outright.

    The operator revokes a leaked key; the tenant issues another one an hour
    later and nothing looks wrong. That is why this is its own test rather than
    a consequence of the blanket 403 above -- it is the reason that 403 exists.
    """
    organization, tenant_key = await org_factory("Self Service")
    response = await api_client.post(
        f"/api/organizations/{organization.id}/api-keys",
        json={"name": "self-issued"},
        headers={"X-API-Key": tenant_key},
    )
    assert response.status_code == 403, response.text


async def test_the_platform_key_can_create_and_list_an_organization(
    api_client: Any, db_session: Any
) -> None:
    """The other half of the privilege check, including the derived counts."""
    from sqlalchemy import delete

    from app.models import Organization

    name = f"pytest-org-{uuid.uuid4().hex[:10]}"
    created = await api_client.post(
        "/api/organizations", json={"name": name}, headers=PLATFORM_HEADERS
    )
    assert created.status_code == 201, created.text
    body = created.json()
    organization_id = body["id"]

    try:
        assert body["name"] == name
        assert body["num_applications"] == 0, "a fresh org has no applications"
        assert body["num_api_keys"] == 0, "a fresh org has no keys"
        assert body["is_active"] is True

        listed = await api_client.get("/api/organizations", headers=PLATFORM_HEADERS)
        assert listed.status_code == 200, listed.text
        assert any(item["id"] == organization_id for item in listed.json()), (
            "the new organization is not in its own list"
        )

        # The counts are aggregates, so they have to be recomputed -- asserted
        # after a change, or a stale default of 0 would pass.
        await _mint(api_client, organization_id, name="first")
        again = await api_client.get("/api/organizations", headers=PLATFORM_HEADERS)
        row = next(item for item in again.json() if item["id"] == organization_id)
        assert row["num_api_keys"] == 1, "the derived key count did not update"
    finally:
        await db_session.execute(
            delete(Organization).where(Organization.id == organization_id)
        )
        await db_session.commit()


async def test_an_explicit_slug_is_kept(api_client: Any, db_session: Any) -> None:
    """A caller-supplied slug wins over the derived one."""
    from sqlalchemy import delete

    from app.models import Organization

    name = f"pytest-slug-{uuid.uuid4().hex[:8]}"
    created = await api_client.post(
        "/api/organizations",
        json={"name": name, "slug": "acme-support"},
        headers=PLATFORM_HEADERS,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    try:
        assert body["slug"] == "acme-support", "an explicit slug was overridden"
    finally:
        await db_session.execute(
            delete(Organization).where(Organization.id == body["id"])
        )
        await db_session.commit()


async def test_a_duplicate_organization_name_is_a_conflict(
    api_client: Any, db_session: Any
) -> None:
    """Two organizations a user cannot tell apart are worse than a 409."""
    from sqlalchemy import delete

    from app.models import Organization

    name = f"pytest-dup-{uuid.uuid4().hex[:10]}"
    created_ids: list[Any] = []
    try:
        first = await api_client.post(
            "/api/organizations", json={"name": name}, headers=PLATFORM_HEADERS
        )
        assert first.status_code == 201, first.text
        created_ids.append(first.json()["id"])

        second = await api_client.post(
            "/api/organizations", json={"name": name}, headers=PLATFORM_HEADERS
        )
        assert second.status_code == 409, second.text
        assert name in second.text
    finally:
        for organization_id in created_ids:
            await db_session.execute(
                delete(Organization).where(Organization.id == organization_id)
            )
        await db_session.commit()


async def test_an_unknown_organization_id_is_a_404(api_client: Any) -> None:
    """Neither key route invents an organization that does not exist."""
    missing = uuid.uuid4()
    assert (
        await api_client.get(
            f"/api/organizations/{missing}/api-keys", headers=PLATFORM_HEADERS
        )
    ).status_code == 404

    created = await api_client.post(
        f"/api/organizations/{missing}/api-keys",
        json={"name": "orphan"},
        headers=PLATFORM_HEADERS,
    )
    assert created.status_code == 404, created.text


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


async def test_revoking_a_key_keeps_the_row_and_preserves_the_timestamp(
    api_client: Any, org_factory: Any
) -> None:
    """Revocation stamps a column; the row is the evidence it happened.

    A real DELETE would take ``created_at``, ``last_used_at`` and the name with
    it, and those are what answer "was this key used after it leaked?".
    """
    organization, _ = await org_factory("Revocation")
    body = await _mint(api_client, organization.id, name="leaked")

    revoked = await api_client.delete(
        f"/api/organizations/{organization.id}/api-keys/{body['id']}",
        headers=PLATFORM_HEADERS,
    )
    assert revoked.status_code == 200, revoked.text

    listed = await api_client.get(
        f"/api/organizations/{organization.id}/api-keys", headers=PLATFORM_HEADERS
    )
    row = {item["id"]: item for item in listed.json()}[body["id"]]
    assert row["is_active"] is False, "a revoked key still reads as active"
    assert row["revoked_at"] is not None

    # Idempotent, and the *first* timestamp survives the retry -- otherwise
    # "when was this revoked" becomes a moving target.
    again = await api_client.delete(
        f"/api/organizations/{organization.id}/api-keys/{body['id']}",
        headers=PLATFORM_HEADERS,
    )
    assert again.status_code == 200, again.text
    relisted = await api_client.get(
        f"/api/organizations/{organization.id}/api-keys", headers=PLATFORM_HEADERS
    )
    after = {item["id"]: item for item in relisted.json()}[body["id"]]
    assert after["revoked_at"] == row["revoked_at"], (
        "the retry overwrote the moment of revocation"
    )


async def test_a_revoked_key_stops_authenticating(
    api_client: Any, org_factory: Any
) -> None:
    """Revocation is a real refusal, not a label on a list.

    ``test_revoking_a_key_keeps_the_row_and_preserves_the_timestamp`` shows the
    row says ``is_active: false``. This asserts the interesting half: presenting
    the key stops working, which is the only thing that makes the flag mean
    anything.
    """
    organization, _ = await org_factory("Dead Key")
    body = await _mint(api_client, organization.id, name="short-lived")
    headers = {"X-API-Key": body["key"]}

    before = await api_client.post(
        "/api/traces",
        json={
            "application": f"pytest-live-{uuid.uuid4().hex[:8]}",
            "kind": "chat",
            "status": "running",
        },
        headers=headers,
    )
    assert before.status_code == 201, before.text

    await api_client.delete(
        f"/api/organizations/{organization.id}/api-keys/{body['id']}",
        headers=PLATFORM_HEADERS,
    )

    after = await api_client.post(
        "/api/traces",
        json={
            "application": f"pytest-dead-{uuid.uuid4().hex[:8]}",
            "kind": "chat",
            "status": "running",
        },
        headers=headers,
    )
    assert after.status_code == 401, after.text


async def test_a_key_belonging_to_another_organization_is_a_404(
    api_client: Any, org_factory: Any
) -> None:
    """A 403 would confirm the key id exists, and that is the enumeration.

    The same rule as every other cross-tenant lookup in this codebase: an answer
    that reveals nothing is the only safe one.
    """
    mine, _ = await org_factory("Mine")
    theirs, _ = await org_factory("Theirs")

    body = await _mint(api_client, theirs.id, name="not-yours")
    response = await api_client.delete(
        f"/api/organizations/{mine.id}/api-keys/{body['id']}",
        headers=PLATFORM_HEADERS,
    )
    assert response.status_code == 404, response.text
    assert "no api key" in response.text.lower()


# ---------------------------------------------------------------------------
# The escape hatch, restated for this surface
# ---------------------------------------------------------------------------


async def test_the_empty_shared_key_still_reaches_the_organizations_surface(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no shared key configured, an absent header is a platform caller.

    ``test_tenancy`` pins the same behaviour for reads. It is restated here
    because this surface is where it actually matters operationally: on a fresh
    local checkout there is no key at all, and creating the first organization
    has to work without one or local setup is impossible.
    """
    from app.core import security

    monkeypatch.setattr(security.settings, "ragops_api_key", "")

    listed = await api_client.get("/api/organizations")
    assert listed.status_code == 200, listed.text
    assert isinstance(listed.json(), list)


def test_platform_headers_use_a_non_credential_constant() -> None:
    """The header constant is not a key that would work anywhere.

    A test that authenticated with a real-looking credential is one CI log away
    from leaking it, so the shared key these tests present is asserted to be the
    obviously-fake literal it is.
    """
    assert TEST_SHARED_KEY == "test-shared-key"
    assert API_KEY_PATTERN.match(TEST_SHARED_KEY) is None, (
        "the shared key is shaped like a real credential"
    )
    assert PLATFORM_HEADERS["X-API-Key"] == TEST_SHARED_KEY