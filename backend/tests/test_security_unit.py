"""Unit tests for key material, without a database or a server.

Every assertion here is checked against an *independently computed* expected
value -- a hand-rolled HMAC in the test, not a call back into the module under
test. Comparing `hash_api_key(...)` to `hash_api_key(...)` would pass even if
the function returned a constant.

No generated key is ever printed or asserted on by value: these tests exist to
prove the *properties* (scheme, length, uniqueness, digest correctness), and a
key that reaches a test log is a key in a transcript.
"""

from __future__ import annotations

import hashlib
import hmac
import re

import pytest

from app.core.security import (
    KEY_PREFIX_SCHEME,
    Principal,
    generate_api_key,
    hash_api_key,
    resolve_api_key,
    verify_api_key,
)
from app.models.tenancy import KEY_PREFIX_LENGTH

# `secrets.token_urlsafe(32)` -> ceil(32*4/3) = 43 url-safe base64 chars.
EXPECTED_SUFFIX_LENGTH = 43
KEY_PATTERN = re.compile(rf"^{KEY_PREFIX_SCHEME}_[A-Za-z0-9_-]{{{EXPECTED_SUFFIX_LENGTH}}}$")


@pytest.fixture(autouse=True)
def _pepper(monkeypatch):
    """A fixed pepper so the digest is reproducible within a run."""
    from app.config import settings

    monkeypatch.setattr(settings, "api_key_pepper", "unit-test-pepper", raising=False)


class TestKeyShape:
    def test_matches_documented_pattern(self) -> None:
        plaintext, _ = generate_api_key()
        assert KEY_PATTERN.match(plaintext), "key shape changed from the contract"

    def test_prefix_is_the_first_key_prefix_length_chars(self) -> None:
        """What the list view shows is a genuine prefix of the real key.

        Asserted on lengths and slicing rather than on the value, so no key
        material reaches the assertion output.
        """
        plaintext, _ = generate_api_key()
        assert len(plaintext) > KEY_PREFIX_LENGTH
        assert plaintext[:KEY_PREFIX_LENGTH].startswith(KEY_PREFIX_SCHEME)

    def test_keys_are_unique(self) -> None:
        generated = {generate_api_key()[0] for _ in range(1000)}
        assert len(generated) == 1000, "token_urlsafe repeated a key"

    def test_hash_is_not_the_plaintext(self) -> None:
        plaintext, digest = generate_api_key()
        assert digest != plaintext
        assert plaintext not in digest


class TestHashing:
    def test_matches_an_independent_hmac(self) -> None:
        """Ground truth computed here, not by calling the module under test."""
        plaintext, digest = generate_api_key()
        expected = hmac.new(
            b"unit-test-pepper", plaintext.encode(), hashlib.sha256
        ).hexdigest()
        assert digest == expected

    def test_is_64_lowercase_hex_chars(self) -> None:
        _, digest = generate_api_key()
        assert len(digest) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

    def test_pepper_actually_changes_the_digest(self) -> None:
        """If this fails, setting API_KEY_PEPPER does nothing."""
        from app.config import settings

        plaintext, _ = generate_api_key()
        before = hash_api_key(plaintext)
        settings.api_key_pepper = "a-different-pepper"
        after = hash_api_key(plaintext)
        assert before != after

    def test_empty_pepper_still_hashes(self) -> None:
        """A clean checkout must boot with no API_KEY_PEPPER set.

        The digest is HMAC with an all-zero key, which is *not* plain
        SHA-256: HMAC zero-pads its key to the block size before use. What
        matters for the empty-pepper case is that the result is still a stable
        64-char digest and still needs no secret to compute -- which is the
        documented cost of leaving the pepper unset.
        """
        from app.config import settings

        settings.api_key_pepper = ""
        plaintext, digest = generate_api_key()

        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest == hash_api_key(plaintext), "must be deterministic"
        # Not the plain SHA-256, contrary to what the docstrings used to claim.
        assert digest != hashlib.sha256(plaintext.encode()).hexdigest()
        # And it is computable by anyone holding the database, which is the
        # whole point of warning when the pepper is unset.
        assert digest == hmac.new(b"", plaintext.encode(), hashlib.sha256).hexdigest()


class TestVerifyApiKey:
    """The legacy single-row check, whose behaviour must not change."""

    def test_no_shared_key_configured_allows_everything(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "", raising=False)
        assert verify_api_key(None) is True
        assert verify_api_key("anything") is True

    def test_missing_header_is_rejected(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        assert verify_api_key(None) is False

    def test_wrong_key_is_rejected(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        assert verify_api_key("nope") is False

    def test_exact_key_is_accepted(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        assert verify_api_key("dev-key") is True


class TestPrincipal:
    def test_platform_principal_can_do_everything(self) -> None:
        p = Principal(is_platform=True, scopes=frozenset({"ingest", "read"}))
        assert p.can_read and p.can_ingest
        p.require_scope("ingest")  # must not raise

    def test_tenant_scopes_are_checked(self) -> None:
        p = Principal(is_platform=False, organization_id=None, scopes=frozenset({"read"}))
        assert p.can_read is True
        assert p.can_ingest is False
        with pytest.raises(Exception) as exc:
            p.require_scope("ingest")
        assert exc.value.status_code == 403

    def test_empty_scopes_denies_everything(self) -> None:
        p = Principal(is_platform=False, scopes=frozenset())
        assert p.can_read is False and p.can_ingest is False

    def test_is_platform_overrides_scopes(self) -> None:
        """The platform principal is not scope-limited even with no scopes."""
        p = Principal(is_platform=True, scopes=frozenset())
        assert p.can_read and p.can_ingest


class TestResolutionTable:
    """The decision table, tested with the database stubbed out.

    These are the semantics that make steps 5-7 safe, so they are pinned here
    rather than left to the endpoint tests in step 10.
    """

    @pytest.mark.asyncio
    async def test_no_key_presented_is_platform(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        p = await resolve_api_key(_NoSession(), None)
        assert p.is_platform and p.organization_id is None

    @pytest.mark.asyncio
    async def test_empty_shared_key_escapes_hatch(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "", raising=False)
        p = await resolve_api_key(_NoSession(), "literally-anything")
        assert p.is_platform, "an empty shared key must not 401"

    @pytest.mark.asyncio
    async def test_matching_shared_key_is_platform_with_no_key_id(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        p = await resolve_api_key(_NoSession(), "dev-key")
        assert p.is_platform
        assert p.key_id is None, "the legacy shared key is not a tenant key"

    @pytest.mark.asyncio
    async def test_unknown_key_is_rejected_not_widened(self, monkeypatch) -> None:
        """The load-bearing row: a bad key must never fall back to platform."""
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        p = await resolve_api_key(_EmptyResultSession(), "rag_not_a_real_key")
        assert p is None, "an unknown key must resolve to None (=> 401), not platform"

    @pytest.mark.asyncio
    async def test_live_api_key_resolves_to_its_organization(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        session = _StubResultSession(
            _Row(
                id="key-1",
                organization_id="org-1",
                scopes="ingest,read",
                revoked_at=None,
                expires_at=None,
            )
        )
        p = await resolve_api_key(session, "rag_some_real_key")
        assert p is not None
        assert p.is_platform is False
        assert p.organization_id == "org-1"
        assert p.scopes == {"ingest", "read"}

    @pytest.mark.asyncio
    async def test_revoked_key_is_rejected(self, monkeypatch) -> None:
        from datetime import datetime, timezone

        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        session = _StubResultSession(
            _Row(
                id="key-1",
                organization_id="org-1",
                scopes="ingest,read",
                revoked_at=datetime.now(timezone.utc),
                expires_at=None,
            )
        )
        assert await resolve_api_key(session, "rag_revoked") is None

    @pytest.mark.asyncio
    async def test_expired_key_is_rejected(self, monkeypatch) -> None:
        from datetime import datetime, timedelta, timezone

        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        session = _StubResultSession(
            _Row(
                id="key-1",
                organization_id="org-1",
                scopes="ingest,read",
                revoked_at=None,
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
        )
        assert await resolve_api_key(session, "rag_expired") is None

    @pytest.mark.asyncio
    async def test_key_expiring_in_the_future_is_accepted(self, monkeypatch) -> None:
        from datetime import datetime, timedelta, timezone

        from app.config import settings

        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)
        session = _StubResultSession(
            _Row(
                id="key-1",
                organization_id="org-1",
                scopes="read",
                revoked_at=None,
                expires_at=datetime.now(timezone.utc) + timedelta(days=1),
            )
        )
        p = await resolve_api_key(session, "rag_future")
        assert p is not None and p.organization_id == "org-1"


class TestWriteGuardHeaderRules:
    """The write guard's header cases.

    These are the behaviours the pre-tenancy guard had, and the first attempt
    at the tenancy rework broke one of them: `current_principal` resolves "no
    key presented" to a *platform* principal (correct for reads), and letting
    that reach the write path made an unauthenticated POST succeed. These
    tests exist so that cannot happen again silently.
    """

    @pytest.mark.asyncio
    async def test_missing_header_is_401_on_a_write(self, monkeypatch) -> None:
        from fastapi import HTTPException

        from app.config import settings
        from app.core.security import require_api_key

        monkeypatch.setattr(settings, "auth_enabled", True, raising=False)
        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)

        with pytest.raises(HTTPException) as exc:
            await require_api_key(
                session=_NoSession(),
                x_api_key=None,
                principal=Principal(
                    is_platform=True, scopes=frozenset({"ingest", "read"})
                ),
            )
        assert exc.value.status_code == 401, (
            "an unauthenticated write must be rejected even though the "
            "principal says platform"
        )

    @pytest.mark.asyncio
    async def test_wrong_header_is_rejected_on_a_write(self, monkeypatch) -> None:
        """A bad key must not reach a write.

        The exact status is not asserted here, because where the 401 comes
        from depends on the resolution order. In real routing
        :func:`current_principal` runs first and raises 401 for an unknown
        key; calling the guard directly with that outcome already expressed
        lands on the scope check instead. Both are rejections, and the
        property worth pinning is simply that the write does not succeed.
        """
        from fastapi import HTTPException

        from app.config import settings
        from app.core.security import require_api_key

        monkeypatch.setattr(settings, "auth_enabled", True, raising=False)
        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)

        with pytest.raises(HTTPException) as exc:
            await require_api_key(
                session=_NoSession(),
                x_api_key="wrong",
                principal=Principal(is_platform=False, scopes=frozenset()),
            )
        assert exc.value.status_code in (401, 403)
        assert exc.value.status_code != 200

    @pytest.mark.asyncio
    async def test_missing_header_allowed_when_no_key_configured(
        self, monkeypatch
    ) -> None:
        """The escape hatch: nothing configured means nothing to enforce."""
        from app.config import settings
        from app.core.security import require_api_key

        monkeypatch.setattr(settings, "auth_enabled", True, raising=False)
        monkeypatch.setattr(settings, "ragops_api_key", "", raising=False)

        principal = await require_api_key(
            session=_NoSession(), x_api_key=None, principal=Principal(is_platform=True)
        )
        assert principal.is_platform

    @pytest.mark.asyncio
    async def test_read_only_key_cannot_ingest(self, monkeypatch) -> None:
        """A valid key with the wrong scope is 403, not 401: it *is* a key."""
        from fastapi import HTTPException

        from app.config import settings
        from app.core.security import require_api_key

        monkeypatch.setattr(settings, "auth_enabled", True, raising=False)
        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)

        with pytest.raises(HTTPException) as exc:
            await require_api_key(
                session=_NoSession(),
                x_api_key="rag_read_only",
                principal=Principal(is_platform=False, scopes=frozenset({"read"})),
            )
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_auth_disabled_short_circuits_everything(self, monkeypatch) -> None:
        from app.config import settings
        from app.core.security import require_api_key

        monkeypatch.setattr(settings, "auth_enabled", False, raising=False)
        monkeypatch.setattr(settings, "ragops_api_key", "dev-key", raising=False)

        principal = await require_api_key(
            session=_NoSession(), x_api_key=None, principal=Principal(is_platform=True)
        )
        assert principal.is_platform


# ---------------------------------------------------------------------------
# Stand-ins for the session, so the resolution logic is testable without a
# database. `execute` is the only method `resolve_api_key` touches.
# ---------------------------------------------------------------------------
class _Row:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _NoSession:
    async def execute(self, *args, **kwargs):
        raise AssertionError("no DB query expected: a platform key short-circuits")


class _EmptyResultSession:
    async def execute(self, *args, **kwargs):
        return _Result(None)


class _StubResultSession:
    def __init__(self, row) -> None:
        self._row = row

    async def execute(self, *args, **kwargs):
        return _Result(self._row)


class _Result:
    def __init__(self, row) -> None:
        self._row = row

    def one_or_none(self):
        return self._row
