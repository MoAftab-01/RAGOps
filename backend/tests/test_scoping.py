"""Scope resolution and cache-key isolation, without a database.

These are the properties step 3 has to get right *before* any row carries an
organization, because every later step assumes them. Three of them were real
bugs rather than hypotheticals:

* seven analytics endpoints shared one Redis key, so whichever ran first
  poisoned the others with a schema-mismatched body (a 500);
* the ``organization_id`` component was missing, so two companies reading the
  same window would have shared one cached dashboard -- and because the
  *shape* of both payloads matches, no response-model validation would catch
  it;
* the AST guard that was meant to catch the filter sites missed every one of
  them, because it compared an AST node to a class with ``==``.

Each has a test here that fails against the code as it was.

No database and no Redis: these assert on key derivation and object plumbing,
both of which are pure functions.
"""

from __future__ import annotations

import ast
import pathlib
import uuid
from datetime import datetime, timezone

import pytest

from app.core.cache import make_key
from app.repositories.common import WindowFilter
from app.services.window import AnalyticsScope, cache_parts

# Two fixed organizations and a fixed window, so a failure is reproducible
# rather than dependent on whatever uuid4 happened to return.
ORG_A = uuid.UUID(int=1)
ORG_B = uuid.UUID(int=2)
START = datetime(2026, 8, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 15, tzinfo=timezone.utc)

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"


def _scope(
    organization_id: uuid.UUID | None = None, application_id: uuid.UUID | None = None
) -> AnalyticsScope:
    return AnalyticsScope(
        start=START,
        end=END,
        label="custom",
        application_id=application_id,
        organization_id=organization_id,
    )


def _key(scope: AnalyticsScope, endpoint: str) -> str:
    return make_key("analytics", **cache_parts(scope, endpoint))


class TestCacheKeyIsolation:
    """The leak this whole step exists to close."""

    def test_two_organizations_do_not_share_a_key(self) -> None:
        """The bug: identical window, no application filter, one key.

        Both companies' payloads validate against the same response model, so a
        shared key does not error -- it silently serves company A's dashboard to
        company B. Nothing downstream would catch it.
        """
        assert _key(_scope(ORG_A), "dashboard-overview") != _key(
            _scope(ORG_B), "dashboard-overview"
        )

    def test_platform_and_tenant_do_not_share_a_key(self) -> None:
        """``None`` must be a distinct component, not a colliding absence.

        ``str(None)`` would render as the literal ``"None"``, so a real
        organization is never at risk of colliding with it -- but the
        implementation uses ``""``, and this pins that the platform scope
        cannot be reached by any tenant either.
        """
        assert _key(_scope(None), "dashboard-overview") != _key(
            _scope(ORG_A), "dashboard-overview"
        )

    def test_endpoints_do_not_share_a_key(self) -> None:
        """All seven call sites used to pass identical parts into one namespace.

        Whichever endpoint ran first cached its payload; the next one validated
        that payload against its own response model and raised a 500.
        """
        endpoints = [
            "dashboard-overview",
            "token-analytics",
            "cost-analytics",
            "latency-breakdown",
            "token-efficiency",
            "cost-quality",
            "application-stats",
        ]
        keys = {_key(_scope(ORG_A), name) for name in endpoints}
        assert len(keys) == len(endpoints), "two endpoints still share a key"

    def test_endpoint_name_is_required(self) -> None:
        """A new route must not be able to forget the discriminator.

        This is the structural half of the fix: the old signature took
        ``*extra``, so omitting the endpoint was not an error.
        """
        with pytest.raises(TypeError):
            cache_parts(_scope(ORG_A))  # type: ignore[call-arg]

    def test_key_is_stable_for_the_same_scope(self) -> None:
        """A cache that misses every time would be a different bug.

        The dashboard polls on a timer, so the same window has to hash to the
        same key across requests.
        """
        assert _key(_scope(ORG_A), "dashboard-overview") == _key(
            _scope(ORG_A), "dashboard-overview"
        )

    def test_organization_id_reaches_the_key_parts(self) -> None:
        """Guards the ``if organization_id else ""`` branch directly.

        Without this, an edit that dropped the component from the dict would
        fail the two-organization test with a message about keys rather than
        about the missing field.
        """
        assert cache_parts(_scope(ORG_A), "x")["organization_id"] == str(ORG_A)
        assert cache_parts(_scope(None), "x")["organization_id"] == ""


class TestWindowFilterPlumbing:
    """The organization has to survive the trip from scope to query."""

    def test_window_filter_carries_the_organization(self) -> None:
        window = _scope(ORG_A).window_filter()
        assert window.organization_id == ORG_A

    def test_platform_scope_is_both_none(self) -> None:
        """Platform-wide is the *pair* being None, not just the application."""
        window = _scope(None).window_filter()
        assert window.application_id is None and window.organization_id is None

    def test_unscoped_application_stays_tenant_scoped(self) -> None:
        """The load-bearing case: no ``?application=`` on a tenant read.

        This is what "``application_id is None`` means all of *this company's*
        applications" has to mean in practice. If the organization were dropped
        here, every unscoped read would go platform-wide and still look right.
        """
        window = _scope(ORG_A).window_filter()
        assert window.application_id is None
        assert window.organization_id == ORG_A, "an unscoped read lost its tenant"

    def test_application_stats_replacement_keeps_the_organization(self) -> None:
        """``applications.py`` rebuilds its scope with ``dataclasses.replace``.

        That is how a per-application tenant read would escape to platform
        scope if ``organization_id`` were not a field: ``replace`` copies
        everything it is not given, so the bug would be invisible in review.
        """
        from dataclasses import replace

        scoped = replace(_scope(ORG_A), application_id=uuid.UUID(int=9))
        assert scoped.window_filter().organization_id == ORG_A

    def test_describe_includes_the_organization(self) -> None:
        """The one log line that fires for every analytics read.

        If a cross-tenant leak ever happens, this is where the answer to
        "which org was it looking at?" lives.
        """
        assert WindowFilter(
            start=START, end=END, organization_id=ORG_A
        ).describe()["organization_id"] == str(ORG_A)


class TestFilterSiteGuard:
    """The ``ast`` check that answers "did I get all the sites?".

    Its first version reported zero sites across the whole codebase, which
    looked like good news and was in fact a broken predicate: it compared an
    AST node to a *class* (``node.ctx == ast.Load``), and an instance never
    equals its class. The test would have passed on a codebase where every
    filter site was unscoped -- the exact bug it exists to catch.
    """

    def _sites(self) -> list[tuple[str, int]]:
        return self._walk(APP_ROOT)

    @staticmethod
    def _is_application_id_guard(test: ast.expr) -> bool:
        """``<something>.application_id is not None`` as an ``if`` test."""
        if not isinstance(test, ast.Compare):
            return False
        if not any(isinstance(op, ast.IsNot) for op in test.ops):
            return False
        left = test.left
        name = left.attr if isinstance(left, ast.Attribute) else getattr(left, "id", None)
        return name == "application_id" and isinstance(left.ctx, ast.Load)

    @classmethod
    def _walk(cls, root: pathlib.Path) -> list[tuple[str, int]]:
        """Every ``if <something>.application_id is not None:`` site.

        The ``if`` is load-bearing, and the first version of this predicate
        matched any ``Compare``. That over-matched by exactly one --
        ``applications.py:154`` is ``if scope.application_id is not None and
        scope.application_id != application_id``, a *conflict check* between the
        path id and the query parameter. It builds no SQL, so routing it
        through :func:`apply_tenant_filter` would be meaningless; leaving it
        bare would fail the guard. Matching the filter shape specifically is
        what lets the guard be strict without demanding the impossible.

        Paths are normalised to forward slashes: ``Path.relative_to`` keeps the
        platform separator, so a hard-coded ``"api/v1/dashboard.py"`` would not
        match on Windows -- and an assertion like that failing for a path-format
        reason is exactly the kind of test that gets "fixed" by being deleted.
        """
        found: list[tuple[str, int]] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            rel = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and cls._is_application_id_guard(node.test):
                    found.append((rel, node.lineno))
        return found

    def test_the_predicate_itself_finds_known_sites(self) -> None:
        """A guard that silently matches nothing is worse than no guard.

        Pinned against the one site that still exists by design --
        ``repositories/common.py``'s own predicate builder -- so a future
        refactor of this predicate cannot pass by matching zero.

        The pin used to name ``dashboard.py`` and ``traces.py`` and to require
        30 sites. Both were accurate before step 4 and became false the moment
        the work landed: converting those files is the *point*, so a test that
        kept them would have failed on success. A guard whose anchors are all
        "sites that still need converting" is a to-do list wearing a test's
        clothes; after the conversion there is exactly one legitimate site left
        and it is the helper.
        """
        sites = self._sites()
        files = {f for f, _ in sites}
        assert files == {"repositories/common.py"}, (
            f"expected only the helper's own predicate builder, found {sorted(files)}"
        )
        assert sites, (
            "if this is 0 the predicate regressed, not the codebase -- see the "
            "class docstring for the first version that reported success while "
            "checking nothing"
        )

    def test_the_conflict_check_is_not_a_filter_site(self) -> None:
        """``applications.py:154`` must stay out of the guard's way.

        It is ``if scope.application_id is not None and ...``: a guard on the
        path id disagreeing with ``?application=``, which builds no SQL. The
        guard's predicate is ``if``-shaped precisely so this site is exempt --
        an earlier ``Compare``-shaped version counted it, and demanded a
        conversion that would have been meaningless.

        Still worth pinning after step 4, because the exemption is what lets the
        guard above assert an exact file set rather than a subset.
        """
        sites = set(self._sites())
        assert ("api/v1/applications.py", 154) not in sites

    def test_every_site_goes_through_apply_tenant_filter(self) -> None:
        """No bare application_id test may remain outside the helper.

        This used to skip while the 30 sites were unconverted. It no longer
        can: a skip is invisible in a green run, and a guard that has quietly
        stopped guarding is how "did I get all 30?" gets answered "I think so".
        The skip branch is gone rather than left in place, because there is no
        longer a state in which skipping would be honest -- if a bare site
        appears, someone has introduced one, and that is a finding.
        """
        # `common.py` is excluded: `apply_tenant_filter` is where the
        # comparison legitimately lives, and the point of this test is that
        # every *other* file routes through it.
        bare = [
            (f, line)
            for f, line in self._walk(APP_ROOT)
            if f != "repositories/common.py"
        ]
        assert not bare, (
            f"{len(bare)} bare application_id filter site(s) outside "
            f"repositories/common.py: {bare[:5]} -- route them through "
            "apply_tenant_filter / window_predicates / tenant_predicates"
        )
