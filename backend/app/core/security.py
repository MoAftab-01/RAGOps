"""API-key authentication, tenant principals and CORS.

RAGOps is multi-tenant, so "is this key valid" is no longer a boolean. A request
resolves to a :class:`Principal` -- either *platform* (unscoped, sees
everything) or *tenant* (scoped to exactly one organization) -- and every
read route filters on the organization that comes back.

The resolution table, which is the whole design:

======================  =========================================  ==============
Presented               Matches                                    Result
======================  =========================================  ==============
(nothing)               --                                          platform
(anything)              ``ragops_api_key`` is empty                 platform
(anything)              ``ragops_api_key``                          platform, key_id=None
(anything)              a live :class:`ApiKey` row                  that key's org
(anything)              revoked, expired, or unknown                **401**
======================  =========================================  ==============

The last row is the load-bearing one: **a presented key is authoritative**.
Sending a wrong key is rejected rather than quietly degrading to a
platform-wide principal, while sending no key at all still works. That is what
lets organization scoping bind on reads without turning ``X-API-Key`` into a
required parameter on fourteen routes.

No argon2, bcrypt or passlib, on purpose. ``secrets.token_urlsafe(32)`` is
~256 bits of OS CSPRNG output, so there is nothing to brute-force; slow hashing
exists to make *low-entropy* secrets expensive to guess, which does not apply
here. No salt column either: a password is looked up by user id, but a key is
looked up **by its own hash** on a UNIQUE index, so the digest is the identity.

Honest limitation: ``WHERE key_hash = :presented`` is an indexed probe, not a
constant-time comparison. Wrapping the hash in argon2 to fix that would not
help and would cost a dependency. ``hmac.compare_digest`` below is still
load-bearing on the legacy ``ragops_api_key`` path, which is the single-row
comparison it exists for.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.tenancy import ApiKey

logger = get_logger("ragops.security")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

#: Separator for a generated key. The scheme is a literal prefix, not a parser
#: requirement -- the key is authenticated by its hash, so nothing downstream
#: depends on this shape. It exists so a leaked key is recognisable at a glance
#: in a log or a support ticket.
KEY_PREFIX_SCHEME = "rag"


# ----------------------------------------------------------------------
# Key material
# ----------------------------------------------------------------------
def hash_api_key(raw: str) -> str:
    """Digest a plaintext key for storage.

    HMAC-SHA256 keyed by ``settings.api_key_pepper``. An empty pepper makes the
    key an all-zero block -- so the digest is reproducible by anyone holding
    the database rather than only by someone holding the pepper. It is still a
    perfectly good *identifier*; what it stops being is a secret-dependent
    value. See the ``api_key_pepper`` note in config.
    """
    return hmac.new(
        settings.api_key_pepper.encode(), raw.encode(), hashlib.sha256
    ).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Mint a key. Returns ``(plaintext, hash)``.

    The plaintext exists only in this return value. It is shown to the operator
    once, at creation, and there is no code path that can read it back.
    """
    if not settings.api_key_pepper:
        logger.warning(
            "auth.pepper_missing",
            detail=(
                "API_KEY_PEPPER is unset: stored digests depend on no secret. "
                "Set it in production."
            ),
        )
    raw = f"{KEY_PREFIX_SCHEME}_{secrets.token_urlsafe(32)}"
    return raw, hash_api_key(raw)


def verify_api_key(provided: str | None) -> bool:
    """Constant-time comparison so the key cannot be recovered by timing.

    Unchanged from the pre-tenancy implementation, and kept separate from
    :func:`resolve_api_key` on purpose: this is the single-row comparison
    against ``settings.ragops_api_key`` that ``hmac.compare_digest`` is
    meaningful for.
    """
    if not settings.ragops_api_key:
        return True
    if provided is None:
        return False
    return hmac.compare_digest(provided, settings.ragops_api_key)


# ----------------------------------------------------------------------
# Principals
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making this request, and what they may see.

    ``is_platform`` is the single switch every scoping decision reads. A
    platform principal sees everything; a tenant principal is confined to
    ``organization_id``. The two legacy cases both produce a platform
    principal, which is what keeps the local demo and the shared ``dev-key``
    working exactly as they did before tenancy existed.
    """

    is_platform: bool = True
    organization_id: uuid.UUID | None = None
    organization_name: str | None = None
    key_id: uuid.UUID | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)

    @property
    def can_read(self) -> bool:
        return self.is_platform or "read" in self.scopes

    @property
    def can_ingest(self) -> bool:
        return self.is_platform or "ingest" in self.scopes

    def require_scope(self, scope: str) -> None:
        """Raise 403 unless this principal holds ``scope``."""
        if self.is_platform or scope in self.scopes:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This API key does not have the '{scope}' scope.",
        )


#: What an unauthenticated request resolves to. Reads stay open by design --
#: the dashboard has to work before anything is instrumented -- and this is the
#: object that encodes that decision.
PLATFORM_PRINCIPAL = Principal(is_platform=True, scopes=frozenset({"ingest", "read"}))


async def resolve_api_key(
    session: AsyncSession, provided: str | None
) -> Principal | None:
    """Resolve a presented key to a tenant principal, or ``None``.

    ``None`` means "this key is not usable" -- unknown, revoked, or expired --
    and the caller turns that into a 401. It is deliberately *not* the same as
    "no key presented", which yields a platform principal: a bad key must fail
    loudly rather than silently widen the request's visibility.
    """
    if provided is None:
        return PLATFORM_PRINCIPAL

    if not settings.ragops_api_key:
        # Escape hatch: an empty shared key means the deployment has opted out
        # of key checks entirely. Unchanged behaviour, on purpose.
        return PLATFORM_PRINCIPAL

    if hmac.compare_digest(provided, settings.ragops_api_key):
        return Principal(is_platform=True, scopes=frozenset({"ingest", "read"}))

    digest = hash_api_key(provided)
    row = (
        await session.execute(
            select(
                ApiKey.id,
                ApiKey.organization_id,
                ApiKey.scopes,
                ApiKey.revoked_at,
                ApiKey.expires_at,
            ).where(ApiKey.key_hash == digest)
        )
    ).one_or_none()
    if row is None:
        return None

    now = datetime.now(timezone.utc)
    if row.revoked_at is not None:
        return None
    if row.expires_at is not None and row.expires_at <= now:
        return None

    # `scopes` is a comma-joined string rather than a join table: there are two
    # of them, the set is never queried by membership, and a join table would
    # be a second table to migrate for no query it enables.
    return Principal(
        is_platform=False,
        organization_id=row.organization_id,
        key_id=row.id,
        scopes=frozenset(s for s in row.scopes.split(",") if s),
    )


async def current_principal(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """Resolve the calling principal for the request.

    Declares ``session`` via ``Depends`` so the ``ApiKey`` lookup shares the
    route's session: FastAPI caches a dependency per request, so this does not
    open a second connection.
    """
    if not settings.auth_enabled:
        # Footgun, deliberately not defended against: AUTH_ENABLED=false
        # disables tenancy as well as authentication, and a *valid tenant key*
        # is then ignored rather than honoured. It is a local-development
        # switch; see the startup warning in app/main.py.
        return PLATFORM_PRINCIPAL

    principal = await resolve_api_key(session, x_api_key)
    if principal is None:
        logger.warning("auth.rejected", reason="unknown_revoked_or_expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key.",
        )
    return principal


async def require_api_key(
    session: Annotated[AsyncSession, Depends(get_db)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    principal: Annotated[Principal, Depends(current_principal)] = None,
) -> Principal:
    """Dependency guarding telemetry *write* endpoints.

    Returns the principal rather than ``None`` so a write route can bind the
    organization onto the row it creates without resolving the key a second
    time.

    Three behaviours are load-bearing and preserved from the pre-tenancy
    version:

    * an empty ``ragops_api_key`` allows everything;
    * a **missing** header on a write is a 401, not a pass;
    * a wrong header is a 401.

    The second one is the subtle one. :func:`current_principal` treats "no key
    presented" as a platform principal, which is right for *reads* -- the
    dashboard has to work before anything is instrumented -- but wrong for
    writes, which have always required a key whenever one is configured. So
    the header's absence is checked here, explicitly, rather than inferred
    from the principal.
    """
    if not settings.auth_enabled:
        return PLATFORM_PRINCIPAL
    if not settings.ragops_api_key:
        # Escape hatch: no shared key configured, so there is nothing to check.
        return PLATFORM_PRINCIPAL
    if x_api_key is None:
        logger.warning("auth.rejected", path_guard="write", reason="missing_header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Send it as the X-API-Key header.",
        )
    if not principal.can_ingest:
        logger.warning(
            "auth.rejected",
            reason="missing_scope",
            required="ingest",
            scopes=sorted(principal.scopes),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This API key does not have the 'ingest' scope.",
        )
    return principal


AuthDependency = Annotated[None, Depends(require_api_key)]
TenantPrincipal = Annotated[Principal, Depends(current_principal)]
