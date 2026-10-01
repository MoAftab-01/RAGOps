"""Onboarding: create a company, mint its keys, revoke them.

This is the surface that turns RAGOps from a single-tenant tool into one a
company can be given, and it is deliberately the *only* platform-privileged
write surface in the API. Everything else in ``/api`` either reads telemetry or
appends to it, and all of it is scoped to whichever organization the presented
key belongs to. Here, the caller is creating and managing those scopes, so
every route requires a platform principal.

**The rule that shapes the whole router: a plaintext key is returned exactly
once.** ``POST .../api-keys`` is the only response in the API that contains a
usable credential, and it is the only response that ever will. The value is
built from :func:`~app.core.security.generate_api_key`'s return tuple and passed
to the schema; it is never read back off the row, because a read-back would make
"served exactly once" a property of the code path rather than of the schema, and
the read-back is exactly the edit that would eventually serve it twice.

Two further decisions worth stating:

**Revoking sets ``revoked_at``; it does not delete the row.** A leaked key is
rarely a hypothetical, and ``last_used_at`` on a revoked key is the evidence
that answers "was it used?". DELETE-verb-that-sets-a-column is a real mismatch
with HTTP semantics, and it is taken knowingly: the alternative returns 204 and
destroys the audit trail, which is the thing an operator needs most.

**An organization a tenant key cannot reach is a 404, not a 403.** Listing a
known org id while being refused would let a tenant key enumerate the customer
list by walking the id space. Creating an organization is a 403 -- the answer
reveals nothing that was not already known, since the caller is asking a
question about a resource that would not exist.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DbSession, Principal, TenantPrincipal, WriteGuard, not_found
from app.core.logging import get_logger
from app.core.security import generate_api_key
from app.models import Application, Organization
from app.models.tenancy import ApiKey
from app.schemas.common import MessageResponse
from app.schemas.tenancy import (
    ApiKeyCreate,
    ApiKeyOut,
    CreatedApiKey,
    OrganizationCreate,
    OrganizationOut,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/organizations", tags=["organizations"])


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_platform(principal: Principal) -> None:
    """403 unless the caller resolved to the platform principal.

    Reached through the tenant resolver, so a *tenant* key here is refused while
    an absent key is allowed: an operator working locally against a fresh
    database should be able to create the first organization without first
    configuring a shared key.
    """
    if principal.is_platform:
        return
    logger.warning(
        "organizations.denied",
        reason="not_platform",
        organization_id=str(principal.organization_id) if principal.organization_id else None,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Only a platform key can manage organizations. Send the shared "
            "RAGOPS_API_KEY, or omit the header on a local deployment."
        ),
    )


async def _organization_or_404(
    session: AsyncSession, organization_id: uuid.UUID
) -> Organization:
    """Load one organization, or raise the 404 the rest of the API uses."""
    row = (
        await session.execute(
            select(Organization).where(Organization.id == organization_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise await not_found(f"No organization with id {organization_id}.")
    return row


async def _organization_counts(
    session: AsyncSession, organization_id: uuid.UUID
) -> tuple[int, int]:
    """``(num_applications, num_api_keys)`` for one organization.

    Two counts in one round trip per entity rather than a correlated subquery in
    the list statement. That is the wrong trade at 10,000 organizations and the
    right one here, where the list is bounded by MAX_PAGE_SIZE and the queries
    ride indexes that already exist for the tenancy predicates.
    """
    applications = (
        await session.execute(
            select(func.count())
            .select_from(Application)
            .where(Application.organization_id == organization_id)
        )
    ).scalar_one()
    keys = (
        await session.execute(
            select(func.count()).select_from(ApiKey).where(ApiKey.organization_id == organization_id)
        )
    ).scalar_one()
    return int(applications), int(keys)


async def to_out(session: AsyncSession, organization: Organization) -> OrganizationOut:
    """Build the response schema, filling the two aggregate counts."""
    num_applications, num_api_keys = await _organization_counts(session, organization.id)
    return OrganizationOut(
        id=organization.id,
        name=organization.name,
        slug=organization.slug,
        is_active=organization.is_active,
        num_applications=num_applications,
        num_api_keys=num_api_keys,
        created_at=organization.created_at,
        updated_at=organization.updated_at,
    )


def _key_out(key: ApiKey) -> ApiKeyOut:
    """Build the read schema for one key.

    ``is_active`` is computed here rather than stored, so a key that expired an
    hour ago reports as inactive without anything having to run. ``scopes`` is
    split from the stored comma-joined string, and a malformed stored value
    degrades to an empty list rather than raising -- a row this code could not
    render is not a reason to make the whole key list 500.
    """
    now = datetime.now(timezone.utc)
    return ApiKeyOut(
        id=key.id,
        organization_id=key.organization_id,
        name=key.name,
        key_prefix=key.key_prefix,
        scopes=[scope for scope in (key.scopes or "").split(",") if scope],
        is_active=key.revoked_at is None and (key.expires_at is None or key.expires_at > now),
        last_used_at=key.last_used_at,
        revoked_at=key.revoked_at,
        expires_at=key.expires_at,
        created_at=key.created_at,
        updated_at=key.updated_at,
    )


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


@router.get("", response_model=list[OrganizationOut], summary="List organizations")
async def list_organizations(session: DbSession, principal: TenantPrincipal) -> list[OrganizationOut]:
    """Every organization, active ones first, then by name.

    A plain list rather than a :class:`~app.schemas.common.Page`: the set of
    companies is bounded by how many an operator has onboarded -- tens, not
    millions -- and the Settings panel needs all of them to populate a picker.
    ``is_active=false`` rows are included rather than hidden, because the useful
    question is not "which companies exist" but "which of these can I send
    telemetry to", and deactivating must not make an organization unfindable.
    """
    _require_platform(principal)
    stmt: Select[tuple[Organization]] = select(Organization).order_by(
        Organization.is_active.desc(), Organization.name.asc()
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [await to_out(session, row) for row in rows]


@router.post(
    "",
    response_model=OrganizationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create an organization",
    dependencies=[WriteGuard],
)
async def create_organization(
    session: DbSession, payload: OrganizationCreate, principal: TenantPrincipal
) -> OrganizationOut:
    """Create a company, ready to receive telemetry.

    ``name`` is unique, so a repeat POST is a 409 rather than a second row with
    the same name -- two organizations a user cannot tell apart are worse than a
    request that has to be renamed. The check and the insert are not atomic; the
    unique index is what actually guarantees it, and the IntegrityError it raises
    is caught below and reported as the same 409. Without that catch a
    simultaneous double-submit would surface as a 500.

    No key is minted here. Creating a company and creating a credential are
    separate acts, so a key is issued only when somebody asks for one and the
    plaintext is delivered on that response alone.
    """
    _require_platform(principal)

    existing = (
        await session.execute(select(Organization.id).where(Organization.name == payload.name))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An organization named {payload.name!r} already exists.",
        )

    organization = Organization(
        name=payload.name,
        slug=payload.slug or _slugify(payload.name),
        is_active=payload.is_active,
    )
    session.add(organization)
    try:
        # Flushed rather than committed: the caller owns the transaction, and
        # ``to_out`` needs the generated id and timestamps.
        await session.flush()
    except Exception as exc:  # noqa: BLE001 - re-raised as a 409 below
        if not _is_unique_violation(exc):
            raise
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An organization named {payload.name!r} already exists.",
        ) from exc

    logger.info(
        "organizations.created",
        organization_id=str(organization.id),
        name=organization.name,
    )
    return await to_out(session, organization)


def _slugify(name: str) -> str:
    """A URL-friendly handle derived from the display name.

    Best effort and never required to succeed: the slug exists for humans to
    read and nothing resolves against it, so an input that slugifies to nothing
    yields ``None`` rather than an error. Truncated to the column width.
    """
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug[:128] or None


def _is_unique_violation(exc: Exception) -> bool:
    """Whether ``exc`` is a PostgreSQL unique-constraint violation.

    Checked on the exception's own ``sqlstate`` rather than by importing the
    ``asyncpg`` exception class, so this stays correct whichever driver the
    configured URL selects and does not depend on the driver being installed at
    import time.
    """
    return getattr(getattr(exc, "orig", exc), "sqlstate", None) == "23505"


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


@router.get(
    "/{organization_id}/api-keys",
    response_model=list[ApiKeyOut],
    summary="List an organization's keys",
)
async def list_api_keys(
    session: DbSession, organization_id: uuid.UUID, principal: TenantPrincipal
) -> list[ApiKeyOut]:
    """Every key belonging to one organization, newest first.

    Revoked keys are included. They are inert -- ``resolve_api_key`` refuses a
    row with ``revoked_at`` set -- so listing them is safe, and hiding them
    would remove the only evidence that a leaked key was dealt with.

    Revoked keys sort last rather than being filtered out: an operator scanning
    this list is usually looking for the one key they are about to revoke, and a
    disappearing row makes them re-click to find out what they just did.
    """
    _require_platform(principal)
    await _organization_or_404(session, organization_id)
    stmt = (
        select(ApiKey)
        .where(ApiKey.organization_id == organization_id)
        .order_by(ApiKey.revoked_at.is_not(None), ApiKey.created_at.desc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [_key_out(row) for row in rows]


@router.post(
    "/{organization_id}/api-keys",
    response_model=CreatedApiKey,
    status_code=status.HTTP_201_CREATED,
    summary="Mint an API key",
    dependencies=[WriteGuard],
)
async def create_api_key(
    session: DbSession,
    organization_id: uuid.UUID,
    payload: ApiKeyCreate,
    principal: TenantPrincipal,
) -> CreatedApiKey:
    """Mint a key and return its plaintext **once**.

    This response is the only place a usable credential exists in this system.
    ``generate_api_key()`` returns ``(plaintext, digest)``; the digest is what
    gets stored, and the plaintext is carried straight into the response object.
    It is deliberately not read back from the row: the row cannot produce it, so
    this construction is what makes "shown once" a property of the design rather
    than of the current code path.

    ``key_prefix`` is the first 12 characters, kept so a list of keys is
    recognisable and a revoke button can say which key it is revoking. It is
    display material, never a credential: 12 characters of a 43-character,
    256-bit key is not a key.
    """
    _require_platform(principal)
    await _organization_or_404(session, organization_id)

    plaintext, digest = generate_api_key()

    # ``last_used_at`` is stamped on first use rather than at creation: a key
    # that was minted and never sent is what an unused-credentials sweep is
    # looking for, and setting it now would make that key look exercised.
    key = ApiKey(
        organization_id=organization_id,
        name=payload.name,
        key_prefix=plaintext[:12],
        key_hash=digest,
        # The column is a comma-joined string; the schema canonicalises the
        # order, so this join is a rendering and not a decision. It must never be
        # ``str(payload.scopes)`` -- that puts Python's list repr, brackets and
        # all, into a column the resolver splits on commas.
        scopes=",".join(payload.scopes or ()),
        expires_at=payload.expires_at,
    )
    session.add(key)
    await session.flush()

    logger.info(
        "api_key.created",
        key_id=str(key.id),
        organization_id=str(organization_id),
        name=key.name,
        scopes=key.scopes,
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
    )
    # Never log ``plaintext`` -- not here, not on an error path. The prefix is
    # already in the row, and a log line is not a place a credential belongs.
    return CreatedApiKey(**_key_out(key).model_dump(), key=plaintext)


@router.delete(
    "/{organization_id}/api-keys/{key_id}",
    response_model=MessageResponse,
    summary="Revoke an API key",
    dependencies=[WriteGuard],
)
async def revoke_api_key(
    session: DbSession,
    organization_id: uuid.UUID,
    key_id: uuid.UUID,
    principal: TenantPrincipal,
) -> MessageResponse:
    """Revoke a key by stamping ``revoked_at``.

    DELETE sets a column and returns 200 rather than 204, and the mismatch is
    deliberate: a real delete would take ``created_at``, ``last_used_at`` and the
    name with it, and those are what answer "was this key used after it leaked?".
    The row stays, ``resolve_api_key`` refuses it, and it stays visible in the
    list.

    Idempotent: revoking an already-revoked key succeeds and keeps the original
    ``revoked_at``, so the timestamp stays a fact about the first revocation
    rather than being overwritten by whatever client happened to retry.

    The key must belong to the organization in the path. A key id from another
    organization is a 404, not a 403, for the reason every other cross-tenant
    lookup here is a 404: a 403 confirms the id exists.
    """
    _require_platform(principal)
    await _organization_or_404(session, organization_id)

    key = (
        await session.execute(
            select(ApiKey).where(
                ApiKey.id == key_id, ApiKey.organization_id == organization_id
            )
        )
    ).scalar_one_or_none()
    if key is None:
        raise await not_found(f"No API key {key_id} on organization {organization_id}.")

    if key.revoked_at is not None:
        logger.info("api_key.revoke_noop", key_id=str(key_id), reason="already_revoked")
        return MessageResponse(
            message=f"API key {key.name!r} was already revoked.",
            detail={"key_id": str(key_id), "revoked_at": key.revoked_at.isoformat()},
        )

    key.revoked_at = datetime.now(timezone.utc)
    await session.flush()

    logger.info("api_key.revoked", key_id=str(key_id), organization_id=str(organization_id))
    return MessageResponse(
        message=f"API key {key.name!r} revoked.",
        detail={
            "key_id": str(key_id),
            "revoked_at": key.revoked_at.isoformat(),
        },
    )


__all__ = ["router"]