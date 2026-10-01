"""Organization and API-key schemas.

Two shapes of key exist here and the difference between them is the most
important thing in this file.

:class:`ApiKeyOut` is what every *read* returns: the key's name, its display
prefix, its scopes, and its lifecycle timestamps. There is no field on it that
could carry a usable credential, and that is structural rather than a promise --
the model is built from the ORM row, and the row has no column holding the
plaintext, because only its HMAC digest is persisted.

:class:`CreatedApiKey` is what a *creation* returns: the same metadata plus the
plaintext, in a field that exists nowhere else. The route builds it from
:func:`~app.core.security.generate_api_key`'s return value rather than reading it
back off the row it just inserted, so no code path can be tempted to serve the
same secret twice. The type is named to make the one-shot nature legible at the
call site, because the alternative -- a ``key`` field on :class:`ApiKeyOut` --
would be a field that is populated exactly once and empty on every other
response, which is the shape that eventually leaks.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.tenancy import DEFAULT_KEY_SCOPES
from app.schemas.common import ORMModel

#: The scopes a key may be granted. Fixed rather than free text because they are
#: checked by ``Principal.require_scope``: a typo in this list would silently
#: produce a key that can do nothing, and a client cannot tell that from a key
#: that was revoked.
ALLOWED_SCOPES: frozenset[str] = frozenset({"ingest", "read"})

#: Scopes accepted on creation. Separate from ``ALLOWED_SCOPES`` because a key
#: that may only read, or only ingest, is a real configuration -- a CI job that
#: posts traces should not be able to read the dashboard -- and the default must
#: stay the two together so the common case needs no thought.
DEFAULT_SCOPES: frozenset[str] = ALLOWED_SCOPES


def _normalize_scopes(raw: str | list[str] | None) -> list[str]:
    """Canonicalise a scope list to a sorted, deduped, whitespace-clean list.

    This is the single place the canonical form is defined, and it returns a
    list rather than the comma-joined string the column stores so that what
    comes out satisfies ``scopes: list[str]``. A ``mode="before"`` validator's
    return value goes on to be validated against the field annotation, so a
    validator that returned ``"ingest,read"`` here would fail its own field.

    Sorting matters: without it two requests for ``read,ingest`` and
    ``ingest,read`` would store different strings for the same grant, and a
    row-by-row comparison of keys would show them as different.

    A comma-joined *string* is accepted on input, because the stored form is one
    and a client reading a key back out of the database-shaped API should be able
    to hand it straight back.
    """
    if raw is None:
        return sorted(DEFAULT_SCOPES)
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    cleaned = {value.strip() for value in values if value and value.strip()}
    unknown = cleaned - ALLOWED_SCOPES
    if unknown:
        raise ValueError(
            f"Unknown scope(s): {', '.join(sorted(unknown))}. "
            f"Expected any of: {', '.join(sorted(ALLOWED_SCOPES))}."
        )
    if not cleaned:
        raise ValueError(
            "A key needs at least one scope; a key that can do nothing is a "
            "credential that looks valid and is not."
        )
    return sorted(cleaned)


class OrganizationCreate(BaseModel):
    """A company to onboard.

    ``name`` is unique platform-wide, which makes it the natural human handle
    for "the org I already created" -- a client can test for the collision before
    sending instead of having to interpret a 409. ``slug`` is optional and
    derived from the name when omitted, because it exists for humans to read and
    nothing resolves against it.
    """

    name: str = Field(min_length=1, max_length=128, description="Unique display name")
    slug: str | None = Field(
        default=None, min_length=1, max_length=128, description="Optional URL-friendly handle"
    )
    is_active: bool = Field(default=True, description="Inactive orgs keep data but are not listed as onboardable")


class OrganizationOut(ORMModel):
    """One organization, as returned by every read.

    ``num_applications`` and ``num_api_keys`` are counts of rows that already
    exist, not figures recomputed from telemetry, so they are cheap and exact.

    Not ``model_validate``-able off a row: the two counts are aggregates the
    model has no column for. :func:`app.api.v1.organizations.to_out` builds one
    explicitly, which is why the mismatch is declared here rather than
    discovered at the first request.
    """

    id: uuid.UUID
    name: str
    slug: str | None = None
    is_active: bool = True
    num_applications: int = 0
    num_api_keys: int = 0
    created_at: datetime
    updated_at: datetime


class ApiKeyCreate(BaseModel):
    """A request to mint one key for an organization."""

    name: str = Field(min_length=1, max_length=128, description="Label for the key, e.g. 'CI'")
    scopes: list[str] | None = Field(
        default=None,
        description="Any of: ingest, read. Defaults to both.",
        validate_default=True,
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiry. A key with none never expires; revocation is the control.",
    )

    @field_validator("scopes", mode="before")
    @classmethod
    def _check_scopes(cls, value: object) -> list[str]:
        # Runs on the joined string as well as the list, because the schema also
        # accepts either and this is where both are canonicalised.
        #
        # ``validate_default=True`` is load-bearing and was not obvious: Pydantic
        # skips validators on fields the request never supplied, so without it an
        # omitted ``scopes`` left the field as the bare ``None`` default. The
        # route stores ``str(payload.scopes)``, which turned that into the four
        # literal characters ``"None"`` in the ``scopes`` column -- a key that
        # authenticated successfully and could do nothing at all, and reported
        # ``scopes: ["None"]`` in its own creation response.
        return _normalize_scopes(value)  # type: ignore[arg-type]


class ApiKeyOut(ORMModel):
    """One key's metadata. Deliberately has no plaintext field.

    ``is_active`` is derived rather than stored: it is ``revoked_at is None and
    expires_at is in the future``, and deriving it means an expired key stops
    reading as usable without a background job having to run to notice. That
    also makes this schema un-validatable from a row, for the same reason as
    :class:`OrganizationOut` -- the field has no column behind it.
    """

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    key_prefix: str = Field(description="First 12 characters; never enough to authenticate")
    scopes: list[str] = Field(default_factory=list)
    is_active: bool = True
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CreatedApiKey(ApiKeyOut):
    """A newly minted key, carrying its plaintext exactly once.

    ``key`` is populated from the value ``generate_api_key()`` returned and is
    never readable from the database afterwards. Losing it is not a bug to
    report: the operator who has not copied it revokes the key and mints
    another.
    """

    key: str = Field(description="Shown once. Store it now; it cannot be retrieved again.")


__all__ = [
    "ALLOWED_SCOPES",
    "DEFAULT_KEY_SCOPES",
    "DEFAULT_SCOPES",
    "ApiKeyCreate",
    "ApiKeyOut",
    "CreatedApiKey",
    "OrganizationCreate",
    "OrganizationOut",
]