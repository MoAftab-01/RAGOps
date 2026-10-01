"""Organization and API key records — the company layer.

RAGOps is multi-tenant. An :class:`Organization` owns a set of applications and
the keys that may write telemetry for them; a key belongs to exactly one
organization and is the only thing that scopes a tenant request.

Two deliberate design points, both load-bearing:

**Organizations hang off the application, not off every table.**
``Application.organization_id`` is the single enforcement seam. ``traces``,
``llm_calls``, ``retrieval_calls``, ``spans``, ``users``, ``anomalies``,
``experiments``, ``evaluation_runs`` and ``documents`` all already carry an
``application_id`` and therefore all reach an organization *through* the
application. Adding ``organization_id`` to those nine tables would mean nine
backfills and nine more columns that can drift out of sync with the application
they hang off. One nullable column is the whole design.

**An API key is stored as a digest, never as text.** ``key_hash`` is
HMAC-SHA256 keyed by ``settings.api_key_pepper``; ``key_prefix`` is the first 12
characters, kept only so a list of keys is readable and a revoke button can say
which key it is revoking. The plaintext exists in exactly one place — the return
value of :func:`app.core.security.generate_api_key` — and is shown to the operator
once. There is no code path that can read it back out of the database.

Why no argon2/bcrypt/passlib: the key is 32 bytes from the OS CSPRNG (~256 bits),
so there is nothing to brute-force; slow hashing exists to make *low-entropy*
secrets expensive to guess, which does not apply. No salt column is needed
either — a password is looked up by user id, but a key is looked up *by its own
hash* on a unique index, so the digest is already the identity. This stays on the
stdlib for that reason, per §35 ("do NOT add libraries simply to make the stack
look impressive").
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

#: Characters of the plaintext retained for display. Long enough that two keys
#: in a list are distinguishable, far too short to be a usable credential.
KEY_PREFIX_LENGTH = 12

#: What a key may do when it carries no explicit scope list. ``ingest`` covers
#: the telemetry write path; ``read`` covers the tenant-scoped analytics.
DEFAULT_KEY_SCOPES = "ingest,read"


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A company, team, or any other grouping that owns telemetry."""

    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    slug: Mapped[str | None] = mapped_column(String(128), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ApiKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One credential belonging to exactly one organization.

    Revocation sets ``revoked_at`` rather than deleting the row. A revoked key
    keeps its ``created_at`` and ``last_used_at``, which is exactly what you
    want when someone later asks whether a leaked key was actually used.

    ``key_hash`` is unique and indexed because it is the lookup path: resolving
    a presented key is a single indexed probe, not a scan over stored digests.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_prefix", "key_prefix"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    #: Displayable head of the key, never enough to authenticate with.
    key_prefix: Mapped[str] = mapped_column(String(KEY_PREFIX_LENGTH), nullable=False)
    #: HMAC-SHA256 of the plaintext. The plaintext is not stored anywhere.
    key_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    scopes: Mapped[str] = mapped_column(String(64), default=DEFAULT_KEY_SCOPES, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
