"""SQLAlchemy declarative base, shared column types and mixins."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit naming convention so Alembic can autogenerate deterministic,
# reversible constraint names instead of database-assigned ones.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for every ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # SQLAlchemy resolves an annotation by looking up the *parametrised* type
    # first and falling back to its origin, but the fallback only covers a
    # handful of builtins. Listing the concrete forms we use keeps every
    # JSONB column explicit and avoids per-column `mapped_column(JSONB)` noise.
    type_annotation_map = {
        dict: JSONB,
        dict[str, Any]: JSONB,
        list[str]: JSONB,
        list[Any]: JSONB,
        list[dict[str, Any]]: JSONB,
        list[float]: JSONB,
        uuid.UUID: PGUUID(as_uuid=True),
    }


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used instead of ``datetime.utcnow()`` so that
    stored timestamps are never naive."""
    return datetime.now(timezone.utc)


class UUIDPrimaryKeyMixin:
    """Surrogate UUID primary key."""

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    """``created_at`` / ``updated_at`` audit columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
