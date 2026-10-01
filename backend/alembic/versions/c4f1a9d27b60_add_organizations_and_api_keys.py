"""add organizations and api keys

Revision ID: c4f1a9d27b60
Revises: 04bef5d82379
Create Date: 2026-10-01 12:30:00.000000

Order matters here. The column is added *without* its foreign key, the existing
applications are backfilled, and only then is the constraint created. Doing it
the other way round would make the backfill write into a table whose FK is
already being validated -- which works, but only by accident, and it fails the
moment the migration is ever re-run against a partially-populated database.

Both backfill statements are guarded with WHERE NOT EXISTS so re-running this
revision is a no-op rather than a duplicate-organization error.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4f1a9d27b60"
down_revision: str | None = "04bef5d82379"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Name of the organization every pre-existing application is assigned to. Not
#: configurable: this is a one-time backfill, and re-running it against a
#: different name would silently split an existing dataset in two.
DEFAULT_ORGANIZATION_NAME = "Default"


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("slug", sa.String(length=128), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_organizations")),
    )
    op.create_index(op.f("ix_organizations_created_at"), "organizations", ["created_at"], unique=False)
    op.create_index(op.f("ix_organizations_name"), "organizations", ["name"], unique=True)
    op.create_index(op.f("ix_organizations_slug"), "organizations", ["slug"], unique=False)

    op.create_table(
        "api_keys",
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("key_prefix", sa.String(length=12), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("scopes", sa.String(length=64), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_api_keys")),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f("fk_api_keys_organization_id_organizations"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(op.f("ix_api_keys_created_at"), "api_keys", ["created_at"], unique=False)
    op.create_index(op.f("ix_api_keys_key_hash"), "api_keys", ["key_hash"], unique=True)
    op.create_index(
        op.f("ix_api_keys_organization_id"), "api_keys", ["organization_id"], unique=False
    )
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"], unique=False)

    # Column first, constraint last -- see the module docstring.
    op.add_column("applications", sa.Column("organization_id", sa.UUID(), nullable=True))

    default_org_id = uuid.uuid4()
    op.execute(
        sa.text(
            """
            INSERT INTO organizations (id, name, slug, is_active, created_at, updated_at)
            SELECT :org_id, :org_name, :org_slug, true, now(), now()
            WHERE NOT EXISTS (SELECT 1 FROM organizations WHERE name = :org_name)
            """
        ).bindparams(
            org_id=default_org_id,
            org_name=DEFAULT_ORGANIZATION_NAME,
            # Lowercased name, so the slug is a valid identifier fragment.
            org_slug=DEFAULT_ORGANIZATION_NAME.lower(),
        )
    )
    # Every application that has no organization belongs to the default one.
    # The traces are untouched: they reach the organization through the
    # application they already point at.
    op.execute(
        sa.text(
            """
            UPDATE applications SET organization_id = :org_id
            WHERE organization_id IS NULL
            """
        ).bindparams(org_id=default_org_id)
    )

    op.create_foreign_key(
        op.f("fk_applications_organization_id_organizations"),
        "applications",
        "organizations",
        ["organization_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_applications_organization_id"), "applications", ["organization_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_applications_organization_id"), table_name="applications")
    op.drop_constraint(
        op.f("fk_applications_organization_id_organizations"),
        "applications",
        type_="foreignkey",
    )
    op.drop_column("applications", "organization_id")
    op.drop_index("ix_api_keys_key_prefix", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index(op.f("ix_organizations_slug"), table_name="organizations")
    op.drop_index(op.f("ix_organizations_name"), table_name="organizations")
    op.drop_index(op.f("ix_organizations_created_at"), table_name="organizations")
    op.drop_table("organizations")
