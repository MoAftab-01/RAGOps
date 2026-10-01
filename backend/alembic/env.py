"""Alembic environment for RAGOps.

Two things are deliberate here:

* The database URL comes from ``app.config.get_settings()`` rather than
  ``alembic.ini``. Alembic therefore honours the same environment variables and
  the same ``.env`` file as the running application, and no credential is ever
  written into a tracked file.
* Online migrations use the *async* engine that the app itself uses
  (``DATABASE_URL``); the offline path emits plain SQL from the same metadata.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.models import Base  # noqa: F401  (imports register every table)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001, ARG001
    """Keep autogenerate focused on the application schema."""
    if type_ == "table" and name in {"alembic_version", "spatial_ref_sys"}:
        return False
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection (``alembic upgrade head --sql``)."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations through the async engine, then dispose the pool."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
