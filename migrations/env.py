"""Alembic async environment. DB URL comes from app settings (env vars), never from alembic.ini."""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import app.models  # noqa: F401  (registers all tables on Base.metadata)
from app.config import get_settings
from app.database.base import Base

config = context.config
target_metadata = Base.metadata

# Tests can inject a URL via env; otherwise use application settings.
DATABASE_URL = os.environ.get("ALEMBIC_DATABASE_URL") or get_settings().database_url


def _include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    # PostGIS/tiger internals and views are not managed by autogenerate.
    if type_ == "table" and name in {"spatial_ref_sys"}:
        return False
    return True


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


run_migrations_online()