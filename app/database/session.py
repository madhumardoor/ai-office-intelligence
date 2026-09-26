"""Async SQLAlchemy engine and session management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings

logger = logging.getLogger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the pooled async engine.

    - pool_pre_ping: transparently recover from dropped connections.
    - statement_timeout: server-side guard against runaway queries.
    """
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "statement_timeout": str(settings.db_statement_timeout_ms),
                "application_name": settings.app_name,
            }
        },
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory. expire_on_commit=False avoids lazy-load surprises in async code."""
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session and guarantee rollback on error / close on exit."""
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise